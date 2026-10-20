#!/usr/bin/env python3
"""CSV-worker: один постоянный backend модели и отдельный runtime для каждой строки."""

import csv
import fcntl
import importlib
import os
import re
import socket
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread
from urllib.request import Request, urlopen

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Запуск через `python runner/start.py` добавляет в sys.path только runner/, а нам нужны модули проекта.
sys.path.insert(0, str(ROOT))

from runner.metrics import read_metrics

FIELDS = [
    "job_id", "model", "backend", "task", "split", "sample_no", "feature",
    "feature_envs", "token_limit", "tokens_used", "status", "worker_id", "started_at", "heartbeat_at",
    "result", "tool_calls", "tool_timeouts",
    "task_time_s", "tool_time_s", "eval_log", "error",
]


def sandbox_for_backend(backend: str) -> str:
    """Возвращает явно объявленный backend-модулем тип sandbox."""
    module = importlib.import_module(f"runner.backends.{backend}")
    return module.sandbox()


def feature_environment(value: str) -> dict[str, str]:
    """Разбирает CSV-строку вида NAME=value,OTHER=value для одной job."""
    result: dict[str, str] = {}
    for item in filter(None, (part.strip() for part in value.split(","))):
        name, separator, setting = item.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(f"invalid feature_envs item: {item}")
        if name == "FEATURES":
            raise ValueError("feature_envs cannot override FEATURES")
        if name in result:
            raise ValueError(f"duplicate feature_envs variable: {name}")
        result[name] = setting
    return result


class Plan:
    """Безопасно распределяет и обновляет редактируемые строки CSV между worker'ами."""
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    @contextmanager
    def locked(self):
        # Worker'ы используют общий lock-файл, а атомарная запись не показывает читателям неполный CSV.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _read(self) -> list[dict[str, str]]:
        # Строгая проверка заголовка ловит ручное переименование до записи результата не в ту колонку.
        with self.path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"CSV columns must be: {','.join(FIELDS)}")
            rows = list(reader)
        ids = [row["job_id"] for row in rows]
        # Номера строк меняются при редактировании, поэтому стабильным ключом служит job_id.
        if not all(job_id.isascii() and job_id.isdecimal() for job_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("job_id must be numeric and unique")
        return rows

    def _write(self, rows: list[dict[str, str]]) -> None:
        # Атомарная замена не позволяет редактору увидеть частично записанный CSV.
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            shutil.copy2(self.path, self.path.with_suffix(self.path.suffix + ".bak"))
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def claim(self, model: str, backend: str, worker_id: str) -> dict[str, str] | None:
        # Строка становится running под тем же lock, поэтому два worker'а не заберут одну задачу.
        with self.locked():
            rows = self._read()
            for row in rows:
                if not row["status"] and row["model"] == model and row["backend"] == backend:
                    now = datetime.now(timezone.utc).isoformat()
                    row["status"] = "running"
                    row["worker_id"] = worker_id
                    row["started_at"] = now
                    row["heartbeat_at"] = now
                    self._write(rows)
                    return row.copy()
        return None

    def heartbeat(self, job_id: str, worker_id: str) -> None:
        # Проверка владельца не позволяет старому worker обновить уже перезапущенную задачу.
        with self.locked():
            rows = self._read()
            row = next((item for item in rows if item["job_id"] == job_id), None)
            if row and row["status"] == "running" and row["worker_id"] == worker_id:
                row["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
                self._write(rows)

    def recover_orphans(self, stale_after: int) -> int:
        # Без свежего heartbeat процесс считается погибшим, но задача не запускается повторно молча.
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after)
        recovered = 0
        with self.locked():
            rows = self._read()
            for row in rows:
                if row["status"] != "running":
                    continue
                value = row["heartbeat_at"] or row["started_at"]
                try:
                    alive_at = datetime.fromisoformat(value)
                except (TypeError, ValueError):
                    alive_at = datetime.min.replace(tzinfo=timezone.utc)
                if alive_at.tzinfo is None:
                    alive_at = alive_at.replace(tzinfo=timezone.utc)
                if alive_at < cutoff:
                    row.update(status="interrupted", result="interrupted", error="1")
                    recovered += 1
            if recovered:
                self._write(rows)
        return recovered

    def finish(self, job_id: str, values: dict[str, str]) -> None:
        # Повторное чтение сохраняет строки, которые пользователь добавил после начала задачи.
        with self.locked():
            rows = self._read()
            row = next((item for item in rows if item["job_id"] == job_id), None)
            if row is None:
                raise RuntimeError(f"job {job_id} disappeared from the plan")
            row.update(values)
            self._write(rows)


def _http_request(url: str, api_key: str, timeout: int = 10) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    urlopen(Request(url, headers=headers), timeout=timeout).close()


def wait_http(process: subprocess.Popen | None, url: str, api_key: str = "", timeout: int = 600) -> None:
    """Ждёт готовность сервиса и одновременно обнаруживает его преждевременное завершение."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"process exited with code {process.returncode}")
        try:
            _http_request(url, api_key, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"service did not become ready: {url}")


def stop(process: subprocess.Popen | None) -> None:
    """Мягко останавливает дочерний процесс, затем убивает его, чтобы не оставлять утечки."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()


def job_environment(
    row: dict[str, str], slot: int, model_base_url: str, model_api_key: str, scaffold_port_base: int,
    state: Path, runtime: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    features = [] if not row["feature"] or row["feature"] == "baseline" else [
        feature.strip() for feature in row["feature"].split(",")
    ]
    if any(re.fullmatch(r"[a-z0-9_]+", feature) is None for feature in features):
        raise ValueError(f"invalid feature sequence: {row['feature']}")
    if len(features) != len(set(features)):
        # Повтор одной stateful-фичи считаем опечаткой в плане.
        raise ValueError(f"duplicate feature: {row['feature']}")
    # Одна переменная сохраняет точный порядок middleware из строки CSV.
    env["FEATURES"] = ",".join(features)
    env.update(feature_environment(row["feature_envs"]))
    # У каждого параллельного слота свой scaffold и порт; backend модели остаётся общим.
    scaffold_port = scaffold_port_base + slot
    env.update({
        "PYTHONPATH": str(ROOT),
        "XDG_RUNTIME_DIR": str(runtime),
        "INSPECT_LOG_DIR": str(state / "inspect-logs"),
        "BASE_MODEL_API_BASE_URL": model_base_url,
        "BASE_MODEL_API_KEY": model_api_key,
        "BASE_MODEL_NAME": row["model"],
        "SCAFFOLD_PORT": str(scaffold_port),
        "SCAFFOLD_API_KEY": "local",
        "SCAFFOLD_MODEL_NAME": f"benchmark-job-{row['job_id']}",
        "BENCHMARK_MODEL_BASE_URL": f"http://127.0.0.1:{scaffold_port}/v1",
        "BENCHMARK_MODEL_API_KEY": "local",
    })
    return env


def run_job(
    plan: Plan, row: dict[str, str], worker_id: str, worker_logs: Path, slot: int,
    model_base_url: str, model_api_key: str, scaffold_port_base: int,
    heartbeat_interval: int, benchmark_runner: Path,
) -> None:
    # Runtime короткий из-за лимита AF_UNIX; диагностические файлы сразу пишутся в постоянные logs.
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    base = worker_logs / "jobs" / f"{row['job_id']}-{run_id}"
    state = base
    suffix = 2
    while state.exists():
        state = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    (state / "inspect-logs").mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix=f"benchmark-{row['job_id']}-"))
    processes: list[subprocess.Popen] = []
    heartbeat_stop = Event()
    heartbeat = Thread(
        target=lambda: _heartbeat_loop(plan, row["job_id"], worker_id, heartbeat_interval, heartbeat_stop),
        daemon=True,
    )
    heartbeat.start()
    started = time.monotonic()
    try:
        env = job_environment(
            row, slot, model_base_url, model_api_key, scaffold_port_base, state, runtime
        )
        # Tool-support хранит состояние браузера и сокета, поэтому каждой задаче нужен свой экземпляр.
        tool_log = (state / "tool-support.log").open("w")
        tool = subprocess.Popen([str(ROOT / ".venv/bin/inspect-tool-support"), "server"], env=env, stdout=tool_log, stderr=subprocess.STDOUT)
        processes.append(tool)
        socket = runtime / "inspect-tool-support.sock"
        for _ in range(60):
            if socket.exists():
                break
            if tool.poll() is not None:
                raise RuntimeError("inspect-tool-support exited")
            time.sleep(0.5)
        else:
            raise TimeoutError("inspect-tool-support did not become ready")

        # Scaffold отдельный: выбранные фичи и их состояние принадлежат одному sample.
        scaffold_log = (state / "scaffold.log").open("w")
        scaffold = subprocess.Popen([
            str(ROOT / ".venv/bin/python"), "-m", "uvicorn", "svc_scaffold.main:app",
            "--ws", "none", "--host", "127.0.0.1", "--port", env["SCAFFOLD_PORT"],
        ], cwd=ROOT, env=env, stdout=scaffold_log, stderr=subprocess.STDOUT)
        processes.append(scaffold)
        wait_http(scaffold, f"http://127.0.0.1:{env['SCAFFOLD_PORT']}/health", timeout=60)

        cmd = [
            str(ROOT / ".venv/bin/python"), str(benchmark_runner),
            "--task", row["task"],
            "--split", row["split"] or "validation",
            "--sample-no", row["sample_no"],
            "--model", env["SCAFFOLD_MODEL_NAME"],
            "--sandbox", sandbox_for_backend(row["backend"]),
            "--time-limit", os.environ.get("SAMPLE_TIME_LIMIT", "1800"),
            "--message-limit", os.environ.get("MESSAGE_LIMIT", "100"),
        ]
        # Пишем вывод в файлы задачи, чтобы параллельные запуски Inspect не перемешались в терминале.
        completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        (state / "inspect.stdout").write_text(completed.stdout, encoding="utf-8")
        (state / "inspect.stderr").write_text(completed.stderr, encoding="utf-8")
        logs = sorted((state / "inspect-logs").glob("*.eval"), key=lambda path: path.stat().st_mtime)
        if completed.returncode or not logs:
            raise RuntimeError(completed.stderr.strip()[-500:] or f"inspect exited with {completed.returncode}")
        values = read_metrics(logs[-1])
        values.update(status="done", task_time_s=f"{time.monotonic() - started:.3f}", eval_log=str(logs[-1]))
        plan.finish(row["job_id"], values)
    except Exception as error:
        # Ошибка остаётся видна в CSV; ручная очистка status явно ставит строку на повтор.
        # Полный текст лежит рядом с остальными логами, чтобы не раздувать одну строку CSV.
        (state / "error.log").write_text(str(error), encoding="utf-8")
        plan.finish(row["job_id"], {
            "status": "error", "result": "error", "task_time_s": f"{time.monotonic() - started:.3f}",
            "error": "1",
        })
    finally:
        heartbeat_stop.set()
        heartbeat.join()
        for process in reversed(processes):
            stop(process)
        shutil.rmtree(runtime, ignore_errors=True)


def _heartbeat_loop(
    plan: Plan, job_id: str, worker_id: str, interval: int, stopped: Event
) -> None:
    """Подтверждает жизнь задачи, пока аварийное завершение процесса не оборвёт обновления."""
    while not stopped.wait(interval):
        plan.heartbeat(job_id, worker_id)


def main() -> int:
    # Значения, переданные shell, приоритетнее локального файла конфигурации.
    load_dotenv(ROOT / ".env", override=False)
    cache = ROOT / "_state/cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("XDG_DATA_HOME", str(cache / "data"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("INSPECT_EVALS_CACHE_DIR", str(cache / "inspect_evals"))
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(cache / "playwright"))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(cache / "vllm"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "torchinductor"))
    # PLAN необязателен: обычный запуск всегда читает репозиторный plans/default.csv.
    plan_path = Path(os.environ.get("PLAN", ROOT / "plans/default.csv")).resolve()
    model = os.environ.get("MODEL", "")
    model_context_limit = os.environ.get("MODEL_CONTEXT_LIMIT", "")
    backend_name = os.environ.get("BACKEND", "llama_docker")
    benchmark = os.environ.get("BENCHMARK", "")
    parallel = int(os.environ.get("PARALLEL", "1"))
    gpu = os.environ.get("GPU", "")
    gpu_number = int(gpu) if gpu else 0
    # GPU влияет на порты по умолчанию, чтобы два worker'а не столкнулись на localhost.
    model_port = int(os.environ.get("MODEL_PORT", str(18080 + gpu_number)))
    default_scaffold_port = 29080 if backend_name == "external" else 19080 + gpu_number * 100
    scaffold_port_base = int(os.environ.get("SCAFFOLD_PORT_BASE", str(default_scaffold_port)))
    if not model or not model_context_limit or not benchmark or parallel < 1:
        raise SystemExit("MODEL, MODEL_CONTEXT_LIMIT and BENCHMARK are required; PARALLEL must be positive")
    if re.fullmatch(r"[a-z0-9_]+", benchmark) is None:
        raise SystemExit("BENCHMARK must contain only lowercase letters, digits and underscores")
    benchmark_runner = ROOT / "benchmark" / benchmark / "run.py"
    if not benchmark_runner.is_file():
        raise SystemExit(f"benchmark runner not found: {benchmark_runner}")
    if backend_name == "vllm" and not gpu:
        raise SystemExit("GPU is required for vllm")
    if not plan_path.is_file():
        raise SystemExit(f"plan not found: {plan_path}")
    for executable in (ROOT / ".venv/bin/inspect", ROOT / ".venv/bin/inspect-tool-support"):
        if not executable.is_file():
            raise SystemExit(f"missing {executable}; install the Inspect environment first")

    plan = Plan(plan_path)
    backend = importlib.import_module(f"runner.backends.{backend_name}")
    if backend_name == "external":
        if not (os.environ.get("MODEL_API_BASE_URL") or os.environ.get("BASE_MODEL_API_BASE_URL")):
            raise SystemExit("MODEL_API_BASE_URL is required for external backend")
        if not os.environ.get("MODEL_API_KEY"):
            raise SystemExit("MODEL_API_KEY is required for external backend")
    model_base_url = backend.base_url(model_port)
    model_api_key = backend.api_key()
    worker_name = (
        "worker-external-local"
        if backend_name == "external"
        else f"worker-{'gpu-' + gpu if gpu else backend_name + '-local'}"
    )
    # Lock имеет стабильный путь: уникальные каталоги запусков не защитили бы друг от друга.
    # Неблокирующий lock сразу сообщает о втором worker'е для того же GPU/backend.
    lock_file = (ROOT / "_state" / f"{worker_name}.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"worker already runs for {backend_name} GPU {gpu or 'local'}")

    worker_run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    base = ROOT / "_state/logs" / f"{worker_name}-{worker_run_id}"
    state = base
    suffix = 2
    while state.exists():
        state = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    state.mkdir(parents=True)

    stale_after = int(os.environ.get("WORKER_STALE_AFTER", "120"))
    if stale_after < 3:
        raise SystemExit("WORKER_STALE_AFTER must be at least 3 seconds")
    heartbeat_interval = min(30, stale_after // 3)
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{state.name}"
    recovered = plan.recover_orphans(stale_after)
    if recovered:
        print(f"Interrupted orphaned jobs: {recovered}")

    (ROOT / "_state/cache/llama").mkdir(parents=True, exist_ok=True)
    backend_log = (state / "backend.log").open("w")
    command = backend.command(model, model_port, parallel)
    process = subprocess.Popen(
        command, cwd=ROOT, env=os.environ | backend.environment(gpu),
        stdout=backend_log, stderr=subprocess.STDOUT,
    ) if command else None
    try:
        # Локальный backend нужно дождаться; внешний обнаруживается только реальными запросами модели.
        if process is not None:
            wait_http(process, f"{model_base_url}/models", model_api_key)
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            # Этот цикл и есть диспетчер: он забирает строку при освобождении слота.
            futures = set()
            while True:
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"model backend exited with code {process.returncode}")
                futures = {future for future in futures if not future.done()}
                while len(futures) < parallel:
                    row = plan.claim(model, backend_name, worker_id)
                    if row is None:
                        break
                    slot = next(slot for slot in range(parallel) if all(getattr(future, "slot", None) != slot for future in futures))
                    future = pool.submit(
                        run_job, plan, row, worker_id, state, slot, model_base_url,
                        model_api_key, scaffold_port_base, heartbeat_interval, benchmark_runner,
                    )
                    future.slot = slot
                    futures.add(future)
                if not futures:
                    break
                wait(futures, timeout=1)
    finally:
        stop(process)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
