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
import uuid
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
    "token_limit", "tokens_used", "status", "worker_id", "started_at", "heartbeat_at",
    "result", "tool_calls", "tool_timeouts",
    "task_time_s", "tool_time_s", "eval_log", "error",
]


def sandbox_for_backend(backend: str) -> str:
    """Оставляет Docker-backend в контейнере, а инструменты нативного backend — локально."""
    return "docker" if backend.endswith("docker") else "local"


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


def wait_http(process: subprocess.Popen, url: str, timeout: int = 600) -> None:
    """Ждёт готовность сервиса и одновременно обнаруживает его преждевременное завершение."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited with code {process.returncode}")
        try:
            urlopen(Request(url), timeout=2).close()
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
    row: dict[str, str], slot: int, model_port: int, scaffold_port_base: int, state: Path
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
    # У каждого параллельного слота свой scaffold и порт; backend модели остаётся общим.
    scaffold_port = scaffold_port_base + slot
    env.update({
        "PYTHONPATH": str(ROOT),
        "XDG_RUNTIME_DIR": str(state / "runtime"),
        "INSPECT_LOG_DIR": str(state / "inspect-logs"),
        "BASE_MODEL_API_BASE_URL": f"http://127.0.0.1:{model_port}/v1",
        "BASE_MODEL_API_KEY": os.environ.get("BASE_MODEL_API_KEY", "local"),
        "BASE_MODEL_NAME": row["model"],
        "SCAFFOLD_PORT": str(scaffold_port),
        "SCAFFOLD_API_KEY": "local",
        "SCAFFOLD_MODEL_NAME": f"gaia-job-{row['job_id']}",
        "GAIA_MODEL_BASE_URL": f"http://127.0.0.1:{scaffold_port}/v1",
        "GAIA_MODEL_API_KEY": "local",
        "GAIA_MODEL_NAME": f"gaia-job-{row['job_id']}",
    })
    return env


def run_job(
    plan: Plan, row: dict[str, str], worker_id: str, slot: int, model_port: int,
    scaffold_port_base: int, heartbeat_interval: int,
) -> None:
    # Каталог job_id изолирует браузерный runtime, сокет tool-support и логи конкретной задачи.
    state = ROOT / "_state" / "jobs" / row["job_id"]
    (state / "runtime").mkdir(parents=True, exist_ok=True)
    (state / "inspect-logs").mkdir(parents=True, exist_ok=True)
    env = job_environment(row, slot, model_port, scaffold_port_base, state)
    processes: list[subprocess.Popen] = []
    heartbeat_stop = Event()
    heartbeat = Thread(
        target=lambda: _heartbeat_loop(plan, row["job_id"], worker_id, heartbeat_interval, heartbeat_stop),
        daemon=True,
    )
    heartbeat.start()
    started = time.monotonic()
    try:
        # Tool-support хранит состояние браузера и сокета, поэтому каждой задаче нужен свой экземпляр.
        tool_log = (state / "tool-support.log").open("w")
        tool = subprocess.Popen([str(ROOT / ".venv/bin/inspect-tool-support"), "server"], env=env, stdout=tool_log, stderr=subprocess.STDOUT)
        processes.append(tool)
        socket = state / "runtime" / "inspect-tool-support.sock"
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
        wait_http(scaffold, f"http://127.0.0.1:{env['SCAFFOLD_PORT']}/health", 60)

        task = row["task"]
        if not task.startswith("gaia"):
            raise ValueError("this first worker version supports GAIA tasks only")
        cmd = [
            str(ROOT / ".venv/bin/inspect"), "eval", f"runner/custom_gaia.py@{task}",
            "--model", f"openai-api/gaia_model/{env['SCAFFOLD_MODEL_NAME']}",
            # Browser/sandbox следует backend: Docker остаётся в контейнере, нативный vLLM — локально.
            "--sandbox", sandbox_for_backend(row["backend"]),
            "--max-connections", "1", "--limit", f"{row['sample_no']}-{row['sample_no']}",
            "--time-limit", os.environ.get("GAIA_SAMPLE_TIME_LIMIT", "1800"),
            "--message-limit", os.environ.get("GAIA_MESSAGE_LIMIT", "100"),
            "-T", f"split={row['split'] or 'validation'}",
            "-T", f"max_attempts={os.environ.get('GAIA_MAX_ATTEMPTS', '1')}",
            "-T", f"code_timeout={os.environ.get('GAIA_TOOL_TIMEOUT', '180')}",
            "--no-fail-on-error",
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


def _heartbeat_loop(
    plan: Plan, job_id: str, worker_id: str, interval: int, stopped: Event
) -> None:
    """Подтверждает жизнь задачи, пока аварийное завершение процесса не оборвёт обновления."""
    while not stopped.wait(interval):
        plan.heartbeat(job_id, worker_id)


def main() -> int:
    # Значения, переданные shell, приоритетнее локального файла конфигурации.
    load_dotenv(ROOT / ".env", override=False)
    # PLAN необязателен: обычный запуск всегда читает репозиторный plans/default.csv.
    plan_path = Path(os.environ.get("PLAN", ROOT / "plans/default.csv")).resolve()
    model = os.environ.get("MODEL", "")
    model_context_limit = os.environ.get("MODEL_CONTEXT_LIMIT", "")
    backend_name = os.environ.get("BACKEND", "llama_docker")
    parallel = int(os.environ.get("PARALLEL", "1"))
    gpu = os.environ.get("GPU", "")
    gpu_number = int(gpu) if gpu else 0
    # GPU влияет на порты по умолчанию, чтобы два worker'а не столкнулись на localhost.
    model_port = int(os.environ.get("MODEL_PORT", str(18080 + gpu_number)))
    scaffold_port_base = int(os.environ.get("SCAFFOLD_PORT_BASE", str(19080 + gpu_number * 100)))
    if not model or not model_context_limit or parallel < 1:
        raise SystemExit("MODEL and MODEL_CONTEXT_LIMIT are required; PARALLEL must be positive")
    if backend_name == "vllm" and not gpu:
        raise SystemExit("GPU is required for vllm")
    if not plan_path.is_file():
        raise SystemExit(f"plan not found: {plan_path}")
    for executable in (ROOT / ".venv/bin/inspect", ROOT / ".venv/bin/inspect-tool-support"):
        if not executable.is_file():
            raise SystemExit(f"missing {executable}; install the Inspect environment first")

    plan = Plan(plan_path)
    backend = importlib.import_module(f"runner.backends.{backend_name}")
    # GPU-lock общий для всех backend-типов: на одной карте разрешён ровно один worker.
    state = ROOT / "_state" / f"worker-{'gpu-' + gpu if gpu else backend_name + '-local'}"
    state.mkdir(parents=True, exist_ok=True)
    # Неблокирующий lock сразу сообщает о втором worker'е для того же GPU/backend.
    lock_file = (state / "worker.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"worker already runs for {backend_name} GPU {gpu or 'local'}")

    stale_after = int(os.environ.get("WORKER_STALE_AFTER", "120"))
    if stale_after < 3:
        raise SystemExit("WORKER_STALE_AFTER must be at least 3 seconds")
    heartbeat_interval = min(30, stale_after // 3)
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    recovered = plan.recover_orphans(stale_after)
    if recovered:
        print(f"Interrupted orphaned jobs: {recovered}")

    (ROOT / "_state/llama-cache").mkdir(parents=True, exist_ok=True)
    backend_log = (state / "backend.log").open("w")
    process = subprocess.Popen(
        backend.command(model, model_port, parallel), cwd=ROOT,
        env=os.environ | backend.environment(gpu), stdout=backend_log, stderr=subprocess.STDOUT,
    )
    try:
        # Fail-fast: backend обязан стартовать с запрошенным PARALLEL, автоматического снижения нет.
        wait_http(process, f"http://127.0.0.1:{model_port}/v1/models")
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            # Этот цикл и есть диспетчер: он забирает строку при освобождении слота.
            futures = set()
            while True:
                futures = {future for future in futures if not future.done()}
                while len(futures) < parallel:
                    row = plan.claim(model, backend_name, worker_id)
                    if row is None:
                        break
                    slot = next(slot for slot in range(parallel) if all(getattr(future, "slot", None) != slot for future in futures))
                    future = pool.submit(
                        run_job, plan, row, worker_id, slot, model_port,
                        scaffold_port_base, heartbeat_interval,
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
