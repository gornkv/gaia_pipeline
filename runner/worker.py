#!/usr/bin/env python3
"""Worker общего Google-плана или локального CSV-снимка."""

import csv
import fcntl
import importlib
import os
import re
import secrets
import socket
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import requests

ROOT = Path(__file__).resolve().parent.parent
# Запуск через `python runner/worker.py` добавляет в sys.path только runner/, а нам нужны модули проекта.
sys.path.insert(0, str(ROOT))

from runner.job import run_job, stop, wait_http

FIELDS = [
    "benchmark", "task", "split", "sample_no", "backend", "model", "feature", "feature_envs",
    "worker_id", "job_id", "status", "result", "error", "max_input_tokens", "tool_calls",
    "tool_timeouts", "started_at", "heartbeat_at", "full_time_sum", "tool_time_sum",
    "avg_job_time_limit", "avg_tps", "avg_tool_count", "avg_tool_time", "jobs_count",
]


def job_id_for(row: dict[str, str], worker_id: str, existing: set[str]) -> str:
    value = row["job_id"]
    if not value:
        while not value or value in existing:
            value = secrets.token_hex(4)
    elif row["worker_id"] == worker_id:
        base, separator, suffix = value.rpartition("-")
        value = f"{base}-{int(suffix) + 1}" if separator else f"{value}-2"
        while value in existing:
            base, _, suffix = value.rpartition("-")
            value = f"{base}-{int(suffix) + 1}"
    return value


def avg_columns(avg: dict, row: dict) -> dict[str, str]:
    stats = row.pop("_avg", {})
    avg_tool_time = row.pop("_avg_tool_time", 0.0)
    if row["result"] in {"solved", "unsolved"} and not row["error"] and stats.get("avg_usage_complete"):
        # Это отношение пикового входного контекста ко времени модели, не скорость генерации.
        avg["tokens"] += int(row["max_input_tokens"])
        avg["time"] += stats["avg_response_time"]
        avg["tools"] += int(row["tool_calls"])
        avg["tool_time"] += avg_tool_time
        avg["jobs"] += 1
    return {
        "avg_tps": f"{avg['tokens'] / avg['time']:.2f}" if avg["time"] else "",
        "avg_tool_count": f"{avg['tools'] / avg['jobs']:.2f}" if avg["jobs"] else "",
        "avg_tool_time": f"{avg['tool_time'] / avg['tools']:.2f}" if avg["tools"] else "",
        "jobs_count": str(avg["jobs"]),
    }


@contextmanager
def reserve_ports(parallel: int):
    """Закрепляет соседние пары 18000+i/19000+i за worker на всё время работы."""
    if not 1 <= parallel <= 100:
        raise ValueError("PARALLEL must be between 1 and 100")
    directory = ROOT / "_state/ports"
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as reserved:
        # Курсор и его файл не удаляем: все процессы должны блокировать один и тот же inode.
        with (directory / "cursor").open("a+") as cursor:
            fcntl.flock(cursor, fcntl.LOCK_EX)
            cursor.seek(0)
            start = int(cursor.read() or "0") % 100
            for step in range(100):
                offset = (start + step) % 100
                if offset + parallel > 100:
                    continue
                try:
                    with ExitStack() as candidate:
                        for index in range(offset, offset + parallel):
                            lock = candidate.enter_context((directory / f"{index}.lock").open("a+"))
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            for port in (18000 + index, 19000 + index):
                                with socket.socket() as connection:
                                    connection.bind(("127.0.0.1", port))
                        reserved.enter_context(candidate.pop_all())
                    break
                except OSError:
                    # Частично захваченный блок освобождается перед следующей попыткой.
                    continue
            else:
                raise RuntimeError("No free model/scaffold port block (100 slots)")
            cursor.seek(0)
            cursor.truncate()
            cursor.write(str((offset + parallel) % 100))
            cursor.flush()
        # Блокировки пар остаются захваченными, даже когда scaffold временно остановлен.
        yield 18000 + offset, 19000 + offset


class Plan:
    """Google-план либо однократно загруженная локальная очередь CSV."""
    def __init__(self, location: str | Path):
        self.remote = str(location).startswith("https://")
        if self.remote:
            self.url = os.environ.get("PLAN_API_URL") or str(location)
            if not self.url.startswith("https://script.google.com/macros/s/"):
                raise ValueError("Google Sheets PLAN requires PLAN_API_URL pointing to the deployed Apps Script /exec URL")
            self.token = os.environ["PLAN_GOOGLE_SHEETS_TOKEN"]
        else:
            with Path(location).open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != FIELDS:
                    raise ValueError(f"CSV columns must be: {','.join(FIELDS)}")
                self.rows = list(reader)
            ids = [row["job_id"] for row in self.rows if row["job_id"]]
            if not all(re.fullmatch(r"[a-zA-Z0-9]+(?:-[0-9]+)?", value) for value in ids) or len(ids) != len(set(ids)):
                raise ValueError("job_id must be alphanumeric and unique when nonempty")
            self.ids = set(ids)

    def _call(self, action: str, **values):
        # Выдачу job не повторяем вслепую: потеряться мог только ответ, а не сама запись.
        response = requests.post(self.url, json={"token": self.token, "action": action, **values}, timeout=60)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Google Sheets plan: {data.get('error', 'invalid response')}")
        return data["result"]

    def claim(self, model: str, backend: str, worker_id: str) -> dict[str, str] | None:
        if self.remote:
            return self._call("claim", model=model, backend=backend, worker_id=worker_id)
        for row in self.rows:
            if row["status"] not in {"done", "error"} and row["model"] == model and row["backend"] == backend:
                self.rows.remove(row)
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                job_id = job_id_for(row, worker_id, self.ids)
                self.ids.add(job_id)
                row = {key: row[key] for key in FIELDS[:8]} | dict.fromkeys(FIELDS[8:], "") | {"job_id": job_id}
                row.update(status="running", worker_id=worker_id, started_at=now, heartbeat_at=now)
                return row.copy()
        return None

    def heartbeat(self, job_id: str, worker_id: str, **values) -> None:
        if self.remote:
            self._call("heartbeat", job_id=job_id, worker_id=worker_id, values=values)


def run_worker(resources: ExitStack) -> int:
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
    plan_location = os.environ.get("PLAN") or os.environ.get("PLAN_API_URL")
    if not plan_location:
        raise SystemExit("Set PLAN to a Google Sheets URL or CSV path, or set PLAN_API_URL")
    plan = Plan(plan_location)
    model = os.environ.get("MODEL", "")
    model_context_limit = os.environ.get("MODEL_CONTEXT_LIMIT", "")
    backend_name = os.environ.setdefault("MODEL_BACKEND", "llama_docker")
    parallel = int(os.environ.get("PARALLEL", "1"))
    gpu = os.environ.get("GPU", "")
    if not model or not model_context_limit or parallel < 1:
        raise SystemExit("MODEL and MODEL_CONTEXT_LIMIT are required; PARALLEL must be positive")
    if backend_name == "vllm" and not gpu:
        raise SystemExit("GPU is required for vllm")
    for executable in (ROOT / ".venv/bin/inspect", ROOT / ".venv/bin/inspect-tool-support"):
        if not executable.is_file():
            raise SystemExit(f"missing {executable}; install the Inspect environment first")

    backend = importlib.import_module(f"backends.{backend_name}")
    model_port, scaffold_port_base = resources.enter_context(reserve_ports(parallel))
    model_base_url = f"http://127.0.0.1:{model_port}/v1"
    model_api_key = "local"
    if gpu:
        # Стабильный lock защищает GPU от второго worker независимо от выбранного backend.
        lock_file = (ROOT / "_state" / f"gpu-{gpu}.lock").open("w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"worker already runs for GPU {gpu}")

    # Короткое имя одинаково в CSV и каталоге логов; дата и время до минут записаны в UTC.
    worker_name = os.environ.get("WORKER_NAME") or socket.gethostname()[:5]
    worker_id = f"{worker_name}-{gpu or backend_name}-{datetime.now(timezone.utc):%y%m%d%H%M}"
    base = ROOT / "_state/logs" / worker_id
    state = base
    suffix = 2
    while state.exists():
        state = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    state.mkdir(parents=True)
    worker_id = state.name

    stale_after = int(os.environ.get("WORKER_STALE_AFTER", "120"))
    if stale_after < 3:
        raise SystemExit("WORKER_STALE_AFTER must be at least 3 seconds")
    heartbeat_interval = min(30, stale_after // 3)
    recovered = plan._call("recover_orphans", stale_after=stale_after) if plan.remote else 0
    if recovered:
        print(f"Interrupted orphaned jobs: {recovered}")

    (ROOT / "_state/cache/llama").mkdir(parents=True, exist_ok=True)
    results = resources.enter_context((state / "results.csv").open("a", newline="", encoding="utf-8"))
    writer = csv.DictWriter(results, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    results.flush()
    sync_failed = False
    avg = dict.fromkeys(("tokens", "time", "tools", "tool_time", "jobs"), 0)
    backend_log = resources.enter_context((state / "backend.log").open("w"))
    command = backend.command(model, model_port, parallel)
    process = subprocess.Popen(
        command, cwd=ROOT,
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
                for future in list(futures):
                    if not future.done():
                        continue
                    row = future.row | future.result()
                    row["heartbeat_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    row.update(avg_columns(avg, row))
                    # Сначала сохраняем результат на диск, потом пытаемся обновить Google.
                    writer.writerow(row)
                    results.flush()
                    os.fsync(results.fileno())
                    futures.remove(future)
                    if plan.remote:
                        try:
                            plan._call("finish", job_id=row["job_id"], values={key: row[key] for key in FIELDS[8:]})
                        except Exception as error:
                            with (state / "sync-errors.log").open("a", encoding="utf-8") as log:
                                log.write(f"job {row['job_id']}: {error}\n")
                            sync_failed = True
                # При сбое записи Google завершаем активные job, но новые не берём.
                while not sync_failed and len(futures) < parallel:
                    try:
                        row = plan.claim(model, backend_name, worker_id)
                    except Exception as error:
                        with (state / "sync-errors.log").open("a", encoding="utf-8") as log:
                            log.write(f"claim: {error}\n")
                        sync_failed = True
                        break
                    if row is None:
                        break
                    slot = next(slot for slot in range(parallel) if all(getattr(future, "slot", None) != slot for future in futures))
                    future = pool.submit(
                        run_job, plan, row, worker_id, state, slot, model_base_url,
                        model_api_key, scaffold_port_base, heartbeat_interval,
                    )
                    future.row = row
                    future.slot = slot
                    futures.add(future)
                if not futures:
                    break
                wait(futures, timeout=1)
    finally:
        stop(process)
    return 1 if sync_failed else 0


def main() -> int:
    # Освобождаем резервирование и при ошибке подготовки, и после остановки всех процессов.
    with ExitStack() as resources:
        return run_worker(resources)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
