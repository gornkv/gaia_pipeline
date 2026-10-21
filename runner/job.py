"""Полный цикл одной job: окружение, процессы, heartbeat и результат Inspect."""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any, TYPE_CHECKING
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from runner.worker import Plan

ROOT = Path(__file__).resolve().parent.parent


def sandbox_for_backend(backend: str) -> str:
    """Возвращает явно объявленный backend-модулем тип sandbox."""
    module = importlib.import_module(f"backends.{backend}")
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
        "JOB_STATE_DIR": str(state),
        "BASE_MODEL_API_BASE_URL": model_base_url,
        "BASE_MODEL_API_KEY": model_api_key,
        "BASE_MODEL_NAME": row["model"],
        "SCAFFOLD_API_KEY": "local",
        "SCAFFOLD_MODEL_NAME": f"benchmark-job-{row['job_id']}",
        "BENCHMARK_MODEL_BASE_URL": f"http://127.0.0.1:{scaffold_port}/v1",
        "BENCHMARK_MODEL_API_KEY": "local",
    })
    return env



def run_job(
    plan: Plan, row: dict[str, str], worker_id: str, worker_logs: Path, slot: int,
    model_base_url: str, model_api_key: str, scaffold_port_base: int,
    heartbeat_interval: int,
) -> dict[str, str]:
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
    started = time.monotonic()
    heartbeat = Thread(
        target=lambda: _heartbeat_loop(plan, row["job_id"], worker_id, heartbeat_interval, heartbeat_stop, state / "heartbeat-errors.log", started),
        daemon=True,
    )
    heartbeat.start()
    error_code = "job:internal"
    values = {}
    try:
        if re.fullmatch(r"[a-z0-9_]+", row["benchmark"]) is None:
            raise ValueError("invalid benchmark name")
        benchmark_runner = ROOT / "benchmark" / row["benchmark"] / "run.py"
        if not benchmark_runner.is_file():
            raise ValueError(f"benchmark runner not found: {benchmark_runner}")
        env = job_environment(
            row, slot, model_base_url, model_api_key, scaffold_port_base, state, runtime
        )
        # Tool-support хранит состояние браузера и сокета, поэтому каждой задаче нужен свой экземпляр.
        error_code = "tools:start"
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
        error_code = "scaffold:start"
        scaffold_log = (state / "scaffold.log").open("w")
        scaffold = subprocess.Popen([
            str(ROOT / ".venv/bin/python"), "-m", "uvicorn", "runner.scaffold:create_app", "--factory",
            "--ws", "none", "--host", "127.0.0.1", "--port", str(scaffold_port_base + slot),
        ], cwd=ROOT, env=env, stdout=scaffold_log, stderr=subprocess.STDOUT)
        processes.append(scaffold)
        wait_http(scaffold, f"http://127.0.0.1:{scaffold_port_base + slot}/health", timeout=60)
        avg_stats = json.loads((state / "avg.json").read_text())
        values["avg_job_time_limit"] = str(avg_stats["avg_job_time_limit"])
        avg_remaining_time = max(1, avg_stats["avg_job_time_limit"] - (time.monotonic() - started))

        error_code = "inspect:process"
        cmd = [
            str(ROOT / ".venv/bin/python"), str(benchmark_runner),
            "--task", row["task"],
            "--split", row["split"] or "validation",
            "--sample-no", row["sample_no"],
            "--model", env["SCAFFOLD_MODEL_NAME"],
            "--sandbox", sandbox_for_backend(row["backend"]),
            "--time-limit", str(math.ceil(avg_remaining_time)),
        ]
        # Пишем вывод в файлы задачи, чтобы параллельные запуски Inspect не перемешались в терминале.
        completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=avg_remaining_time)
        (state / "inspect.stdout").write_text(completed.stdout, encoding="utf-8")
        (state / "inspect.stderr").write_text(completed.stderr, encoding="utf-8")
        logs = sorted((state / "inspect-logs").glob("*.eval"), key=lambda path: path.stat().st_mtime)
        if completed.returncode or not logs:
            raise RuntimeError(completed.stderr.strip()[-500:] or f"inspect exited with {completed.returncode}")
        error_code = "job:internal"
        values.update(read_metrics(logs[-1]), status="done")
    except subprocess.TimeoutExpired as error:
        (state / "error.log").write_text(str(error), encoding="utf-8")
        values.update(status="done", result="timeout", error="job:timeout")
    except Exception as error:
        # Ошибка остаётся видна в CSV; ручная очистка status явно ставит строку на повтор.
        # Полный текст лежит рядом с остальными логами, чтобы не раздувать одну строку CSV.
        (state / "error.log").write_text(str(error), encoding="utf-8")
        values.update({
            "status": "error", "result": "error",
            "error": error_code,
            "worker_id": worker_id,
        })
    finally:
        heartbeat_stop.set()
        heartbeat.join()
        for process in reversed(processes):
            stop(process)
        shutil.rmtree(runtime, ignore_errors=True)
    values.update(worker_id=worker_id, full_time_sum=str(round(time.monotonic() - started)))
    if (state / "avg.json").exists():
        # Счётчики включают внутренние запросы фич и сохраняются до остановки scaffold.
        values["_avg"] = json.loads((state / "avg.json").read_text())
    return values



def _heartbeat_loop(
    plan: Plan, job_id: str, worker_id: str, interval: int, stopped: Event, error_log: Path, started: float
) -> None:
    """Подтверждает жизнь задачи, пока аварийное завершение процесса не оборвёт обновления."""
    while not stopped.wait(interval):
        try:
            path = error_log.parent / "avg.json"
            stats = json.loads(path.read_text()) if path.exists() else {}
            plan.heartbeat(job_id, worker_id, full_time_sum=str(round(time.monotonic() - started)),
                           max_input_tokens=str(stats.get("max_input_tokens", 0)))
        except Exception as error:
            # Временная ошибка сети не должна незаметно убить поток heartbeat навсегда.
            with error_log.open("a", encoding="utf-8") as log:
                log.write(f"{datetime.now(timezone.utc).isoformat()} {error}\n")



def _dump(value: Any) -> Any:
    # Inspect использует Pydantic events; словари упрощают доступ к полям разных версий.
    return value.model_dump() if hasattr(value, "model_dump") else value


def _seconds(start: Any, end: Any) -> float:
    """Считает неотрицательную длительность из datetime или ISO timestamp."""
    if not start or not end:
        return 0.0
    if isinstance(start, str):
        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if isinstance(end, str):
        end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return max(0.0, (end - start).total_seconds())


def _tool_stats(tools: list[dict[str, Any]], time_limit: float) -> tuple[float, int]:
    durations = [_seconds(event.get("timestamp"), event.get("completed")) for event in tools]
    return sum(durations), sum(duration > time_limit for duration in durations)


def _max_context_tokens(events: list[dict[str, Any]]) -> int:
    """Возвращает наибольший входной контекст, включая отклонённый backend запрос."""
    maximum = 0
    for event in events:
        if event.get("event") != "model":
            continue
        usage = (_dump(event.get("output")) or {}).get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        if not isinstance(prompt_tokens, int):
            input_tokens = usage.get("input_tokens") or 0
            cached_tokens = usage.get("input_tokens_cache_read") or 0
            prompt_tokens = input_tokens + cached_tokens
        maximum = max(maximum, prompt_tokens)
        error = str(event.get("error") or "")
        match = re.search(r"request \((\d+) tokens\).*context size", error)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum


def _is_context_overflow(events: list[dict[str, Any]]) -> bool:
    return any(
        event.get("event") == "model"
        and re.search(r"request \(\d+ tokens\).*context size", str(event.get("error") or ""))
        for event in events
    )


def read_metrics(path: Path) -> dict[str, str]:
    """Извлекает результат, токены и стоимость tools из закрытого eval-лога."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path))
    sample = log.samples[0]
    events = [_dump(event) for event in (sample.events or [])]
    tools = [event for event in events if event.get("event") == "tool"]
    # Это сумма времени tools, поэтому при параллельности она может превысить wall time задачи.
    tool_time_limit = float(os.getenv("TOOL_TIMEOUT", "180"))
    tool_time, tool_timeouts = _tool_stats(tools, tool_time_limit)
    max_context_tokens = _max_context_tokens(events)
    scores = [_dump(score).get("value") for score in (sample.scores or {}).values()]
    timed_out = any(
        event.get("event") == "sample_limit" and event.get("type") == "time" for event in events
    )
    return {
        # Ошибка выполнения отличается от корректно завершённого, но неверного ответа.
        "result": "context_overflow" if _is_context_overflow(events) else "timeout" if timed_out else "sample_error" if sample.error else "solved" if any(
            score == "C" or score is True or isinstance(score, (int, float)) and score > 0 for score in scores
        ) else "unsolved",
        "tool_calls": str(len(tools)),
        # Один ToolEvent — один вызов; тип ошибки не создаёт дополнительного инкремента.
        "tool_timeouts": str(tool_timeouts),
        "tool_time_sum": f"{tool_time:.2f}",
        "max_input_tokens": str(max_context_tokens),
        "_avg_tool_time": tool_time,
        # Подробности остаются в логах; источник внутри sample без догадок не уточняем.
        "error": "inspect:sample" if sample.error else "",
    }
