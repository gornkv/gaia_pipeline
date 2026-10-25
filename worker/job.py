from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Callable
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent


def feature_environment(value: str) -> dict[str, str]:
    result: dict[str, str] = {}


    for item in filter(None, (part.strip() for part in value.split(","))):
        name, separator, setting = item.partition("=")
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(f"invalid feature_envs item: {item}")
        if name == "SCAFFOLD_FEATURES":

            raise ValueError("feature_envs cannot override SCAFFOLD_FEATURES")
        if name in result:
            raise ValueError(f"duplicate feature_envs variable: {name}")
        result[name] = setting
    return result


def _http_request(url: str, api_key: str, timeout: int = 10) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    urlopen(Request(url, headers=headers), timeout=timeout).close()


def wait_http(process: subprocess.Popen | None, url: str, api_key: str = "", timeout: int = 600) -> None:


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
    if process is None or process.poll() is not None:
        return
    process.terminate()


    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def job_environment(
    row: dict[str, str], slot: int, scaffold_port_base: int,
    logs: Path, runtime: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    features = [] if not row["feature"] or row["feature"] == "baseline" else [
        feature.strip() for feature in row["feature"].split(",")
    ]
    if any(re.fullmatch(r"[a-z0-9_]+", feature) is None for feature in features):
        raise ValueError(f"invalid feature sequence: {row['feature']}")
    if len(features) != len(set(features)):

        raise ValueError(f"duplicate feature: {row['feature']}")

    env["SCAFFOLD_FEATURES"] = ",".join(features)
    env.update(feature_environment(row["feature_envs"]))

    if env["WORKER_TOOL_SANDBOX"] not in {"local", "docker"}:
        raise ValueError("WORKER_TOOL_SANDBOX must be local or docker")

    scaffold_port = scaffold_port_base + slot
    env.update({

        "PATH": str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", os.defpath),
        "PYTHONPATH": str(ROOT),
        "XDG_RUNTIME_DIR": str(runtime),
        "INSPECT_LOG_DIR": str(logs / "inspect-logs"),
        "INSPECT_TRACE_FILE": str(logs / "inspect-trace.log"),
        "BENCHMARK_MODEL_BASE_URL": f"http://127.0.0.1:{scaffold_port}/v1",
    })
    return env


@contextmanager
def job_directories(job_id: str, worker_id: str, worker_logs: Path):
    runtime_root = ROOT / "_state/runtime/jobs" / worker_id
    runtime_root.mkdir(parents=True, exist_ok=True)


    with tempfile.TemporaryDirectory(prefix=f"{job_id}-", dir=runtime_root, ignore_cleanup_errors=True) as runtime:
        logs = worker_logs / "jobs" / Path(runtime).name
        (logs / "inspect-logs").mkdir(parents=True, exist_ok=True)


        with tempfile.TemporaryDirectory(prefix=f"benchmark-{job_id}-", ignore_cleanup_errors=True) as socket_runtime:
            yield logs, Path(socket_runtime)


def job_snapshot(job: dict) -> dict:
    snapshot = {"job": job["report"] | {"full_time_seconds": time.monotonic() - job["started"]}}
    for component in ("benchmark", "scaffold", "calls"):
        path = job["logs"] / f"{component}.json"


        snapshot[component] = json.loads(path.read_text()) if path.exists() else None
    return snapshot


def send_heartbeats(job: dict, stopped: Event, interval: float, on_progress: Callable) -> None:


    while not stopped.wait(interval):
        on_progress(job_snapshot(job))


def start_tools(env: dict, logs: Path, processes: list) -> None:
    with (logs / "tool-support.log").open("w") as output:
        process = subprocess.Popen(
            [str(ROOT / ".venv/bin/inspect-tool-support"), "server"],
            env=env, stdout=output, stderr=subprocess.STDOUT,
        )


    processes.append(process)
    for _ in range(60):
        if (Path(env["XDG_RUNTIME_DIR"]) / "inspect-tool-support.sock").exists():
            return
        if process.poll() is not None:
            raise RuntimeError("inspect-tool-support exited")
        time.sleep(0.5)
    raise TimeoutError("inspect-tool-support did not become ready")


def start_scaffold(env: dict, logs: Path, port: int, processes: list) -> None:
    with (logs / "scaffold.log").open("w") as output:
        process = subprocess.Popen([
            str(ROOT / ".venv/bin/python"), "-m", f"scaffolds.{env['WORKER_SCAFFOLD']}", str(port), str(logs),
        ], cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    processes.append(process)

    wait_http(process, f"http://127.0.0.1:{port}/health", timeout=60)


def run_benchmark(row: dict, env: dict, logs: Path, remaining: float) -> None:
    command = [
        str(ROOT / ".venv/bin/python"), str(ROOT / "benchmarks" / row["benchmark"] / "run.py"),
        "--task", row["task"], "--split", row["split"],
        "--sample-no", row["sample_no"], "--model", env["WORKER_MODEL"],
        "--sandbox", env["WORKER_TOOL_SANDBOX"], "--time-limit", str(math.ceil(remaining)),
        "--report-file", str(logs / "benchmark.json"),
        "--calls-file", str(logs / "calls.json"),
    ]


    with (logs / "inspect.stdout").open("w") as stdout, (logs / "inspect.stderr").open("w") as stderr:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=remaining, check=True)


def run_job(
    row: dict[str, str], worker_id: str, worker_logs: Path, slot: int,
    scaffold_port_base: int,
    heartbeat_interval: float, on_progress: Callable,
) -> dict:
    with job_directories(row["job_id"], worker_id, worker_logs) as (logs, runtime):


        job = {
            "logs": logs, "started": time.monotonic(),
            "report": {"job_id": row["job_id"], "worker_id": worker_id, "status": "running",
                       "error": None, "time_limit_seconds": None},
        }
        stopped = Event()
        heartbeat = Thread(target=send_heartbeats, args=(job, stopped, heartbeat_interval, on_progress), daemon=True)
        heartbeat.start()
        processes = []
        error_code = "job:internal"
        final_status, final_error = "done", None
        try:
            if re.fullmatch(r"[a-z0-9_]+", row["benchmark"]) is None:
                raise ValueError("invalid benchmark name")
            env = job_environment(row, slot, scaffold_port_base, logs, runtime)

            error_code = "tools:start"
            start_tools(env, logs, processes)
            error_code = "scaffold:start"
            start_scaffold(env, logs, scaffold_port_base + slot, processes)
            job["report"]["time_limit_seconds"] = job_snapshot(job)["scaffold"]["time_limit_seconds"]

            remaining = max(1, job["report"]["time_limit_seconds"] - (time.monotonic() - job["started"]))
            error_code = "benchmark:process"
            run_benchmark(row, env, logs, remaining)
        except subprocess.TimeoutExpired as error:


            (logs / "error.log").write_text(str(error), encoding="utf-8")
            final_error = "job:timeout"
        except subprocess.CalledProcessError as error:
            (logs / "error.log").write_text(str(error), encoding="utf-8")
            final_status = "error"
            final_error = f"benchmark:{signal.Signals(-error.returncode).name.lower()}" if error.returncode < 0 else f"benchmark:exit_{error.returncode}"
        except Exception as error:
            (logs / "error.log").write_text(str(error), encoding="utf-8")
            final_status, final_error = "error", error_code
        finally:


            stopped.set()
            heartbeat.join()
            for process in reversed(processes):
                stop(process)
        job["report"].update(status=final_status, error=final_error)

        return job_snapshot(job)
