import csv
import fcntl
import json
import logging
import os
import shutil
import socket
import signal
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

import requests
import psutil
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PORT_RANGE = (18000, 18999)

sys.path.insert(0, str(ROOT))

from worker.job import run_job, stop, wait_http, stop_token_processes


FIELDS = [
    "benchmark", "index", "model", "feature", "feature_envs", "experiment_id",
    "worker_id", "job_id", "status", "score", "result", "error", "max_input_tokens", "tool_calls",
    "tool_timeouts", "started_at", "full_time_sum", "tool_time_sum",
    "tps", "avg_tps", "avg_job_time_limit", "avg_tool_count", "avg_tool_time", "jobs_count",
]


def load_config() -> None:

    os.environ["WORKER_ROOT"] = str(ROOT)
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / "worker/.env.default", override=False)
    scaffold = f"scaffolds/{os.environ['WORKER_SCAFFOLD']}"
    for component in (f"backends/{os.environ['WORKER_BACKEND']}", "benchmarks", "scaffolds", scaffold, f"{scaffold}/features"):
        load_dotenv(ROOT / component / ".env.default", override=False)
    for path in sorted((ROOT / "benchmarks").glob("*/.env.default")):
        load_dotenv(path, override=False)


def snapshot_columns(snapshot: dict) -> dict:
    job = snapshot["job"]
    benchmark = snapshot["benchmark"]
    scaffold = snapshot["scaffold"]
    calls = snapshot["calls"]


    durations = [] if calls is None else [call["duration_seconds"] for call in calls if call["duration_seconds"] is not None]
    max_input = None if scaffold is None else scaffold["model"]["max_input_tokens"]
    intervals = []
    for call in calls or []:
        start = datetime.fromisoformat(call["started_at"].replace("Z", "+00:00")).timestamp()
        end = job["timestamp"] if call["duration_seconds"] is None else start + call["duration_seconds"]
        intervals.append((start, min(end, job["timestamp"])))
    tool_wall_time, previous_end = 0.0, float("-inf")
    for start, end in sorted(intervals):
        tool_wall_time += max(0.0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    non_tool_time = job["full_time_seconds"] - tool_wall_time
    score = benchmark.get("score") if benchmark is not None else None
    if score is None and benchmark is not None and benchmark["result"] in {"solved", "unsolved"}:
        score = int(benchmark["result"] == "solved")
    error = job["error"] or (benchmark["error"] if benchmark is not None else None) or ""
    status = "error" if job["status"] == "done" and error else job["status"]
    return {
        "status": status,
        "score": "" if score is None else score,
        "result": "interrupted" if job["status"] == "interrupted" else "timeout" if job["error"] == "job:timeout" else "error" if job["status"] == "error" else "" if job["status"] == "running" and benchmark is None else benchmark["result"] or "",
        "error": error,
        "max_input_tokens": "" if max_input is None else str(max_input),
        "tps": f"{max_input / non_tool_time:.2f}" if max_input is not None and calls is not None and non_tool_time > 0 else "",
        "tool_calls": "" if calls is None else str(len(calls)),


        "tool_timeouts": "" if calls is None else str(sum(call["duration_seconds"] is not None and call["duration_seconds"] > call["time_limit_seconds"] for call in calls)),
        "tool_time_sum": "" if calls is None else f"{sum(durations):.2f}",
        "full_time_sum": str(round(job["full_time_seconds"])),
        "avg_job_time_limit": "" if job["time_limit_seconds"] is None else str(job["time_limit_seconds"]),
    }


def avg_columns(avg: dict, snapshot: dict) -> dict[str, str]:
    model = snapshot["scaffold"]["model"] if snapshot["scaffold"] is not None else None
    if model is not None and snapshot["benchmark"] is not None:
        benchmark = snapshot["benchmark"]


        if snapshot["job"]["status"] == "done" and not snapshot["job"]["error"] and not benchmark["error"] and benchmark["result"] in {"solved", "unsolved"} and model["usage_complete"] and model["max_input_tokens"] is not None:
            avg["tools"] += len(snapshot["calls"])
            avg["tool_time"] += sum(call["duration_seconds"] for call in snapshot["calls"] if call["duration_seconds"] is not None)
            avg["jobs"] += 1


    return {
        "avg_tool_count": f"{avg['tools'] / avg['jobs']:.2f}" if avg["jobs"] else "",
        "avg_tool_time": f"{avg['tool_time'] / avg['tools']:.2f}" if avg["tools"] else "",
        "jobs_count": str(avg["jobs"]),
    }


@contextmanager
def reserve_ports(parallel: int):
    first, last = PORT_RANGE
    size = last - first + 1
    directory = ROOT / "_state/runtime/workers/ports"
    directory.mkdir(parents=True, exist_ok=True)


    with ExitStack() as reserved:

        with (directory / "cursor").open("a+") as cursor:
            fcntl.flock(cursor, fcntl.LOCK_EX)
            cursor.seek(0)
            start = int(cursor.read() or "0") % size


            for step in range(size):
                offset = (start + step) % size
                if offset + parallel + 1 > size:

                    continue
                try:
                    with ExitStack() as candidate:
                        for port in range(first + offset, first + offset + parallel + 1):
                            lock = candidate.enter_context((directory / f"{port}.lock").open("a+"))
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            with socket.socket() as connection:
                                connection.bind(("127.0.0.1", port))


                        reserved.enter_context(candidate.pop_all())
                    break
                except OSError:

                    continue
            else:
                raise RuntimeError(f"No free model/scaffold port block ({first}–{last})")
            cursor.seek(0)
            cursor.truncate()
            cursor.write(str((offset + parallel + 1) % size))
            cursor.flush()

        yield first + offset


def plan_request(action: str, **values):
    for attempt in range(3):
        try:
            response = requests.post(
                os.environ["WORKER_GOOGLE_SHEETS_API_URL"],
                json={"token": os.environ["WORKER_GOOGLE_SHEETS_TOKEN"], "action": action, **values}, timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("error") == "plan:busy":
                raise requests.exceptions.RetryError("Google Sheets plan busy")
            break
        except requests.RequestException:
            logging.getLogger(__name__).warning("Google Sheets action=%s attempt=%d/3 job_id=%s failed", action, attempt + 1, values.get("job_id", ""), exc_info=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    if not data.get("ok"):
        raise RuntimeError(f"Google Sheets plan: {data.get('error', 'invalid response')}")
    return data["result"]


@contextmanager
def reserve_gpu():
    gpu = os.environ["WORKER_GPU_NUM"]
    if not gpu:
        yield
        return
    directory = ROOT / "_state/runtime/gpus"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{gpu}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"worker already runs for WORKER_GPU_NUM {gpu}")
        yield


def create_worker_logs() -> Path:
    name = os.environ["WORKER_NAME"] or socket.gethostname()[:5]
    device = os.environ["WORKER_GPU_NUM"] or os.environ["WORKER_BACKEND"]
    base = ROOT / "_state/logs" / f"{name}-{device}-{datetime.now(timezone.utc):%y%m%d%H%M}"
    logs = base
    suffix = 2
    while True:
        try:
            logs.mkdir(parents=True)
            return logs
        except FileExistsError:
            logs = base.with_name(f"{base.name}-{suffix}")
            suffix += 1


@contextmanager
def start_backend(logs: Path, port: int):
    os.environ["BACKEND_MODEL_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    with (logs / "backend.log").open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", f"backends.{os.environ['WORKER_BACKEND']}", str(port)],
            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
        )

    try:
        wait_http(process, f"http://127.0.0.1:{port}/v1/models", "local", timeout=float(os.environ["WORKER_BACKEND_START_TIMEOUT"]), model=os.environ["WORKER_MODEL"])
        yield process
    finally:
        stop(process)


def publish_result(results: dict, row: dict, snapshot: dict):
    job_id = snapshot["job"]["job_id"]
    if job_id in results["published"]:
        return
    final = snapshot["job"]["status"] != "running"
    values = row | snapshot_columns(snapshot)
    values.update(worker_id=results["logs"].name)

    values.update(avg_columns(results["avg"], snapshot))
    if values["max_input_tokens"] != "":
        results["max_input_tokens"][job_id] = int(values["max_input_tokens"])
    elapsed = time.time() - results["started_at"]
    values["avg_tps"] = f"{sum(results['max_input_tokens'].values()) / elapsed:.2f}" if elapsed > 0 else ""
    if final:
        results["writer"].writerow(values)
        results["output"].flush()
        os.fsync(results["output"].fileno())
        results["published"].add(job_id)
    results["pending"][job_id] = {"job_id": job_id, "values": {key: values[key] for key in FIELDS[6:]}}


def flush_results(results: dict) -> bool:
    if not results["pending"]:
        return True
    try:
        rejected = plan_request("update_many", updates=list(results["pending"].values())) or []
    except Exception:
        logging.exception("Google Sheets batch update failed; retaining %d updates for retry", len(results["pending"]))
        return False
    for item in rejected:
        logging.warning("Google Sheets job %s rejected: %s; discarding obsolete table update", item["job_id"], item["error"])
    for job_id, update in results["pending"].items():
        if update["values"]["status"] != "running":
            (ROOT / "_state/runtime/jobs" / results["logs"].name / f"{job_id}.json").unlink(missing_ok=True)
    results["pending"].clear()
    return True


def save_job_control(control: dict) -> None:
    path = control["path"]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(control["record"]))
    temporary.replace(path)


def create_job_control(row: dict, worker_id: str) -> dict:
    path = ROOT / "_state/runtime/jobs" / worker_id / f"{row['job_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    control = {"path": path, "stopped": Event(), "record": {
        "row": row, "token": uuid4().hex, "heartbeat_at": time.time(), "snapshot": None, "stop_error": None,
    }}
    save_job_control(control)
    return control


def request_job_stop(control: dict, error: str) -> None:
    if not control["record"]["stop_error"]:
        control["record"]["stop_error"] = error
    control["stopped"].set()


def record_job_progress(control: dict, snapshot: dict) -> None:
    final = snapshot["job"]["status"] != "running"
    if control["stopped"].is_set():
        if not final:
            return
        error = control["record"]["stop_error"]
        snapshot["job"].update(status="interrupted" if error == "job:interrupted" else "error", error=error)
    control["record"].update(heartbeat_at=time.time(), snapshot=snapshot)
    save_job_control(control)


def restore_job_results(worker_id: str) -> None:
    directory = ROOT / "_state/runtime/jobs"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "recovery.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in directory.glob("*/*.json"):
            if path.parent.name == worker_id:
                continue
            owner = ROOT / "_state/runtime/workers" / f"{path.parent.name}.json"
            if owner.exists():
                worker = json.loads(owner.read_text())
                try:
                    if psutil.Process(worker["pid"]).create_time() == worker["created"]:
                        continue
                except psutil.NoSuchProcess:
                    pass
            record = json.loads(path.read_text())
            snapshot = record["snapshot"]
            final = snapshot is not None and snapshot["job"]["status"] != "running"
            if not final and time.time() - record["heartbeat_at"] <= float(os.environ["WORKER_STALE_AFTER"]):
                continue
            stop_token_processes(record["token"])
            error = record.get("stop_error") or "job:heartbeat_timeout"
            status = "interrupted" if error == "job:interrupted" else "error"
            values = snapshot_columns(snapshot) if final else {"status": status, "result": "interrupted" if status == "interrupted" else "error", "error": error}
            try:
                plan_request("update", job_id=record["row"]["job_id"], values=values | {"worker_id": record["row"]["worker_id"]})
            except Exception:
                logging.exception("Could not restore job %s; retaining runtime record", record["row"]["job_id"])
                continue
            path.unlink(missing_ok=True)


def run_jobs(process: subprocess.Popen, model_port: int, results: dict) -> int:
    parallel = int(os.environ["WORKER_PARALLEL"])
    interval = float(os.environ["WORKER_HEARTBEAT_INTERVAL"])
    last_sync, next_claim = time.monotonic(), 0.0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = set()
        stop_error = "worker:failed"
        try:
            while True:
                sync_due = time.monotonic() - last_sync >= interval
                if sync_due:
                    restore_job_results(results["logs"].name)
                if process.poll() is not None:
                    stop_error = "backend:exited"
                    raise RuntimeError(f"model backend exited with code {process.returncode}")
                finished = False
                for future in list(futures):
                    control = future.control
                    if future.done():
                        record_job_progress(control, future.result())
                        publish_result(results, future.row, control["record"]["snapshot"])
                        futures.remove(future)
                        finished = True
                    else:
                        if time.time() - control["record"]["heartbeat_at"] > float(os.environ["WORKER_STALE_AFTER"]):
                            if not control["stopped"].is_set():
                                logging.warning("Job %s heartbeat expired; stopping its processes", future.row["job_id"])
                            request_job_stop(control, "job:heartbeat_timeout")
                            stop_token_processes(control["record"]["token"])
                        if sync_due and control["record"]["snapshot"] is not None:
                            publish_result(results, future.row, control["record"]["snapshot"])
                if sync_due or finished:
                    flush_results(results)
                    last_sync = time.monotonic()
                exhausted = False
                while len(futures) < parallel and time.monotonic() >= next_claim:
                    try:
                        row = plan_request(
                            "claim", model=os.environ["WORKER_MODEL_ALIAS"] or os.environ["WORKER_MODEL"],
                            worker_id=results["logs"].name,
                            exclude_job_ids=list(results["published"] | {future.row["job_id"] for future in futures}),
                        )
                    except Exception:
                        logging.exception("Google Sheets claim failed; retrying after heartbeat interval")
                        next_claim = time.monotonic() + interval
                        break
                    if row is None:
                        next_claim = time.monotonic() + interval
                        exhausted = True
                        break
                    slot = next(slot for slot in range(parallel) if all(future.slot != slot for future in futures))
                    control = create_job_control(row, results["logs"].name)
                    future = pool.submit(
                        run_job, row, results["logs"].name, results["logs"], slot,
                        model_port + 1, interval, partial(record_job_progress, control), control=control,
                    )
                    future.control, future.row, future.slot = control, row, slot
                    futures.add(future)
                if not futures and exhausted and not results["pending"]:
                    break
                if futures:
                    wait(futures, timeout=1)
                else:
                    time.sleep(1)
        except (KeyboardInterrupt, SystemExit) as error:
            if isinstance(error, KeyboardInterrupt) or error.code in (129, 130, 143):
                stop_error = "job:interrupted"
            raise
        finally:
            for future in futures:
                if not future.done():
                    request_job_stop(future.control, stop_error)
                stop_token_processes(future.control["record"]["token"])
            while any(not future.done() for future in futures):
                for future in futures:
                    if not future.done():
                        stop_token_processes(future.control["record"]["token"])
                time.sleep(0.1)
            for future in futures:
                try:
                    snapshot = future.result()
                except Exception:
                    request_job_stop(future.control, stop_error)
                    save_job_control(future.control)
                    logging.exception("Job %s failed during worker shutdown", future.row["job_id"])
                else:
                    record_job_progress(future.control, snapshot)
                    publish_result(results, future.row, snapshot)
    return 0


def execute_plan(logs: Path) -> int:
    with reserve_gpu(), reserve_ports(int(os.environ["WORKER_PARALLEL"])) as model_port:

        restore_job_results(logs.name)

        with (logs / "results.csv").open("a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            output.flush()

            results = {
                "logs": logs, "output": output, "writer": writer,
                "avg": dict.fromkeys(("tools", "tool_time", "jobs"), 0),
                "published": set(), "max_input_tokens": {}, "started_at": psutil.Process().create_time(),
                "pending": {},
            }
            with start_backend(logs, model_port) as process:
                return run_jobs(process, model_port, results)


def stop_processes(record: dict) -> None:
    try:
        owner = psutil.Process(record["pid"])
        if owner.pid != os.getpid() and owner.create_time() == record["created"]:
            owner.terminate()
            try:
                owner.wait(timeout=3)
            except psutil.TimeoutExpired:
                owner.kill()
                try:
                    owner.wait(timeout=3)
                except psutil.TimeoutExpired:

                    if owner.status() != psutil.STATUS_ZOMBIE:
                        raise
    except psutil.NoSuchProcess:
        pass

    stop_token_processes(record["token"], "WORKER_PROCESS_TOKEN")


def archive_logs() -> None:
    logs = ROOT / "_state/logs"
    runtime = ROOT / "_state/runtime/logs"
    runtime.mkdir(parents=True, exist_ok=True)
    try:
        with (runtime / "archive.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            archives = logs / "archives"
            archives.mkdir(parents=True, exist_ok=True)
            active = {path.stem for path in (ROOT / "_state/runtime/workers").glob("*.json")}
            opened = {Path(item.path) for process in psutil.process_iter(["open_files"], ad_value=[])
                      for item in process.info["open_files"]}
            cutoff = datetime.now().date() - timedelta(days=1)
            days = {}
            for path in logs.rglob("*"):
                if not path.is_file() or path.is_symlink() or archives in path.parents or path.relative_to(logs).parts[0] in active or path in opened:
                    continue
                stat = path.stat()
                day = datetime.fromtimestamp(stat.st_mtime).date()
                if day < cutoff:
                    days.setdefault(day, []).append((path, stat))
            for day, files in days.items():
                archive = archives / f"{day}.zip"
                temporary = archive.with_suffix(".zip.tmp")
                if archive.exists():
                    shutil.copyfile(archive, temporary)
                with zipfile.ZipFile(temporary, "a" if archive.exists() else "w", compression=zipfile.ZIP_DEFLATED) as output:
                    existing = set(output.namelist())
                    for path, stat in files:
                        name = path.relative_to(logs).as_posix()
                        if name not in existing:
                            output.write(path, name)
                temporary.replace(archive)
                for path, stat in files:
                    current = path.stat()
                    if (current.st_mtime_ns, current.st_size) == (stat.st_mtime_ns, stat.st_size):
                        path.unlink()
                        parent = path.parent
                        while parent != logs and not any(parent.iterdir()):
                            parent.rmdir()
                            parent = parent.parent
                logging.getLogger(__name__).info("Archived %d log files to %s", len(files), archive)
    except Exception:
        logging.getLogger(__name__).exception("Daily log archiving failed")


def write_worker_heartbeat(path: Path, record: dict) -> None:
    record["heartbeat_at"] = time.time()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record))
    temporary.replace(path)


def update_worker_heartbeat(path: Path, record: dict, stopped: Event) -> None:
    archive_day = datetime.now().date()
    while not stopped.wait(float(os.environ["WORKER_HEARTBEAT_INTERVAL"])):
        with path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                return
            write_worker_heartbeat(path, record)
        if archive_day != datetime.now().date():
            Thread(target=archive_logs, daemon=True).start()
            archive_day = datetime.now().date()


def stop_stale_workers(directory: Path) -> None:
    for path in directory.glob("*.json"):
        with path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if time.time() - record["heartbeat_at"] > float(os.environ["WORKER_STALE_AFTER"]):
                stop_processes(record)
                path.unlink(missing_ok=True)
                path.with_suffix(".tmp").unlink(missing_ok=True)


@contextmanager
def manage_worker_processes():
    directory = ROOT / "_state/runtime/workers"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "registry.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stop_stale_workers(directory)
        logs = create_worker_logs()
        path = directory / f"{logs.name}.json"
        record = {
            "worker_id": logs.name, "pid": os.getpid(),
            "created": psutil.Process().create_time(), "token": uuid4().hex,
        }
        os.environ["WORKER_PROCESS_TOKEN"] = record["token"]
        write_worker_heartbeat(path, record)
    stopped = Event()
    heartbeat = Thread(target=update_worker_heartbeat, args=(path, record, stopped), daemon=True)
    heartbeat.start()
    try:
        yield logs
    finally:
        stopped.set()
        heartbeat.join()
        stop_processes(record)
        with path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path.unlink(missing_ok=True)
            path.with_suffix(".tmp").unlink(missing_ok=True)


def main() -> int:
    for event in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(event, lambda signum, _: sys.exit(128 + signum))
    load_config()
    with manage_worker_processes() as logs:
        logging.basicConfig(filename=logs / "worker.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        archive_logs()
        return execute_plan(logs)


if __name__ == "__main__":
    raise SystemExit(main())
