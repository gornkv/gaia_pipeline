"""Регресс-тесты Google-плана, CSV-журнала, параллельности и порядка фич."""

import csv
import json
import os
from threading import Event
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch, Mock

from benchmarks.gaia.run import prepare_sandbox
from backends import external, llama
from worker.job import (
    run_job,
    feature_environment, job_environment, job_directories, start_scaffold,
)
from worker.worker import (
    FIELDS, plan_request, reserve_ports, reserve_gpu, start_backend, create_worker_logs, execute_plan, avg_columns, snapshot_columns, archive_logs,
)


from dotenv import dotenv_values

DEFAULTS = {
    key: value for path in Path(__file__).resolve().parents[1].rglob(".env.default")
    if "_state" not in path.parts and ".venv" not in path.parts
    for key, value in dotenv_values(path, interpolate=False).items()
}


def job_report(job_id="1", status="done", tool_time=10, model_time=2):
    return {"job": {"job_id": job_id, "worker_id": "test", "status": status, "error": None,
                     "full_time_seconds": 12.5, "time_limit_seconds": 2300},
            "benchmark": {"result": "solved", "error": None},
            "calls": [
                {"call_id": str(i), "name": "echo", "duration_seconds": tool_time / 2,
                 "time_limit_seconds": 120, "status": "success", "error": None} for i in range(2)],
            "scaffold": {"model": {"calls": 1, "max_input_tokens": 100, "time_sum_seconds": model_time,
                                    "output_tokens_sum": 7, "usage_complete": True},
                         "time_limit_seconds": 2300, "avg_token_factor": 1}}


class RunnerTest(unittest.TestCase):
    def test_local_gaia_files_use_job_directory(self):
        from types import SimpleNamespace
        from benchmarks.gaia.run import _gaia
        sample = SimpleNamespace(input="Read /shared_files/test.txt", setup="mkdir -p /shared_files/",
                                 files={"/shared_files/test.txt": "/dataset/test.txt"})
        task = SimpleNamespace(dataset=[sample])
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict("os.environ", {"WORKER_TOOL_SANDBOX": "local", "XDG_RUNTIME_DIR": directory}), \
             patch("benchmarks.gaia.run.upstream_gaia", return_value=task) as upstream, \
             patch("benchmarks.gaia.run.selected_tools", return_value=[]), patch("benchmarks.gaia.run.react"):
            _gaia("2023_level1")
            target = str(Path(directory) / "shared_files/test.txt")
            self.assertEqual(sample.input, f"Read {target}")
            self.assertEqual(sample.files, {target: "/dataset/test.txt"})
            self.assertIsNone(sample.setup)
            self.assertEqual(upstream.call_args.kwargs["sandbox"], "local")

    def test_job_control_socket_fits_system_temp_directory(self):
        import socket
        from inspect_ai._control.discovery import default_socket_path
        with tempfile.TemporaryDirectory() as directory, patch("worker.job.ROOT", Path(directory)):
            with job_directories("12345678", "worker", Path(directory) / "logs") as (logs, runtime):
                row = dict.fromkeys(FIELDS, "")
                env = job_environment(row, 0, 19080, logs, runtime)
                with patch.dict(os.environ, env):
                    path = default_socket_path(os.getpid())
                    self.assertTrue(path.is_relative_to(runtime))
                    with socket.socket(socket.AF_UNIX) as server:
                        server.bind(str(path))

    @patch("worker.worker.requests.post")
    def test_plan_accepts_custom_endpoint(self, post):
        post.return_value.json.return_value = {"ok": True, "result": None}
        with patch.dict("os.environ", {"WORKER_GOOGLE_SHEETS_API_URL": "plan.csv", "WORKER_GOOGLE_SHEETS_TOKEN": "secret"}):
            plan_request("initialize")
            self.assertEqual(post.call_args.args[0], "plan.csv")

    def test_results_columns_match_google_sheets(self):
        import re
        script = (Path(__file__).resolve().parents[1] / "worker/google_sheets/Code.js").read_text()
        header = re.search(r"const FIELDS = \[(.*?)\];", script, re.S).group(1)
        self.assertEqual(FIELDS, re.findall(r"'([^']+)'", header))

    def test_model_alias_is_not_sent_to_backend_api(self):
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="Qwen3.5-0.8B")
        with patch.dict("os.environ", DEFAULTS | {"WORKER_MODEL": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M", "BACKEND_MODEL_BASE_URL": "http://model/v1"}):
            env = job_environment(row, 0, 19080, Path("/tmp/job"), Path("/tmp/runtime"))
        self.assertEqual(env["WORKER_MODEL"], "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M")
        self.assertEqual(env["BACKEND_MODEL_BASE_URL"], "http://model/v1")
        self.assertEqual(env["BENCHMARK_MODEL_BASE_URL"], "http://127.0.0.1:19080/v1")
        self.assertEqual(row["model"], "Qwen3.5-0.8B")
        self.assertEqual(env["INSPECT_LOG_DIR"], "/tmp/job/inspect-logs")
        self.assertTrue(env["PATH"].startswith(str(Path(__file__).resolve().parents[1] / ".venv/bin") + ":"))
        self.assertEqual(env["INSPECT_TRACE_FILE"], "/tmp/job/inspect-trace.log")

    def setUp(self):
        environment = patch.dict("os.environ", DEFAULTS)
        environment.start()
        self.addCleanup(environment.stop)

    def test_existing_llama_server_skips_installation(self):
        with patch.dict("os.environ", DEFAULTS | {}, clear=True), \
             patch.object(llama.shutil, "which", return_value="/custom/llama-server") as which, \
             patch.object(llama, "urlopen") as download, patch.object(llama.subprocess, "run") as run:
            self.assertIsNone(llama.install())
            which.assert_called_once_with("llama-server")
            download.assert_not_called()
            run.assert_not_called()

    def test_conda_installer_runs_once_and_reuses_environment(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(llama, "ROOT", Path(directory)), \
             patch.dict("os.environ", DEFAULTS | {"BACKEND_LLAMA_ACCELERATOR": "cuda"}, clear=True), patch.object(llama.platform, "system", return_value="Linux"), \
             patch.object(llama.platform, "machine", return_value="x86_64"), \
             patch.object(llama.shutil, "which", return_value=None):
            root = Path(directory) / "_state/cache/llama-conda"
            root.mkdir(parents=True)
            (root / "micromamba-linux-64").touch()
            binary = root / f"{llama.LLAMA_VERSION}-linux-64-cuda/bin/llama-server"
            def run(cmd, **kwargs):
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.touch()
            with patch.object(llama.subprocess, "run", side_effect=run) as commands:
                self.assertIsNone(llama.install())
                self.assertIsNone(llama.install())
                self.assertEqual(commands.call_count, 1)
                self.assertIn(f"llama.cpp={llama.LLAMA_VERSION}=cuda129_*", commands.call_args.args[0])
                self.assertNotIn("sudo", commands.call_args.args[0])
                marker = Path(directory) / f"_state/runtime/llama/{binary.parents[1].name}.ready"
                marker.unlink()
                commands.side_effect = RuntimeError("installation interrupted")
                with self.assertRaisesRegex(RuntimeError, "installation interrupted"):
                    llama.install()
                self.assertFalse(marker.exists())
                commands.side_effect = run
                self.assertIsNone(llama.install())
                self.assertTrue(marker.exists())

    def test_ready_marker_skips_installation_without_probing_server(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(llama, "ROOT", Path(directory)), \
             patch.dict("os.environ", DEFAULTS | {"BACKEND_LLAMA_ACCELERATOR": "metal"}), \
             patch.object(llama.platform, "system", return_value="Darwin"), \
             patch.object(llama.platform, "machine", return_value="arm64"), \
             patch.object(llama.shutil, "which", return_value=None), \
             patch.object(llama, "urlopen") as download, \
             patch.object(llama.subprocess, "run") as run:
            binary = Path(directory) / f"_state/cache/llama-conda/{llama.LLAMA_VERSION}-osx-arm64-metal/bin/llama-server"
            marker = Path(directory) / f"_state/runtime/llama/{llama.LLAMA_VERSION}-osx-arm64-metal.ready"
            marker.parent.mkdir(parents=True)
            marker.touch()
            self.assertIsNone(llama.install())
            self.assertEqual(os.get_exec_path()[0], str(binary.parent))
            run.assert_not_called()
            download.assert_not_called()

    def test_micromamba_latest_is_extracted_and_used_for_pinned_llama(self):
        from io import BytesIO
        import tarfile
        archive = BytesIO()
        with tarfile.open(fileobj=archive, mode="w:bz2") as package:
            member = tarfile.TarInfo("bin/micromamba")
            member.size = len(b"test executable")
            package.addfile(member, BytesIO(b"test executable"))
        archive.seek(0)
        with tempfile.TemporaryDirectory() as directory, patch.object(llama, "ROOT", Path(directory)), \
             patch.dict("os.environ", DEFAULTS | {"BACKEND_LLAMA_ACCELERATOR": "metal"}, clear=True), patch.object(llama.platform, "system", return_value="Darwin"), \
             patch.object(llama.platform, "machine", return_value="arm64"), \
             patch.object(llama.shutil, "which", return_value=None), \
             patch.object(llama, "urlopen", return_value=archive) as download, \
             patch.object(llama.subprocess, "run") as run:
            root = Path(directory) / "_state/cache/llama-conda"
            binary = root / f"{llama.LLAMA_VERSION}-osx-arm64-metal/bin/llama-server"
            def installed(*args, **kwargs):
                binary.parent.mkdir(parents=True)
                binary.touch()
            run.side_effect = installed
            self.assertIsNone(llama.install())
            download.assert_called_once_with("https://micro.mamba.pm/api/micromamba/osx-arm64/latest", timeout=120)
            manager = root / "micromamba-osx-arm64"
            self.assertEqual(manager.read_bytes(), b"test executable")
            self.assertTrue(manager.stat().st_mode & 0o111)
            self.assertEqual(run.call_args.args[0][0], str(manager))
            self.assertIn(f"llama.cpp={llama.LLAMA_VERSION}=cpu_accelerate_*", run.call_args.args[0])

    def test_native_llama_run_uses_explicit_hardware_profile(self):
        with patch.dict("os.environ", DEFAULTS | {
            "WORKER_MODEL_CONTEXT_LIMIT": "262144",
            "WORKER_GPU_NUM": "2", "BACKEND_LLAMA_KV_POOL_SIZE": "786432", "BACKEND_LLAMA_BATCH_SIZE": "1024",
            "BACKEND_LLAMA_ACCELERATOR": "cuda",
        }, clear=True), patch("backends.llama.install", return_value=None), patch("backends.llama.os.execvp") as execute:
            with patch.dict("os.environ", {"WORKER_MODEL": "model", "WORKER_PARALLEL": "6"}):
                llama.run(18023)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
            cmd = execute.call_args.args[1]
            self.assertEqual(cmd[0], "llama-server")
            for flag, value in [("--ctx-size", "786432"), ("--kv-unified-per-slot", "262144"),
                                ("--parallel", "6"), ("--port", "18023"),
                                ("--batch-size", "1024"), ("--fit", "off")]:
                self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_native_llama_m2_profile(self):
        with patch.dict("os.environ", DEFAULTS | {
            "WORKER_MODEL_CONTEXT_LIMIT": "16384", "WORKER_GPU_NUM": "",
            "BACKEND_LLAMA_KV_POOL_SIZE": "65536",
            "BACKEND_LLAMA_ACCELERATOR": "metal",
        }, clear=True), patch("backends.llama.install", return_value=None), patch("backends.llama.os.execvp") as execute:
            with patch.dict("os.environ", {"WORKER_MODEL": "model", "WORKER_PARALLEL": "6"}):
                llama.run(18023)
            cmd = execute.call_args.args[1]
            self.assertEqual(cmd[0], "llama-server")
            for flag, value in [("--ctx-size", "65536"), ("--batch-size", "256"), ("--ubatch-size", "64")]:
                self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_native_llama_defaults_use_small_pool(self):
        with patch.dict("os.environ", DEFAULTS | {
            "WORKER_MODEL_CONTEXT_LIMIT": "24000",
        }, clear=True), patch("backends.llama.install", return_value=None), patch("backends.llama.os.execvp") as execute:
            llama.run(18023)
            cmd = execute.call_args.args[1]
            self.assertEqual(cmd[cmd.index("--ctx-size") + 1], "32768")
            self.assertEqual(cmd[cmd.index("--batch-size") + 1], "256")
            self.assertEqual(cmd[cmd.index("--ubatch-size") + 1], "64")

    def test_job_progress_and_final_share_format_and_use_only_callback(self):
        with tempfile.TemporaryDirectory() as directory, patch("worker.job.ROOT", Path(directory)), patch("worker.job.wait_http"), patch("worker.job.stop"):
            root = Path(directory)
            progress = []
            observed = Event()
            row = dict.fromkeys(FIELDS, "")
            row.update(job_id="1", benchmark="gaia", model="m", index="1")

            def popen(command, **kwargs):
                env = kwargs["env"]
                if command[0].endswith("inspect-tool-support"):
                    (Path(env["XDG_RUNTIME_DIR"]) / "inspect-tool-support.sock").touch()
                else:
                    logs = Path(env["INSPECT_LOG_DIR"]).parent
                    (logs / "scaffold.json").write_text(json.dumps(job_report()["scaffold"]))
                    (logs / "benchmark.json").write_text(json.dumps(job_report()["benchmark"]))
                    (logs / "calls.json").write_text(json.dumps(job_report()["calls"]))
                return Mock(poll=lambda: None)

            def execute(command, **kwargs):
                self.assertEqual(command[1:], ["-m", "benchmarks.gaia"])
                self.assertEqual(kwargs["env"]["BENCHMARK_INDEX"], "1")
                self.assertEqual(kwargs["env"]["WORKER_MODEL"], os.environ["WORKER_MODEL"])
                self.assertEqual(int(kwargs["env"]["BENCHMARK_TIME_LIMIT"]), 2300)
                self.assertTrue(observed.wait(2))
                return Mock(returncode=0)

            def receive(snapshot):
                progress.append(snapshot)
                observed.set()

            with patch("worker.job.subprocess.Popen", side_effect=popen), patch("worker.job.subprocess.run", side_effect=execute):
                final = run_job(row, "test", root / "logs", 0, 19000, 0.01, receive)
            logs = next((root / "logs/jobs").iterdir())
            self.assertTrue(progress)
            self.assertEqual(progress[-1]["job"]["status"], "running")
            self.assertEqual(final["job"]["status"], "done")
            self.assertEqual(final.keys(), progress[-1].keys())
            self.assertEqual(final["calls"], progress[-1]["calls"])
            self.assertTrue((logs / "scaffold.json").is_file())
            self.assertFalse(list(root.glob("_state/runtime/jobs/test/*")))

    def test_job_failures_preserve_error_phase_and_clean_resources(self):
        import subprocess
        for phase, expected_status, expected_error in [
            ("tools", "error", "tools:start"), ("scaffold", "error", "scaffold:start"),
            ("benchmark", "error", "benchmark:exit_1"), ("timeout", "done", "job:timeout"),
            ("sigterm", "error", "benchmark:sigterm"), ("sigkill", "error", "benchmark:sigkill"),
        ]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory, \
                 patch("worker.job.ROOT", Path(directory)), patch("worker.job.stop") as stop_process:
                root = Path(directory)
                processes, socket_paths = [], []
                row = dict.fromkeys(FIELDS, "")
                row.update(job_id="failure", benchmark="gaia", index="1")

                def spawn(command, **kwargs):
                    env = kwargs["env"]
                    runtime = Path(env["XDG_RUNTIME_DIR"])
                    socket_paths.append(runtime)
                    process = Mock(poll=lambda: 1 if phase == "tools" else None)
                    processes.append(process)
                    if command[0].endswith("inspect-tool-support"):
                        if phase != "tools":
                            (runtime / "inspect-tool-support.sock").touch()
                    else:
                        logs = Path(env["INSPECT_LOG_DIR"]).parent
                        (logs / "scaffold.json").write_text(json.dumps(job_report()["scaffold"]))
                    return process

                failure = subprocess.TimeoutExpired("benchmark", 1) if phase == "timeout" else subprocess.CalledProcessError({"sigterm": -15, "sigkill": -9}.get(phase, 1), "benchmark")
                with patch("worker.job.subprocess.Popen", side_effect=spawn), \
                     patch("worker.job.wait_http", side_effect=RuntimeError("not ready") if phase == "scaffold" else None), \
                     patch("worker.job.subprocess.run", side_effect=failure):
                    result = run_job(row, "test", root / "logs", 0, 19000, 0.01, Mock())
                self.assertEqual(result["job"]["status"], expected_status)
                self.assertEqual(result["job"]["error"], expected_error)
                self.assertEqual(stop_process.call_args_list, [unittest.mock.call(process) for process in reversed(processes)])
                self.assertTrue(next((root / "logs/jobs").glob("*/error.log")).is_file())
                self.assertTrue(all(not path.exists() for path in socket_paths))
                self.assertFalse(list(root.glob("_state/runtime/jobs/test/*")))

    def test_google_sync_failure_keeps_all_active_job_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("benchmarks/gaia/run.py", ".venv/bin/inspect", ".venv/bin/inspect-tool-support"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            queue = []
            saved = set()
            for i in range(1, 4):
                row = dict.fromkeys(FIELDS, "")
                row.update(job_id=str(i), model="m", status="running", worker_id="test")
                queue.append(row)

            def call(action, **values):
                if action == "mark_stale_jobs_interrupted":
                    return 0
                if action == "claim":
                    self.assertEqual(values["model"], "m-alias")
                    return queue.pop(0) if queue else None
                self.assertEqual(action, "update")
                if values["values"]["status"] == "running":
                    self.assertEqual(values["values"]["tool_calls"], "2")
                    self.assertFalse(list(root.glob("_state/logs/*/jobs/*/job.json")))
                    return None
                # Результат обязан быть записан до неудачного обращения в Google.
                results = next(root.glob("_state/logs/*/results.csv"))
                with results.open() as source:
                    reader = csv.DictReader(source)
                    self.assertEqual(reader.fieldnames, FIELDS)
                    ids = [row["job_id"] for row in reader]
                    self.assertIn(values["job_id"], ids)
                    self.assertTrue(saved.issubset(ids))
                    self.assertEqual(len(ids), len(set(ids)))
                    saved.update(ids)
                raise ConnectionError("offline")

            def execute_job(row, worker_id, logs, *args):
                logs = logs / "jobs" / row["job_id"]
                logs.mkdir(parents=True)
                snapshot = job_report(row["job_id"], "running")
                args[-1](snapshot)
                args[-1](snapshot)
                return job_report(row["job_id"])

            environment = {
                "WORKER_GOOGLE_SHEETS_API_URL": "https://script.google.com/macros/s/test/exec",
                "WORKER_GOOGLE_SHEETS_TOKEN": "secret", "WORKER_MODEL": "m", "WORKER_MODEL_ALIAS": "m-alias",
                "WORKER_BACKEND": "external", "WORKER_MODEL_CONTEXT_LIMIT": "24000",
                "BENCHMARK": "gaia", "WORKER_PARALLEL": "2", "WORKER_NAME": "test", "WORKER_GPU_NUM": "",
            }
            with patch.dict("os.environ", DEFAULTS | environment, clear=True), patch("worker.worker.ROOT", root), \
                 patch("worker.worker.load_config"), patch("worker.worker.reserve_ports", return_value=nullcontext(18000)), \
                 patch("worker.worker.subprocess.Popen", return_value=Mock(poll=lambda: None)) as launch, \
                 patch("worker.worker.wait_http"), patch("worker.worker.stop"), patch("worker.worker.plan_request", side_effect=call), \
                 patch("worker.worker.run_job", side_effect=execute_job):
                self.assertEqual(execute_plan(create_worker_logs()), 1)
                self.assertEqual(launch.call_args.args[0][1:], ["-m", "backends.external", "18000"])
            with next(root.glob("_state/logs/*/results.csv")).open() as source:
                results = list(csv.DictReader(source))
            self.assertEqual({row["job_id"] for row in results}, {"1", "2"})
            self.assertTrue(all(row["status"] == "done" and row["error"] == "" for row in results))
            self.assertEqual(sorted(row["jobs_count"] for row in results), ["1", "2"])
            self.assertTrue(all(row["tool_calls"] == "2" for row in results))
            self.assertEqual([row["job_id"] for row in queue], ["3"])

    @patch("worker.worker.requests.post")
    def test_google_plan_delegates_atomic_operations(self, post):
        post.return_value.json.return_value = {"ok": True, "result": {"job_id": "1"}}
        with patch.dict("os.environ", DEFAULTS | {
            "WORKER_GOOGLE_SHEETS_API_URL": "https://script.google.com/macros/s/test/exec",
            "WORKER_GOOGLE_SHEETS_TOKEN": "secret",
        }):
            self.assertEqual(plan_request("claim", model="m", worker_id="worker", exclude_job_ids=["already-tried"]), {"job_id": "1"})
            self.assertEqual(post.call_args.kwargs["json"]["exclude_job_ids"], ["already-tried"])
            plan_request("update", job_id="1", values={"worker_id": "worker", "status": "running"})
            plan_request("mark_stale_jobs_interrupted", stale_after=120)
            plan_request("update", job_id="1", values={"worker_id": "worker", "status": "done"})
            post.return_value.json.return_value = {"ok": False, "error": "unauthorized"}
            with self.assertRaisesRegex(RuntimeError, "unauthorized"):
                plan_request("claim", model="m", worker_id="worker", exclude_job_ids=[])
        self.assertEqual([call.kwargs["json"]["action"] for call in post.call_args_list],
                         ["claim", "update", "mark_stale_jobs_interrupted", "update", "claim"])

    @patch("worker.worker.time.sleep")
    @patch("worker.worker.requests.post")
    def test_google_plan_retries_lost_and_empty_responses(self, post, sleep):
        from requests import ConnectionError
        from requests.exceptions import JSONDecodeError
        response = Mock()
        response.json.return_value = {"ok": True, "result": {"job_id": "1"}}
        post.side_effect = [ConnectionError("response lost"), JSONDecodeError("empty", "", 0), response]
        with patch.dict(os.environ, {"WORKER_GOOGLE_SHEETS_API_URL": "https://test/exec", "WORKER_GOOGLE_SHEETS_TOKEN": "secret"}), \
             self.assertLogs("worker.worker", level="WARNING") as logged:
            self.assertEqual(plan_request("claim", model="m", worker_id="w", exclude_job_ids=[]), {"job_id": "1"})
        self.assertEqual(len(logged.records), 2)
        self.assertIn("action=claim attempt=1/3", logged.output[0])
        self.assertIn("ConnectionError: response lost", logged.output[0])
        self.assertIn("attempt=2/3", logged.output[1])
        self.assertNotIn("secret", "\n".join(logged.output))
        self.assertEqual(post.call_count, 3)
        self.assertTrue(all(call == post.call_args_list[0] for call in post.call_args_list))
        self.assertEqual(sleep.call_args_list, [unittest.mock.call(1), unittest.mock.call(2)])

    def test_daily_log_archives_preserve_recent_active_and_open_files(self):
        import zipfile
        from datetime import datetime, timedelta
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory, patch("worker.worker.ROOT", Path(directory)):
            root = Path(directory)
            logs = root / "_state/logs"
            def write(name, age):
                path = logs / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
                stamp = (datetime.now() - timedelta(days=age)).timestamp()
                os.utime(path, (stamp, stamp))
                return path
            today, yesterday = write("today/log", 0), write("yesterday/log", 1)
            old, older = write("finished/log", 2), write("older/log", 3)
            active, opened = write("active/log", 5), write("service/open.log", 5)
            registry = root / "_state/runtime/workers"
            registry.mkdir(parents=True)
            (registry / "active.json").write_text("{}")
            process = SimpleNamespace(info={"open_files": [SimpleNamespace(path=str(opened))]})
            with patch("worker.worker.psutil.process_iter", return_value=[process]):
                archive_logs()
                self.assertTrue(all(path.exists() for path in [today, yesterday, active, opened]))
                self.assertFalse(old.exists())
                self.assertFalse(older.exists())
                day = (datetime.now() - timedelta(days=2)).date()
                archive = logs / "archives" / f"{day}.zip"
                late = write("late/log", 2)
                archive_logs()
                self.assertFalse(late.exists())
                with zipfile.ZipFile(archive) as saved:
                    self.assertEqual(set(saved.namelist()), {"finished/log", "late/log"})
                    self.assertEqual(saved.read("finished/log"), b"finished/log")
                    self.assertIsNone(saved.testzip())
                self.assertEqual(len(list((logs / "archives").glob("*.zip"))), 2)

    @patch("worker.worker.time.sleep")
    @patch("worker.worker.requests.post")
    def test_google_plan_retries_busy_lock(self, post, sleep):
        post.return_value.json.side_effect = [{"ok": False, "error": "plan:busy"}, {"ok": True, "result": None}]
        with patch.dict(os.environ, {"WORKER_GOOGLE_SHEETS_API_URL": "https://test/exec", "WORKER_GOOGLE_SHEETS_TOKEN": "secret"}), \
             self.assertLogs("worker.worker", level="WARNING") as logged:
            self.assertIsNone(plan_request("update", job_id="job1", values={"status": "done"}))
        self.assertEqual(post.call_count, 2)
        self.assertIn("job_id=job1", logged.output[0])
        self.assertIn("plan busy", logged.output[0])

    def test_failed_log_archive_keeps_original_files(self):
        from datetime import datetime, timedelta
        with tempfile.TemporaryDirectory() as directory, patch("worker.worker.ROOT", Path(directory)), \
             patch("worker.worker.psutil.process_iter", return_value=[]):
            path = Path(directory) / "_state/logs/old.log"
            path.parent.mkdir(parents=True)
            path.write_text("important")
            stamp = (datetime.now() - timedelta(days=3)).timestamp()
            os.utime(path, (stamp, stamp))
            with patch("worker.worker.zipfile.ZipFile.write", side_effect=OSError("disk full")), \
                 self.assertLogs("worker.worker", level="ERROR") as logged:
                archive_logs()
            self.assertEqual(path.read_text(), "important")
            self.assertFalse(list((path.parent / "archives").glob("*.zip")))
            self.assertIn("disk full", logged.output[0])

    def test_gpu_lock_is_optional_and_released_after_error(self):
        with tempfile.TemporaryDirectory() as directory, patch("worker.worker.ROOT", Path(directory)):
            with patch.dict(os.environ, {"WORKER_GPU_NUM": ""}), reserve_gpu():
                self.assertFalse((Path(directory) / "_state/runtime/gpus").exists())
            with patch.dict(os.environ, {"WORKER_GPU_NUM": "0"}):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    with reserve_gpu():
                        with self.assertRaises(SystemExit):
                            with reserve_gpu():
                                self.fail("GPU must remain locked")
                        raise RuntimeError("failed")
                with reserve_gpu():
                    pass

    def test_backend_is_stopped_when_readiness_fails(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("worker.worker.subprocess.Popen") as launch, \
             patch("worker.worker.wait_http", side_effect=RuntimeError("not ready")), \
             patch("worker.worker.stop") as stop_backend:
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                with start_backend(Path(directory), 18000):
                    self.fail("backend must not be yielded before readiness")
            stop_backend.assert_called_once_with(launch.return_value)
            self.assertTrue(launch.call_args.kwargs["stdout"].closed)

    def test_job_launches_selected_scaffold_with_explicit_log_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch("worker.job.subprocess.Popen") as launch, \
             patch("worker.job.wait_http") as ready:
            processes = []
            start_scaffold({"WORKER_SCAFFOLD": "custom"}, Path(directory), 18023, processes)
            self.assertEqual(launch.call_args.args[0][1:], ["-m", "scaffolds.custom", "18023", directory])
            self.assertEqual(processes, [launch.return_value])
            ready.assert_called_once_with(launch.return_value, "http://127.0.0.1:18023/health", timeout=60)

    def test_stop_waits_for_killed_process(self):
        import subprocess
        from worker.job import stop
        process = Mock(poll=lambda: None)
        process.wait.side_effect = [subprocess.TimeoutExpired("test", 10), 0]
        stop(process)
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list, [unittest.mock.call(10), unittest.mock.call()])

    def test_port_blocks_remain_reserved_and_release(self):
        with tempfile.TemporaryDirectory() as directory, patch("worker.worker.ROOT", Path(directory)), patch("worker.worker.socket.socket"):
            with reserve_ports(100) as ports:
                self.assertEqual(ports, 18000)
                with reserve_ports(98) as other:
                    self.assertEqual(other, 18101)
                    (Path(directory) / "_state/runtime/workers/ports/cursor").write_text("0")
                    with reserve_ports(1) as following:
                        self.assertEqual(following, 18200)
            (Path(directory) / "_state/runtime/workers/ports/cursor").write_text("0")
            with reserve_ports(100) as ports:
                self.assertEqual(ports, 18000)

    def test_port_cursor_wraps_and_skips_locked_blocks(self):
        with tempfile.TemporaryDirectory() as directory, patch("worker.worker.ROOT", Path(directory)), patch("worker.worker.socket.socket"):
            cursor = Path(directory) / "_state/runtime/workers/ports/cursor"
            cursor.parent.mkdir(parents=True)
            cursor.write_text("998")
            with reserve_ports(1) as last:
                self.assertEqual(last, 18998)
                with reserve_ports(1) as first:
                    self.assertEqual(first, 18000)
                    cursor.write_text("999")
                    with reserve_ports(1) as following:
                        self.assertEqual(following, 18002)

    def test_port_pool_exhaustion_returns_error_after_full_scan(self):
        with tempfile.TemporaryDirectory() as directory, patch("worker.worker.ROOT", Path(directory)), patch("worker.worker.socket.socket") as socket:
            socket.return_value.__enter__.return_value.bind.side_effect = OSError("occupied")
            with self.assertRaisesRegex(RuntimeError, "No free"):
                with reserve_ports(1):
                    self.fail("occupied ports were allocated")

    def test_port_probe_failure_releases_partial_block(self):
        def bind(address):
            if address[1] == 18001:
                raise OSError("port occupied")
        with tempfile.TemporaryDirectory() as directory, patch("worker.worker.ROOT", Path(directory)), patch("worker.worker.socket.socket") as socket:
            socket.return_value.__enter__.return_value.bind.side_effect = bind
            with reserve_ports(2) as ports:
                self.assertEqual(ports, 18002)
                with reserve_ports(1) as following:
                    self.assertEqual(following, 18005)
            # Первая попытка захватила пару 0, но обязана была её освободить после отказа пары 1.
            (Path(directory) / "_state/runtime/workers/ports/cursor").write_text("0")
            socket.return_value.__enter__.return_value.bind.side_effect = None
            with reserve_ports(1) as ports:
                self.assertEqual(ports, 18000)

    def test_tool_timeouts_use_duration_once_and_leave_running_calls_unknown(self):
        snapshot = job_report()
        calls = snapshot["calls"]
        calls[0].update(duration_seconds=121, error={"type": "timeout"}, status="timeout")
        calls[1].update(duration_seconds=None, status="running")
        values = snapshot_columns(snapshot)
        self.assertEqual(values["tool_calls"], "2")
        self.assertEqual(values["tool_timeouts"], "1")
        self.assertEqual(values["tool_time_sum"], "121.00")

    def test_avg_statistics_are_weighted_and_errors_are_excluded(self):
        avg = dict.fromkeys(("tokens", "time", "tools", "tool_time", "jobs"), 0)
        first = avg_columns(avg, job_report())
        self.assertEqual(first["avg_tps"], "50.00")
        self.assertEqual(first["avg_tool_time"], "5.00")
        second = avg_columns(avg, job_report(tool_time=30, model_time=8))
        self.assertEqual(second, {"avg_tps": "20.00", "avg_tool_count": "2.00", "avg_tool_time": "10.00", "jobs_count": "2"})
        self.assertEqual(avg_columns(avg, job_report(status="running")), second)
        snapshot = job_report()
        snapshot["job"]["error"] = "job:timeout"
        self.assertEqual(avg_columns(avg, snapshot), second)

    def test_model_backends_do_not_choose_tool_sandbox(self):
        for backend in [external, llama]:
            self.assertFalse(hasattr(backend, "sandbox"))

    def test_tool_sandbox_is_independent_of_model_backend(self):
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="m")
        with patch.dict("os.environ", DEFAULTS | {}, clear=True):
            for backend in ["external", "llama"]:
                row["backend"] = backend
                env = job_environment(row, 0, 19080, Path("/tmp/job"), Path("/tmp/runtime"))
                self.assertEqual(env["WORKER_TOOL_SANDBOX"], "local")
            row["feature_envs"] = "WORKER_TOOL_SANDBOX=docker"
            self.assertEqual(job_environment(row, 0, 19080, Path("/tmp/job"), Path("/tmp/runtime"))["WORKER_TOOL_SANDBOX"], "docker")
            row["feature_envs"] = "WORKER_TOOL_SANDBOX=invalid"
            with self.assertRaisesRegex(ValueError, "WORKER_TOOL_SANDBOX"):
                job_environment(row, 0, 19080, Path("/tmp/job"), Path("/tmp/runtime"))

    def test_external_backend_starts_local_proxy(self):
        with patch.dict("os.environ", DEFAULTS | {
            "BACKEND_EXTERNAL_API_URL": "https://example.test/v1/",
            "BACKEND_EXTERNAL_API_KEY": "secret",
        }, clear=True), patch("uvicorn.run") as serve:
            external.run(18023)
            self.assertEqual(serve.call_args.kwargs["port"], 18023)

    def test_gaia_cli_prepares_selected_sandbox(self):
        import sys
        from benchmarks.gaia.run import main
        for sandbox in ("local", "docker"):
            with self.subTest(sandbox=sandbox), \
                 patch.dict("os.environ", {"BENCHMARK_MODEL_BASE_URL": "http://model/v1", "WORKER_MODEL": "m",
                                            "WORKER_TOOL_SANDBOX": sandbox, "BENCHMARK_INDEX": "7", "BENCHMARK_TIME_LIMIT": "60"}), \
                 patch("benchmarks.gaia.run.prepare_sandbox") as docker, \
                 patch("benchmarks.gaia.run.subprocess.run") as install, patch("benchmarks.gaia.run.os.execvpe") as execute:
                main()
                command = execute.call_args.args[1]
                self.assertTrue(command[2].endswith("@gaia"))
                self.assertEqual(command[command.index("--limit") + 1], "7-7")
                self.assertNotIn("split=validation", command)
                self.assertEqual(command[command.index("-M") + 1], "client_timeout=60")
                if sandbox == "local":
                    install.assert_called_once_with([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
                    docker.assert_not_called()
                else:
                    docker.assert_called_once_with()
                    install.assert_not_called()

    def test_inspect_client_timeout_reaches_http_transport(self):
        import asyncio
        from inspect_ai.model import get_model
        async def check():
            model = get_model("openai-api/gaia_model/test-timeout", base_url="http://model/v1", api_key="local", client_timeout=2047)
            try:
                self.assertEqual(model.api.client_timeout, 2047)
                self.assertEqual(model.api.client.timeout, 2047)
                self.assertEqual(model.api.http_client.timeout.read, 2047)
            finally:
                await model.api.client.close()
        asyncio.run(check())

    def test_backends_translate_thinking_budget(self):
        with patch.dict("os.environ", DEFAULTS | {"BACKEND_EXTERNAL_THINKING_BUDGET_LIMIT": "2048", "BACKEND_LLAMA_THINKING_BUDGET_LIMIT": "2048"}, clear=True):
            self.assertEqual(
                external.enrich_payload({}), {"reasoning": {"max_tokens": 2048}}
            )
            self.assertEqual(llama.enrich_payload({}), {"reasoning_budget_tokens": 2048})
        for value in ("", "0", "-1"):
            with self.subTest(value=value), patch.dict(
                "os.environ", DEFAULTS | {"BACKEND_EXTERNAL_THINKING_BUDGET_LIMIT": value, "BACKEND_LLAMA_THINKING_BUDGET_LIMIT": value}, clear=True
            ):
                self.assertEqual(external.enrich_payload({}), {})
                self.assertEqual(llama.enrich_payload({}), {})

    @patch("benchmarks.gaia.run.subprocess.run")
    @patch("benchmarks.gaia.run.importlib.util.find_spec")
    def test_existing_gaia_sandbox_image_is_not_pulled(self, find_spec, run):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "gaia"
            package.mkdir()
            (package / "compose.yaml").write_text("services: {}\n")
            find_spec.return_value.submodule_search_locations = [str(package)]
            run.side_effect = [
                unittest.mock.Mock(stdout="image@sha256:digest\n"),
                unittest.mock.Mock(returncode=0),
                unittest.mock.Mock(returncode=0),
            ]
            with patch("benchmarks.gaia.run.ROOT", Path(directory)):
                prepare_sandbox()
            self.assertFalse(any(call.args[0][:2] == ["docker", "pull"] for call in run.call_args_list))
            marker = Path(directory) / "_state/runtime/inspect_ai/docker/inspect-gaia.ready"
            self.assertEqual(marker.read_text(), "image@sha256:digest\n")

    def test_csv_feature_order_is_preserved_in_one_variable(self):
        # Scaffold выполняет middleware последовательно, поэтому порядок через запятую значим.
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="m", feature="new, second")
        env = job_environment(row, 0, 19080, Path("/tmp/job"), Path("/tmp/runtime"))
        self.assertEqual(env["SCAFFOLD_FEATURES"], "new,second")
        self.assertEqual(env["XDG_RUNTIME_DIR"], "/tmp/runtime")

    def test_feature_environments_are_parsed_per_job(self):
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="m", feature="voi", feature_envs="FEATURE_VOI_TOOL_BUDGET=7,FLAG=yes")
        env = job_environment(row, 0, 19080, Path("/tmp/job"), Path("/tmp/runtime"))
        self.assertEqual(env["FEATURE_VOI_TOOL_BUDGET"], "7")
        self.assertEqual(env["FLAG"], "yes")
        self.assertEqual(feature_environment("EMPTY="), {"EMPTY": ""})
        with self.assertRaisesRegex(ValueError, "cannot override SCAFFOLD_FEATURES"):
            feature_environment("SCAFFOLD_FEATURES=voi")

    def test_model_context_limit_is_passed_to_features(self):
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="m", backend="llama")
        with patch.dict("os.environ", DEFAULTS | {"WORKER_MODEL_CONTEXT_LIMIT": "131072"}, clear=True):
            env = job_environment(row, 0, 19080, Path("/tmp/job"), Path("/tmp/runtime"))
        self.assertEqual(env["WORKER_MODEL_CONTEXT_LIMIT"], "131072")


if __name__ == "__main__":
    unittest.main()
