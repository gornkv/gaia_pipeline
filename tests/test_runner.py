"""Регресс-тесты CSV, параллельности, выбора sandbox и порядка фич."""

import csv
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch, Mock

from benchmark.gaia.run import prepare_sandbox
from backends import external, external_local, llama_docker, vllm
from runner.job import (
    _is_context_overflow, _max_context_tokens, _tool_stats, _heartbeat_loop,
    feature_environment, job_environment, sandbox_for_backend,
)
from runner.worker import (
    FIELDS, Plan, reserve_ports, run_worker, job_id_for, avg_columns,
)


class RunnerTest(unittest.TestCase):
    def test_heartbeat_retries_after_network_error(self):
        plan = Mock()
        plan.heartbeat.side_effect = [RuntimeError("network unavailable"), None]
        stopped = Mock()
        stopped.wait.side_effect = [False, False, True]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heartbeat-errors.log"
            (path.parent / "avg.json").write_text('{"max_input_tokens": 1234}')
            with patch("runner.job.time.monotonic", return_value=12):
                _heartbeat_loop(plan, "job", "worker", 1, stopped, path, 2)
            self.assertEqual(plan.heartbeat.call_count, 2)
            plan.heartbeat.assert_called_with("job", "worker", full_time_sum="10", max_input_tokens="1234")
            self.assertIn("network unavailable", path.read_text())

    def test_google_sync_failure_keeps_all_active_job_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("benchmark/gaia/run.py", ".venv/bin/inspect", ".venv/bin/inspect-tool-support"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            queue = []
            for i in range(1, 4):
                row = dict.fromkeys(FIELDS, "")
                row.update(job_id=str(i), model="m", backend="external", status="running", worker_id="test")
                queue.append(row)

            def call(action, **values):
                if action == "recover_orphans":
                    return 0
                if action == "claim":
                    return queue.pop(0) if queue else None
                # Результат обязан быть записан до неудачного обращения в Google.
                results = next(root.glob("_state/logs/*/results.csv"))
                with results.open() as source:
                    self.assertIn(values["job_id"], [row["job_id"] for row in csv.DictReader(source)])
                raise ConnectionError("offline")

            environment = {
                "PLAN": "https://script.google.com/macros/s/test/exec",
                "PLAN_GOOGLE_SHEETS_TOKEN": "secret", "MODEL": "m",
                "MODEL_BACKEND": "external", "MODEL_CONTEXT_LIMIT": "24000",
                "BENCHMARK": "gaia", "PARALLEL": "2", "WORKER_NAME": "test", "GPU": "",
            }
            with patch.dict("os.environ", environment, clear=True), patch("runner.worker.ROOT", root), \
                 patch("runner.worker.load_dotenv"), patch("runner.worker.reserve_ports", return_value=nullcontext((18000, 19000))), \
                 patch.object(external, "command", return_value=None), patch.object(Plan, "_call", side_effect=call), \
                 patch("runner.worker.run_job", return_value={"status": "done", "result": "solved", "error": ""}), \
                 ExitStack() as resources:
                self.assertEqual(run_worker(resources), 1)
            with next(root.glob("_state/logs/*/results.csv")).open() as source:
                results = list(csv.DictReader(source))
            self.assertEqual({row["job_id"] for row in results}, {"1", "2"})
            self.assertTrue(all(row["status"] == "done" and row["error"] == "" for row in results))
            self.assertEqual([row["job_id"] for row in queue], ["3"])

    @patch("runner.worker.requests.post")
    def test_google_plan_delegates_atomic_operations(self, post):
        post.return_value.json.return_value = {"ok": True, "result": {"job_id": "1"}}
        with patch.dict("os.environ", {
            "PLAN_API_URL": "https://script.google.com/macros/s/test/exec",
            "PLAN_GOOGLE_SHEETS_TOKEN": "secret",
        }):
            plan = Plan("https://docs.google.com/spreadsheets/d/test/edit")
            self.assertEqual(plan.claim("m", "external", "worker"), {"job_id": "1"})
            plan.heartbeat("1", "worker")
            plan._call("recover_orphans", stale_after=120)
            plan._call("finish", job_id="1", values={"worker_id": "worker", "status": "done"})
        self.assertEqual([call.kwargs["json"]["action"] for call in post.call_args_list],
                         ["claim", "heartbeat", "recover_orphans", "finish"])
        post.return_value.json.return_value = {"ok": False, "error": "unauthorized"}
        with self.assertRaisesRegex(RuntimeError, "unauthorized"):
            plan.claim("m", "external", "worker")

    def test_port_blocks_remain_reserved_and_release(self):
        with tempfile.TemporaryDirectory() as directory, patch("runner.worker.ROOT", Path(directory)), patch("runner.worker.socket.socket"):
            with reserve_ports(100) as ports:
                self.assertEqual(ports, (18000, 19000))
                # Сокеты проверки уже закрыты, но другой worker всё равно не получит эти пары.
                with self.assertRaisesRegex(RuntimeError, "No free"):
                    with reserve_ports(1):
                        self.fail("reserved ports were allocated twice")
            with reserve_ports(100) as ports:
                self.assertEqual(ports, (18000, 19000))

    def test_port_cursor_wraps_and_skips_locked_blocks(self):
        with tempfile.TemporaryDirectory() as directory, patch("runner.worker.ROOT", Path(directory)), patch("runner.worker.socket.socket"):
            with reserve_ports(98):
                with reserve_ports(2) as last:
                    self.assertEqual(last, (18098, 19098))
                # Курсор вернулся к нулю; первые 98 пар всё ещё удерживает первый worker.
                with reserve_ports(2) as last:
                    self.assertEqual(last, (18098, 19098))

    def test_port_probe_failure_releases_partial_block(self):
        def bind(address):
            if address[1] == 18001:
                raise OSError("port occupied")
        with tempfile.TemporaryDirectory() as directory, patch("runner.worker.ROOT", Path(directory)), patch("runner.worker.socket.socket") as socket:
            socket.return_value.__enter__.return_value.bind.side_effect = bind
            with reserve_ports(2) as ports:
                self.assertEqual(ports, (18002, 19002))
                with reserve_ports(1) as following:
                    self.assertEqual(following, (18004, 19004))
            # Первая попытка захватила пару 0, но обязана была её освободить после отказа пары 1.
            (Path(directory) / "_state/ports/cursor").write_text("0")
            with reserve_ports(1) as ports:
                self.assertEqual(ports, (18000, 19000))

    def test_tool_timeouts_come_from_event_duration_only(self):
        tools = [
            {"timestamp": "2026-01-01T00:00:00+00:00", "completed": "2026-01-01T00:03:01+00:00",
             "error": {"type": "timeout"}},
            {"timestamp": "2026-01-01T00:00:00+00:00", "completed": "2026-01-01T00:03:00+00:00"},
        ]
        self.assertEqual(_tool_stats(tools, 180), (361, 1))

    def test_max_context_includes_cached_tokens_and_rejected_request(self):
        events = [
            {"event": "model", "output": {"usage": {
                "input_tokens": 879, "input_tokens_cache_read": 22593,
            }}},
            {"event": "model", "error": (
                "request (28750 tokens) exceeds the available context size (24064 tokens)"
            )},
        ]
        self.assertEqual(_max_context_tokens(events), 28750)
        self.assertTrue(_is_context_overflow(events))

    def test_csv_is_loaded_once_and_never_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            rows = [dict.fromkeys(FIELDS, "") for _ in range(3)]
            for i, row in enumerate(rows, 1):
                row.update(job_id=str(i), model="m", backend="vllm")
            rows[0]["status"] = "done"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            original = path.read_bytes()
            plan = Plan(path)
            self.assertEqual(plan.claim("m", "vllm", "worker")["job_id"], "2")
            plan.heartbeat("2", "worker")
            self.assertEqual(path.read_bytes(), original)
            path.write_text("edited while worker is running")
            self.assertEqual(plan.claim("m", "vllm", "worker")["job_id"], "3")
            self.assertIsNone(plan.claim("m", "vllm", "worker"))
            self.assertEqual(path.read_text(), "edited while worker is running")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_job_id_rejects_invalid_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            row = dict.fromkeys(FIELDS, "")
            row.update(job_id="job/invalid", model="m", backend="vllm")
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "alphanumeric and unique"):
                Plan(path)

    def test_job_id_retries_follow_worker_identity(self):
        row = {"job_id": "", "worker_id": ""}
        with patch("runner.worker.secrets.token_hex", side_effect=["aaaaaaaa", "bbbbbbbb"]):
            self.assertEqual(job_id_for(row, "alice", {"aaaaaaaa"}), "bbbbbbbb")
        row.update(job_id="abcdef12", worker_id="alice")
        self.assertEqual(job_id_for(row, "alice", set()), "abcdef12-2")
        self.assertEqual(job_id_for(row, "bob", set()), "abcdef12")
        row["job_id"] = "abcdef12-9"
        self.assertEqual(job_id_for(row, "alice", set()), "abcdef12-10")

    def test_avg_statistics_are_weighted_and_errors_are_excluded(self):
        avg = dict.fromkeys(("tokens", "time", "tools", "tool_time", "jobs"), 0)
        row = {"result": "unsolved", "error": "", "max_input_tokens": "100", "tool_calls": "2", "_avg_tool_time": 10,
               "_avg": {"avg_output_tokens": 7, "avg_response_time": 2, "avg_usage_complete": True}}
        first = avg_columns(avg, row)
        self.assertEqual(first["avg_tps"], "50.00")
        self.assertEqual(first["avg_tool_time"], "5.00")
        row.update(_avg={"avg_output_tokens": 100, "avg_response_time": 8, "avg_usage_complete": True}, _avg_tool_time=30)
        second = avg_columns(avg, row)
        self.assertEqual(second, {"avg_tps": "20.00", "avg_tool_count": "2.00", "avg_tool_time": "10.00", "jobs_count": "2"})
        row.update(result="timeout", error="job:timeout", _avg={"avg_output_tokens": 999, "avg_response_time": 1, "avg_usage_complete": True})
        self.assertEqual(avg_columns(avg, row), second)

    def test_backends_receive_parallelism(self):
        # Обе команды backend обязаны получить выбранную параллельность эксперимента.
        with patch.dict("os.environ", {"MODEL_CONTEXT_LIMIT": "32768", "GPU": "2"}):
            vllm_command = vllm.command("model", 18080, 4)
            self.assertIn("4", vllm_command)
            self.assertEqual(vllm_command[:2], ["env", "CUDA_VISIBLE_DEVICES=2"])
            llama_command = llama_docker.command("model", 18080, 4)
            self.assertIn("4", llama_command)
            self.assertEqual(llama_command[llama_command.index("--ctx-size") + 1], "131072")

    def test_backend_explicitly_selects_sandbox(self):
        self.assertEqual(sandbox_for_backend("llama_docker"), "docker")
        self.assertEqual(sandbox_for_backend("vllm"), "local")
        self.assertEqual(sandbox_for_backend("external"), "docker")
        self.assertEqual(sandbox_for_backend("external_local"), "local")
        self.assertIs(external_local.command, external.command)
        self.assertIs(external_local.enrich_payload, external.enrich_payload)

    def test_external_backend_starts_local_proxy(self):
        with patch.dict("os.environ", {
            "MODEL_EXTERNAL_API_URL": "https://example.test/v1/",
            "MODEL_EXTERNAL_API_KEY": "secret",
        }, clear=True):
            self.assertEqual(external.command("model", 18023, 3)[-1], "18023")

    def test_backends_translate_thinking_budget(self):
        with patch.dict("os.environ", {"MODEL_THINKING_BUDGET_LIMIT": "2048"}, clear=True):
            self.assertEqual(
                external.enrich_payload({}), {"reasoning": {"max_tokens": 2048}}
            )
            self.assertEqual(vllm.enrich_payload({}), {"thinking_token_budget": 2048})
            self.assertEqual(llama_docker.enrich_payload({}), {"reasoning_budget_tokens": 2048})
        for value in ("", "0", "-1"):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"MODEL_THINKING_BUDGET_LIMIT": value}, clear=True
            ):
                self.assertEqual(external.enrich_payload({}), {})
                self.assertEqual(vllm.enrich_payload({}), {})
                self.assertEqual(llama_docker.enrich_payload({}), {})

    @patch("benchmark.gaia.run.subprocess.run")
    @patch("benchmark.gaia.run.importlib.util.find_spec")
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
            with patch.dict("os.environ", {"XDG_CACHE_HOME": str(Path(directory) / "cache")}):
                prepare_sandbox()
            self.assertFalse(any(call.args[0][:2] == ["docker", "pull"] for call in run.call_args_list))
            marker = Path(directory) / "cache/docker/inspect-gaia.ready"
            self.assertEqual(marker.read_text(), "image@sha256:digest\n")

    def test_csv_feature_order_is_preserved_in_one_variable(self):
        # Scaffold выполняет middleware последовательно, поэтому порядок через запятую значим.
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="m", feature="new, second")
        env = job_environment(row, 0, "http://model/v1", "key", 19080, Path("/tmp/job"), Path("/tmp/runtime"))
        self.assertEqual(env["FEATURES"], "new,second")
        self.assertEqual(env["XDG_RUNTIME_DIR"], "/tmp/runtime")

    def test_feature_environments_are_parsed_per_job(self):
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="m", feature="voi", feature_envs="VOI_TOOL_BUDGET=7,FLAG=yes")
        env = job_environment(row, 0, "http://model/v1", "key", 19080, Path("/tmp/job"), Path("/tmp/runtime"))
        self.assertEqual(env["VOI_TOOL_BUDGET"], "7")
        self.assertEqual(env["FLAG"], "yes")
        self.assertEqual(feature_environment("EMPTY="), {"EMPTY": ""})
        with self.assertRaisesRegex(ValueError, "cannot override FEATURES"):
            feature_environment("FEATURES=voi")

    def test_model_context_limit_is_passed_to_features(self):
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="m", backend="llama_docker")
        with patch.dict("os.environ", {"MODEL_CONTEXT_LIMIT": "131072"}, clear=True):
            env = job_environment(row, 0, "http://model/v1", "key", 19080, Path("/tmp/job"), Path("/tmp/runtime"))
        self.assertEqual(env["MODEL_CONTEXT_LIMIT"], "131072")


if __name__ == "__main__":
    unittest.main()
