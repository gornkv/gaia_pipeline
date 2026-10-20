"""Регресс-тесты CSV, параллельности, выбора sandbox и порядка фич."""

import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from benchmark.gaia.run import prepare_sandbox
from runner.backends import external, llama_docker, vllm
from runner.metrics import _is_context_overflow, _max_context_tokens, _tool_stats
from runner.start import (
    FIELDS, Plan, feature_environment, job_environment, sandbox_for_backend,
)


class RunnerTest(unittest.TestCase):
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

    def test_claim_and_finish_preserve_other_rows(self):
        # Результат сливается со свежим CSV и не должен затирать посторонние ручные правки.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            rows = [dict.fromkeys(FIELDS, "") for _ in range(2)]
            rows[0].update(job_id="1", model="m", backend="vllm", sample_no="1")
            rows[1].update(job_id="2", model="other", backend="vllm", sample_no="2")
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            plan = Plan(path)
            self.assertEqual(plan.claim("m", "vllm", "worker-1")["job_id"], "1")
            plan.finish("1", {"status": "done", "tokens_used": "12"})
            with path.open() as source:
                saved = list(csv.DictReader(source))
            self.assertEqual(saved[0]["status"], "done")
            self.assertEqual(saved[0]["tokens_used"], "12")
            self.assertEqual(saved[1]["status"], "")

    def test_stale_running_job_becomes_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            row = dict.fromkeys(FIELDS, "")
            row.update(
                job_id="1", model="m", backend="vllm", status="running", worker_id="dead",
                heartbeat_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
            )
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row)
            plan = Plan(path)
            self.assertEqual(plan.recover_orphans(120), 1)
            with path.open() as source:
                saved = next(csv.DictReader(source))
            self.assertEqual(saved["status"], "interrupted")
            self.assertEqual(saved["error"], "1")

    def test_job_id_must_be_an_ascii_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            row = dict.fromkeys(FIELDS, "")
            row.update(job_id="job-1", model="m", backend="vllm")
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "numeric and unique"):
                Plan(path)._read()

    def test_backends_receive_parallelism(self):
        # Обе команды backend обязаны получить выбранную параллельность эксперимента.
        with patch.dict("os.environ", {"MODEL_CONTEXT_LIMIT": "32768"}):
            self.assertIn("4", vllm.command("model", 18080, 4))
            llama_command = llama_docker.command("model", 18080, 4)
            self.assertIn("4", llama_command)
            self.assertEqual(llama_command[llama_command.index("--ctx-size") + 1], "131072")

    def test_backend_explicitly_selects_sandbox(self):
        self.assertEqual(sandbox_for_backend("llama_docker"), "docker")
        self.assertEqual(sandbox_for_backend("vllm"), "local")
        self.assertEqual(sandbox_for_backend("external"), "docker")

    def test_external_backend_uses_url_and_key_without_command(self):
        with patch.dict("os.environ", {
            "MODEL_API_BASE_URL": "https://example.test/v1/",
            "MODEL_API_KEY": "secret",
        }, clear=True):
            self.assertIsNone(external.command("model", 0, 3))
            self.assertEqual(external.base_url(0), "https://example.test/v1")
            self.assertEqual(external.api_key(), "secret")
        with patch.dict("os.environ", {"BASE_MODEL_API_BASE_URL": "https://old.test/v1/"}, clear=True):
            self.assertEqual(external.base_url(0), "https://old.test/v1")

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
