"""Регресс-тесты CSV, параллельности, выбора sandbox и порядка фич."""

import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from runner.backends import llama_docker, vllm
from runner.metrics import _tool_stats
from runner.start import FIELDS, Plan, job_environment, sandbox_for_backend


class RunnerTest(unittest.TestCase):
    def test_tool_timeouts_come_from_event_duration_only(self):
        tools = [
            {"timestamp": "2026-01-01T00:00:00+00:00", "completed": "2026-01-01T00:03:01+00:00",
             "error": {"type": "timeout"}},
            {"timestamp": "2026-01-01T00:00:00+00:00", "completed": "2026-01-01T00:03:00+00:00"},
        ]
        self.assertEqual(_tool_stats(tools, 180), (361, 1))

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
            self.assertIn("4", llama_docker.command("model", 18080, 4))

    def test_docker_suffix_selects_docker_sandbox(self):
        # Соглашение об имени охватывает будущие адаптеры без поддерживаемого вручную списка.
        self.assertEqual(sandbox_for_backend("llama_docker"), "docker")
        self.assertEqual(sandbox_for_backend("future_docker"), "docker")
        self.assertEqual(sandbox_for_backend("vllm"), "local")

    def test_csv_feature_order_is_preserved_in_one_variable(self):
        # Scaffold выполняет middleware последовательно, поэтому порядок через запятую значим.
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="m", feature="new, second")
        env = job_environment(row, 0, 18080, 19080, Path("/tmp/job"))
        self.assertEqual(env["FEATURES"], "new,second")

    def test_model_context_limit_is_passed_to_features(self):
        row = dict.fromkeys(FIELDS, "")
        row.update(job_id="1", model="m", backend="llama_docker")
        with patch.dict("os.environ", {"MODEL_CONTEXT_LIMIT": "131072"}, clear=True):
            env = job_environment(row, 0, 18080, 19080, Path("/tmp/job"))
        self.assertEqual(env["MODEL_CONTEXT_LIMIT"], "131072")


if __name__ == "__main__":
    unittest.main()
