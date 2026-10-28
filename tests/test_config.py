"""Проверяем единственный порядок загрузки и отсутствие скрытых defaults в коде."""

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backends import external
from dotenv import load_dotenv
from worker.worker import ROOT, load_config


class ConfigTest(unittest.TestCase):
    def test_python_bootstrap_installs_once_and_refreshes_changed_requirements(self):
        import importlib
        bootstrap = importlib.import_module("worker.__main__")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            script = root / "worker/__main__.py"
            script.parent.mkdir()
            script.write_text("bootstrap")
            requirements = root / "requirements.txt"
            requirements.write_text("package==1")
            benchmark = root / "benchmarks/gaia/requirements.txt"
            benchmark.parent.mkdir(parents=True)
            benchmark.write_text("gaia==1")
            def create(path):
                (path / "bin").mkdir(parents=True, exist_ok=True)
                (path / "bin/python").touch()
            with patch.object(bootstrap, "__file__", str(script)), \
                 patch.object(bootstrap.venv, "EnvBuilder") as builder, \
                 patch.object(bootstrap.subprocess, "run") as install, \
                 patch.object(bootstrap.os, "execv") as execute:
                builder.return_value.create.side_effect = create
                bootstrap.main()
                bootstrap.main()
                self.assertEqual(install.call_count, 1)
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(execute.call_args.args[1], [str(root / ".venv/bin/python"), "-m", "worker.worker"])
                self.assertIn(str(benchmark), install.call_args.args[0])
                ready = root / "_state/runtime/python/install.ready"
                previous = ready.read_text()
                requirements.write_text("package==2")
                bootstrap.main()
                self.assertEqual(install.call_count, 2)
                self.assertNotEqual(ready.read_text(), previous)

    def test_worker_defaults_use_requested_names_and_small_model(self):
        from dotenv import dotenv_values
        defaults = dotenv_values(ROOT / "worker/.env.default", interpolate=False)
        self.assertEqual(defaults["WORKER_BACKEND"], "llama")
        self.assertEqual(defaults["WORKER_MODEL"], "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M")
        self.assertEqual(defaults["WORKER_MODEL_CONTEXT_LIMIT"], "24000")
        self.assertEqual(defaults["WORKER_GPU_NUM"], "")
        self.assertNotIn("WORKER_PLAN", defaults)
        self.assertFalse(any(name.startswith("WORKER_MAIN_") for name in defaults))

    def test_shell_then_root_then_component_defaults(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            values = {
                "worker": "WORKER_BACKEND=external\nWORKER_SCAFFOLD=custom\nWORKER_GPU_NUM=0\n",
                "backends/external": "BACKEND_EXTERNAL_API_URL=https://default.test/v1\nBACKEND_EXTERNAL_THINKING_BUDGET_LIMIT=\n",
                "benchmarks/gaia": "BENCHMARK_GAIA_TOOL_TIMEOUT=120\n",
                "benchmarks": "BENCHMARK_TOOLS=\n",
                "scaffolds": "SCAFFOLD_AVG_TOKENS_PER_SECOND=30\nSCAFFOLD_FEATURES=\n",
                "scaffolds/custom": "CUSTOM_SCAFFOLD_SETTING=loaded\n",
                "scaffolds/custom/features": "",
            }
            for component, content in values.items():
                path = root / component / ".env.default"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            (root / ".env").write_text("WORKER_GPU_NUM=2\nBACKEND_EXTERNAL_API_URL=https://root.test/v1\n")
            # Пользовательские .env внутри компонентов больше не читаются.
            (root / "backends/external/.env").write_text("BENCHMARK_GAIA_TOOL_TIMEOUT=999\n")
            os.environ["WORKER_GPU_NUM"] = "3"
            with patch("worker.worker.ROOT", root):
                load_config()
            self.assertEqual(os.environ["WORKER_GPU_NUM"], "3")
            self.assertEqual(os.environ["BACKEND_EXTERNAL_API_URL"], "https://root.test/v1")
            self.assertEqual(os.environ["BENCHMARK_GAIA_TOOL_TIMEOUT"], "120")
            self.assertEqual(os.environ["CUSTOM_SCAFFOLD_SETTING"], "loaded")
            self.assertEqual(os.environ["SCAFFOLD_AVG_TOKENS_PER_SECOND"], "30")
            self.assertEqual(os.environ["SCAFFOLD_FEATURES"], "")
            self.assertEqual(external.enrich_payload({}), {})
            del os.environ["BACKEND_EXTERNAL_THINKING_BUDGET_LIMIT"]
            with self.assertRaises(KeyError):
                external.enrich_payload({})

    def test_interpolation_respects_shell_overrides(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XDG_CACHE_HOME": "/custom/cache"}, clear=True):
            path = Path(directory) / ".env.default"
            path.write_text("XDG_CACHE_HOME=/default/cache\nHF_HOME=${XDG_CACHE_HOME}/huggingface\n")
            load_dotenv(path)
            self.assertEqual(os.environ["HF_HOME"], "/custom/cache/huggingface")

    def test_application_has_no_environment_fallback_arguments(self):
        for directory in ["worker", "backends", "benchmarks", "scaffolds"]:
            for path in (ROOT / directory).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                        continue
                    function = ast.unparse(node.func)
                    if function in {"os.getenv", "os.environ.get"}:
                        self.assertLessEqual(len(node.args), 1, str(path))


if __name__ == "__main__":
    unittest.main()
