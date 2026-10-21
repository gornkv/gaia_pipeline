"""Проверки общей библиотеки фич после упрощения структуры scaffold."""

import importlib
import asyncio
import json
import os
import pkgutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import features as feature_modules
from features import common
import httpx
from runner.scaffold import ModelClient, enabled_features, avg_job_time_limit
from benchmark.gaia.run import timed_tool, guard_repetition
from inspect_ai.tool import ToolDef, ToolError


class ScaffoldTest(unittest.TestCase):
    def test_gaia_dataset_revision_is_pinned(self):
        from importlib.metadata import version
        from inspect_evals.gaia.gaia import DATASET_REVISION

        self.assertEqual(version("inspect-evals"), "0.25.0")
        self.assertEqual(DATASET_REVISION, "682dd723ee1e1697e00360edccf2366dc8418dd9")

    def test_no_tool_guard_counts_consecutive_responses_per_sample(self):
        def message(role="assistant", tool_calls=None, text="answer"):
            return SimpleNamespace(role=role, tool_calls=tool_calls, text=text)
        with patch.dict(os.environ, {}, clear=True):
            state = SimpleNamespace(messages=[message() for _ in range(4)])
            self.assertTrue(asyncio.run(guard_repetition(state)))
            state.messages.extend([message("user"), message()])
            with self.assertRaisesRegex(RuntimeError, "5 consecutive"):
                asyncio.run(guard_repetition(state))
            call = SimpleNamespace(function="bash", arguments={"command": "pwd"})
            state.messages.extend([message(tool_calls=[call]), message()])
            self.assertTrue(asyncio.run(guard_repetition(state)))
            self.assertTrue(asyncio.run(guard_repetition(SimpleNamespace(messages=[]))))
            with self.assertRaisesRegex(RuntimeError, "identical actions"):
                asyncio.run(guard_repetition(SimpleNamespace(messages=[message(tool_calls=[call], text=str(i)) for i in range(5)])))
            with self.assertRaisesRegex(RuntimeError, "without tool calls"):
                asyncio.run(guard_repetition(SimpleNamespace(messages=[message(text=str(i)) for i in range(5)])))
        with patch.dict(os.environ, {"MAX_CONSECUTIVE_REPEATS": "1"}):
            with self.assertRaisesRegex(RuntimeError, "1 consecutive"):
                asyncio.run(guard_repetition(SimpleNamespace(messages=[message()])))

    def test_avg_timeout_defaults_factors_and_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(avg_job_time_limit([], 24000), 2300)
            handlers = [SimpleNamespace(avg_token_factor=lambda: 1.5), SimpleNamespace(avg_token_factor=lambda: 2)]
            self.assertEqual(avg_job_time_limit(handlers, 24000), 3900)
        with patch.dict(os.environ, {"AVG_TOKENS_PER_SECOND": "0"}):
            with self.assertRaises(ValueError):
                avg_job_time_limit([], 24000)

    def test_model_statistics_include_reasoning_once_and_all_requests(self):
        async def check():
            responses = [
                {"usage": {"prompt_tokens": 1234, "completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 4}}},
                {"usage": {"input_tokens": 100, "output_tokens": 10}},
                {},
            ]
            transport = httpx.MockTransport(lambda request: httpx.Response(200, json=responses.pop(0)))
            client_class = httpx.AsyncClient
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "BASE_MODEL_API_BASE_URL": "http://model/v1", "BASE_MODEL_NAME": "m",
                "MODEL_BACKEND": "external", "MODEL_CONTEXT_LIMIT": "24000", "JOB_STATE_DIR": directory,
            }), patch("runner.scaffold.httpx.AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw)):
                client = ModelClient()
                await client.chat_completions({"messages": []})
                await client.chat_completions({"messages": []})
                stats = json.loads((Path(directory) / "avg.json").read_text())
                raw = [json.loads(line) for line in (Path(directory) / "model-responses.jsonl").read_text().splitlines()]
                self.assertEqual(json.loads(raw[0]["response_body"])["usage"]["completion_tokens"], 20)
                self.assertNotIn("Authorization", raw[0])
                self.assertEqual(stats["avg_output_tokens"], 30)
                self.assertEqual(stats["max_input_tokens"], 1234)
                self.assertGreater(stats["avg_response_time"], 0)
                self.assertTrue(stats["avg_usage_complete"])
                await client.chat_completions({"messages": []})
                self.assertFalse(client.avg_stats["avg_usage_complete"])
        asyncio.run(check())

    def test_tool_timeout_is_an_error_visible_to_model(self):
        async def slow_tool(value: str) -> str:
            """A tool that takes time.

            Args:
                value: Text to return.
            """
            await asyncio.sleep(1)
            return value
        bounded = timed_tool(slow_tool, 0.01)
        self.assertEqual(ToolDef(bounded).parameters, ToolDef(slow_tool).parameters)
        with self.assertRaisesRegex(ToolError, "timed out"):
            asyncio.run(bounded("test"))

    def test_all_features_import(self):
        # Динамическое обнаружение — единственная связь core с отдельными файлами фич.
        names = []
        with patch.dict(os.environ, {"MODEL_CONTEXT_LIMIT": "32768"}):
            for module in pkgutil.iter_modules(
                feature_modules.__path__, feature_modules.__name__ + "."
            ):
                if module.name.endswith(".common") or module.name.rsplit(".", 1)[-1].startswith("-"):
                    continue
                names.append(importlib.import_module(module.name).Breakpoints.feature_name())
        self.assertIn("budget_tracker", names)
        self.assertEqual(len(names), len(set(names)))

    def test_common_module_is_not_loaded_as_feature(self):
        with patch.dict(os.environ, {"FEATURES": "budget_tracker", "MODEL_CONTEXT_LIMIT": "32768"}):
            handlers = enabled_features()
        self.assertEqual([handler.feature_name() for handler in handlers], ["budget_tracker"])

    def test_csv_feature_order_is_handler_order(self):
        with patch.dict(os.environ, {"FEATURES": "voi,budget_tracker", "MODEL_CONTEXT_LIMIT": "32768"}):
            handlers = enabled_features()
        self.assertEqual([handler.feature_name() for handler in handlers], ["voi", "budget_tracker"])

    def test_branch_context_keeps_only_current_branch(self):
        context = common.BranchContextManager()
        context.start({"messages": [{"role": "user", "content": "task"}]})
        context.after_chat_message({
            "choices": [{"message": {
                "role": "assistant", "tool_calls": [{"id": "new"}],
            }}]
        })
        context.after_tool_call({"messages": [
            {"role": "tool", "tool_call_id": "old", "content": "old"},
            {"role": "tool", "tool_call_id": "new", "content": "new"},
        ]})
        isolated = context.before_chat_message({"messages": [{"role": "user", "content": "noise"}]})
        self.assertEqual([item.get("content") for item in isolated["messages"]], ["task", None, "new"])


if __name__ == "__main__":
    unittest.main()
