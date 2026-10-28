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
from unittest.mock import patch, Mock, AsyncMock

from scaffolds.simple_middleware import features as feature_modules
from scaffolds.simple_middleware.features import common
import httpx
from scaffolds.simple_middleware.app import ModelClient, ModelClientHTTPError, create_app, enabled_features, avg_job_time_limit, Scaffold
from benchmarks.gaia.run import timed_tool, guard_repetition, JobMetrics, selected_tools
from inspect_ai.tool import ToolDef, ToolError


from dotenv import dotenv_values

DEFAULTS = {
    key: value for path in Path(__file__).resolve().parents[1].rglob(".env.default")
    if "_state" not in path.parts and ".venv" not in path.parts
    for key, value in dotenv_values(path, interpolate=False).items()
}


class ScaffoldTest(unittest.TestCase):
    def test_empty_feature_passes_requests_and_responses_without_extra_model_calls(self):
        async def check():
            with patch.dict(os.environ, DEFAULTS | {"SCAFFOLD_FEATURES": "empty"}, clear=True):
                client = Mock(context_limit=65536)
                scaffold = Scaffold(client)
                self.assertEqual([f.feature_name() for f in scaffold.features], ["empty"])
                payloads = [
                    {"messages": [{"role": "user", "content": "task"}]},
                    {"messages": [{"role": "user", "content": "task"},
                                  {"role": "tool", "tool_call_id": "one", "content": "result"}]},
                ]
                for tool_calls in [[], [{"id": "next", "function": {"name": "browser", "arguments": "{}"}}]]:
                    for payload in payloads:
                        response = {"choices": [{"message": {"content": "answer", "tool_calls": tool_calls}}]}
                        client.chat_completions = AsyncMock(return_value=response)
                        self.assertIs(await scaffold.chat_completions(payload), response)
                        client.chat_completions.assert_awaited_once_with(payload)
                        self.assertIs(client.chat_completions.call_args.args[0], payload)
        asyncio.run(check())

    def test_scaffold_preserves_openai_error_message(self):
        from openai import AsyncOpenAI, BadRequestError
        async def check():
            detail = "request (69679 tokens) exceeds the available context size (65536 tokens)"
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DEFAULTS | {
                "BACKEND_MODEL_BASE_URL": "http://backend/v1", "WORKER_MODEL": "m",
                "WORKER_BACKEND": "external", "SCAFFOLD_FEATURES": "",
            }), patch("scaffolds.simple_middleware.app.ModelClient.chat_completions", side_effect=ModelClientHTTPError(400, detail)):
                app = create_app(Path(directory))
                async with AsyncOpenAI(base_url="http://scaffold/v1", api_key="local", http_client=httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                )) as client:
                    with self.assertRaises(BadRequestError) as caught:
                        await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "test"}])
                self.assertEqual(caught.exception.body["message"], detail)
                self.assertIn(detail, str(caught.exception))
        asyncio.run(check())

    def test_scaffold_uses_worker_model_local_key_and_explicit_log_directory(self):
        async def check():
            requests = []
            def respond(request):
                requests.append(request)
                return httpx.Response(200, json={
                    "choices": [{"message": {"role": "assistant", "content": "answer"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                })
            client_class = httpx.AsyncClient
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DEFAULTS | {
                "BACKEND_MODEL_BASE_URL": "http://backend/v1", "WORKER_MODEL": "actual-model",
                "WORKER_MODEL_ALIAS": "plan-alias", "WORKER_BACKEND": "external", "SCAFFOLD_FEATURES": "",
            }, clear=True), patch("scaffolds.simple_middleware.app.httpx.AsyncClient",
                                 side_effect=lambda **kw: client_class(transport=httpx.MockTransport(respond), **kw)):
                app = create_app(Path(directory))
                async with client_class(transport=httpx.ASGITransport(app=app), base_url="http://scaffold") as client:
                    models = await client.get("/v1/models")
                    self.assertEqual(models.json()["data"][0]["id"], "actual-model")
                    response = await client.post("/v1/chat/completions", json={
                        "model": "ignored-wrapper-name", "messages": [{"role": "user", "content": "question"}],
                    })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["model"], "actual-model")
                self.assertEqual(str(requests[0].url), "http://backend/v1/chat/completions")
                self.assertEqual(requests[0].headers["Authorization"], "Bearer local")
                self.assertEqual(json.loads(requests[0].content)["model"], "actual-model")
                self.assertTrue((Path(directory) / "scaffold.json").is_file())
                self.assertTrue((Path(directory) / "model-responses.jsonl").is_file())
        asyncio.run(check())

    def test_scaffold_module_accepts_port_and_log_directory_as_arguments(self):
        import runpy
        import sys
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DEFAULTS | {
            "BACKEND_MODEL_BASE_URL": "http://backend/v1", "WORKER_MODEL": "m",
            "WORKER_BACKEND": "external", "SCAFFOLD_FEATURES": "",
        }, clear=True), patch.object(sys, "argv", ["scaffold", "18023", directory]), patch("uvicorn.run") as serve:
            runpy.run_module("scaffolds.simple_middleware", run_name="__main__")
            self.assertEqual(serve.call_args.kwargs["port"], 18023)
            self.assertTrue((Path(directory) / "scaffold.json").is_file())

    def test_failed_model_request_reports_time_and_rejected_context(self):
        async def check():
            transport = httpx.MockTransport(lambda request: httpx.Response(400, json={"error": {"message": "request (28750 tokens) exceeds the available context size (24000 tokens)"}}))
            client_class = httpx.AsyncClient
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DEFAULTS | {
                "BACKEND_MODEL_BASE_URL": "http://model/v1",
                "WORKER_MODEL": "m", "WORKER_BACKEND": "external",
                "WORKER_MODEL_CONTEXT_LIMIT": "24000",
            }), patch("scaffolds.simple_middleware.app.httpx.AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw)):
                client = ModelClient(Path(directory))
                with self.assertRaises(ModelClientHTTPError):
                    await client.chat_completions({"messages": []})
                model = json.loads((Path(directory) / "scaffold.json").read_text())["model"]
                self.assertEqual(model["calls"], 1)
                self.assertEqual(model["max_input_tokens"], 28750)
                self.assertGreater(model["time_sum_seconds"], 0)
                self.assertFalse(model["usage_complete"])
        asyncio.run(check())

    def test_tool_events_replace_running_record_and_final_report_has_raw_calls(self):
        from datetime import datetime, timezone, timedelta
        from inspect_ai.event import ToolEvent
        from inspect_ai.tool import ToolCallError
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DEFAULTS | {"INSPECT_LOG_DIR": str(Path(directory) / "inspect-logs")}):
            hook = JobMetrics()
            start = datetime.now(timezone.utc)
            event = ToolEvent(id="call", function="echo", arguments={"text": "Привет"}, timestamp=start)
            asyncio.run(hook.on_sample_event(SimpleNamespace(event=event)))
            calls = json.loads((Path(directory) / "calls.json").read_text())
            self.assertEqual(calls[0]["status"], "running")
            self.assertIsNone(calls[0]["response_bytes"])
            event.completed = start + timedelta(seconds=121)
            event.error = ToolCallError(type="timeout", message="test")
            event.result = "response"
            asyncio.run(hook.on_sample_event(SimpleNamespace(event=event)))
            sample = SimpleNamespace(events=[event], scores={"answer": SimpleNamespace(value="C")}, error=None)
            asyncio.run(hook.on_sample_end(SimpleNamespace(sample=sample)))
            data = json.loads((Path(directory) / "benchmark.json").read_text())
            self.assertEqual(data["result"], "solved")
            self.assertNotIn("tool_calls", data)
            calls = json.loads((Path(directory) / "calls.json").read_text())
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["duration_seconds"], 121)
            self.assertGreater(calls[0]["request_bytes"], 0)
            self.assertGreater(calls[0]["response_bytes"], 0)
            for code in ("repeated_actions", "no_tool_calls"):
                sample.error = SimpleNamespace(message=f"benchmark:{code}: Agent stalled")
                asyncio.run(hook.on_sample_end(SimpleNamespace(sample=sample)))
                self.assertEqual(hook.error, f"benchmark:{code}")
            detail = "request (69679 tokens) exceeds the available context size (65536 tokens)"
            sample.events = [SimpleNamespace(event="model", model_dump=lambda **_: {"event": "model", "error": detail})]
            sample.error = SimpleNamespace(message=detail)
            asyncio.run(hook.on_sample_end(SimpleNamespace(sample=sample)))
            self.assertEqual(hook.result, "context_overflow")
            self.assertEqual(hook.error, "model:context_overflow")

    def test_gaia_report_preserves_scalar_score_and_normalizes_inspect_categories(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DEFAULTS | {"INSPECT_LOG_DIR": str(Path(directory) / "inspect-logs")}):
            hook = JobMetrics()
            for native, expected in [("C", 1), ("I", 0), ("P", 0.5), (0.375, 0.375)]:
                sample = SimpleNamespace(events=[], scores={"answer": SimpleNamespace(value=native)}, error=None)
                asyncio.run(hook.on_sample_end(SimpleNamespace(sample=sample)))
                self.assertEqual(json.loads((Path(directory) / "benchmark.json").read_text())["score"], expected)
            sample.error = SimpleNamespace(message="interrupted")
            asyncio.run(hook.on_sample_end(SimpleNamespace(sample=sample)))
            self.assertIsNone(hook.score)

    def test_bash_reports_silent_failures_without_changing_command_or_description(self):
        from benchmarks.gaia.run import bash_with_status
        from inspect_ai.tool import bash
        from inspect_ai.util import ExecResult
        tool = bash_with_status(120)
        self.assertEqual(ToolDef(tool).parameters, ToolDef(bash(120)).parameters)
        self.assertEqual(ToolDef(tool).description, ToolDef(bash(120)).description)
        command = 'curl -s "https://invalid url/"'
        async def check():
            engine = SimpleNamespace(exec=AsyncMock(return_value=ExecResult(success=False, returncode=3, stdout='', stderr='')))
            with patch('benchmarks.gaia.run.sandbox', return_value=engine):
                with self.assertRaisesRegex(ToolError, 'exited with code 3'):
                    await tool(command)
                engine.exec.assert_awaited_once_with(cmd=['bash', '--login', '-c', command], timeout=120)
                engine.exec.return_value = ExecResult(success=True, returncode=0, stdout='', stderr='')
                self.assertEqual(await tool('true'), '')
                engine.exec.return_value = ExecResult(success=False, returncode=2, stdout='partial', stderr='failure')
                with self.assertRaisesRegex(ToolError, 'failure\npartial'):
                    await tool('test')
        asyncio.run(check())

    def test_tool_selection_preserves_order_and_uses_benchmark_defaults(self):
        with patch.dict(os.environ, DEFAULTS | {"BENCHMARK_TOOLS": "python,bash"}):
            self.assertEqual([ToolDef(tool).name for tool in selected_tools(120)], ["python", "bash"])
        with patch.dict(os.environ, DEFAULTS | {"BENCHMARK_TOOLS": "", "BENCHMARK_GAIA_TOOLS": "bash"}):
            self.assertEqual([ToolDef(tool).name for tool in selected_tools(120)], ["bash"])

    def test_gaia_dataset_revision_is_pinned(self):
        from importlib.metadata import version
        from inspect_evals.gaia.gaia import DATASET_REVISION

        self.assertEqual(version("inspect-evals"), "0.25.0")
        self.assertEqual(DATASET_REVISION, "682dd723ee1e1697e00360edccf2366dc8418dd9")

    def test_gaia_system_instructions_match_upstream(self):
        import ast
        import inspect
        from textwrap import dedent
        from inspect_evals.gaia.gaia import default_solver
        from benchmarks.gaia.run import _gaia
        def instructions(function):
            tree = ast.parse(inspect.getsource(function))
            prompt = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "AgentPrompt")
            value = next(keyword.value for keyword in prompt.keywords if keyword.arg == "instructions")
            return dedent(ast.literal_eval(value.args[0]))
        self.assertEqual(instructions(_gaia), instructions(default_solver))

    def test_no_tool_guard_counts_consecutive_responses_per_sample(self):
        def message(role="assistant", tool_calls=None, text="answer"):
            return SimpleNamespace(role=role, tool_calls=tool_calls, text=text)
        with patch.dict(os.environ, DEFAULTS | {}, clear=True):
            state = SimpleNamespace(messages=[message() for _ in range(4)])
            self.assertIs(asyncio.run(guard_repetition(state)), True)
            state.messages.extend([message("user"), message()])
            with self.assertRaisesRegex(RuntimeError, "5 consecutive"):
                asyncio.run(guard_repetition(state))
            call = SimpleNamespace(id="bash", function="bash", arguments={"command": "pwd"})
            state.messages.extend([message(tool_calls=[call]), message()])
            self.assertTrue(asyncio.run(guard_repetition(state)))
            self.assertTrue(asyncio.run(guard_repetition(SimpleNamespace(messages=[]))))
            with self.assertRaisesRegex(RuntimeError, "benchmark:no_tool_calls"):
                asyncio.run(guard_repetition(SimpleNamespace(messages=[message(text=str(i)) for i in range(5)])))
        with patch.dict(os.environ, DEFAULTS | {"BENCHMARK_GAIA_MAX_CONSECUTIVE_REPEATS": "1"}):
            with self.assertRaisesRegex(RuntimeError, "1 consecutive"):
                asyncio.run(guard_repetition(SimpleNamespace(messages=[message()])))

    def test_repeat_guard_distinguishes_progress_from_identical_tool_results(self):
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
        from inspect_ai.tool import ToolCall

        def history(outputs, parallel=False):
            messages = []
            for i, output in enumerate(outputs):
                calls = [ToolCall(id=f"scroll-{i}", function="web_browser_scroll", arguments={"direction": "down"})]
                if parallel:
                    calls.append(ToolCall(id=f"bash-{i}", function="bash", arguments={"command": "pwd"}))
                messages.append(ChatMessageAssistant(content="", tool_calls=calls))
                if parallel:
                    messages.append(ChatMessageTool(tool_call_id=f"bash-{i}", content="/workspace"))
                messages.append(ChatMessageTool(tool_call_id=f"scroll-{i}", content=output))
            return SimpleNamespace(messages=messages)

        with patch.dict(os.environ, DEFAULTS, clear=True):
            for parallel in (False, True):
                with self.subTest(parallel=parallel):
                    self.assertTrue(asyncio.run(guard_repetition(history([f"Page {i}" for i in range(10)], parallel))))
                    with self.assertRaisesRegex(RuntimeError, "benchmark:repeated_actions"):
                        asyncio.run(guard_repetition(history(["Page bottom"] * 5, parallel)))
                    self.assertTrue(asyncio.run(guard_repetition(history(["Old"] * 5 + ["New"] * 4, parallel))))
            pending = history(["Page bottom"] * 4)
            pending.messages.append(ChatMessageAssistant(content="", tool_calls=[
                ToolCall(id="pending", function="web_browser_scroll", arguments={"direction": "down"}),
            ]))
            self.assertTrue(asyncio.run(guard_repetition(pending)))


    def test_repeat_guard_detects_multi_action_cycles_without_rejecting_progress(self):
        from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
        from inspect_ai.tool import ToolCall
        def history(progress):
            messages = []
            for i in range(15):
                messages.append(ChatMessageAssistant(content="", tool_calls=[
                    ToolCall(id=str(i), function="bash", arguments={"command": f"command-{i % 3}"}),
                ]))
                messages.append(ChatMessageTool(tool_call_id=str(i), content=f"result-{i if progress else i % 3}"))
            return SimpleNamespace(messages=messages)
        with patch.dict(os.environ, DEFAULTS, clear=True):
            with self.assertRaisesRegex(RuntimeError, "period=3"):
                asyncio.run(guard_repetition(history(False)))
            self.assertTrue(asyncio.run(guard_repetition(history(True))))

    def test_avg_timeout_defaults_factors_and_validation(self):
        with patch.dict(os.environ, DEFAULTS | {}, clear=True):
            self.assertEqual(avg_job_time_limit([], 24000), 2300)
            handlers = [SimpleNamespace(avg_token_factor=lambda: 1.5), SimpleNamespace(avg_token_factor=lambda: 2)]
            self.assertEqual(avg_job_time_limit(handlers, 24000), 3900)
        with patch.dict(os.environ, DEFAULTS | {"SCAFFOLD_AVG_TOKENS_PER_SECOND": "0"}):
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
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DEFAULTS | {
                "BACKEND_MODEL_BASE_URL": "http://model/v1", "WORKER_MODEL": "m",
                "WORKER_BACKEND": "external", "WORKER_MODEL_CONTEXT_LIMIT": "24000",
            }), patch("scaffolds.simple_middleware.app.httpx.AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw)):
                client = ModelClient(Path(directory))
                await client.chat_completions({"messages": []})
                await client.chat_completions({"messages": []})
                stats = json.loads((Path(directory) / "scaffold.json").read_text())["model"]
                raw = [json.loads(line) for line in (Path(directory) / "model-responses.jsonl").read_text().splitlines()]
                self.assertEqual(json.loads(raw[0]["response_body"])["usage"]["completion_tokens"], 20)
                self.assertNotIn("Authorization", raw[0])
                self.assertEqual(stats["output_tokens_sum"], 30)
                self.assertEqual(stats["max_input_tokens"], 1234)
                self.assertGreater(stats["time_sum_seconds"], 0)
                self.assertTrue(stats["usage_complete"])
                await client.chat_completions({"messages": []})
                self.assertFalse(client.model_stats["usage_complete"])
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
        with patch.dict(os.environ, DEFAULTS | {"WORKER_MODEL_CONTEXT_LIMIT": "32768"}):
            for module in pkgutil.iter_modules(
                feature_modules.__path__, feature_modules.__name__ + "."
            ):
                if module.name.endswith(".common") or module.name.rsplit(".", 1)[-1].startswith("-"):
                    continue
                names.append(importlib.import_module(module.name).Breakpoints.feature_name())
        self.assertIn("budget_tracker", names)
        self.assertEqual(len(names), len(set(names)))

    def test_budget_tracker_preserves_previous_prompt_and_does_not_duplicate_status(self):
        with patch.dict(os.environ, DEFAULTS | {"WORKER_MODEL_CONTEXT_LIMIT": "65536"}):
            tracker = importlib.import_module("scaffolds.simple_middleware.features.budget_tracker").Breakpoints()
        tracker.before_task_call({"messages": []})
        history = [{"role": "user", "content": "task"},
                   {"role": "assistant", "content": "call one"},
                   {"role": "tool", "tool_call_id": "one", "content": "result one"}]
        first = tracker.before_chat_message({"messages": history})
        self.assertEqual(history[-1]["content"], "result one")
        tracker.store.update(tool_calls=2, prompt_tokens=1000, completion_tokens=100)
        history += [{"role": "assistant", "content": "call two"},
                    {"role": "tool", "tool_call_id": "two", "content": "result two"}]
        second = tracker.before_chat_message({"messages": history})
        self.assertEqual(second["messages"][:len(first["messages"])], first["messages"])
        self.assertNotEqual(second["messages"][2]["content"], second["messages"][4]["content"])
        self.assertEqual(tracker.before_chat_message(second), second)
        tracker.after_task_call({"choices": [{"message": {"content": "continue"}}]})
        followup = tracker.before_chat_message({"messages": history + [{"role": "assistant", "content": "continue"}]})
        self.assertEqual(followup["messages"][:-1], second["messages"])
        tracker.before_task_call({"messages": []})
        self.assertEqual(tracker.store["tool_statuses"], {})

    def test_common_module_is_not_loaded_as_feature(self):
        with patch.dict(os.environ, DEFAULTS | {"SCAFFOLD_FEATURES": "budget_tracker", "WORKER_MODEL_CONTEXT_LIMIT": "32768"}):
            handlers = enabled_features()
        self.assertEqual([handler.feature_name() for handler in handlers], ["budget_tracker"])

    def test_csv_feature_order_is_handler_order(self):
        with patch.dict(os.environ, DEFAULTS | {"SCAFFOLD_FEATURES": "voi,budget_tracker", "WORKER_MODEL_CONTEXT_LIMIT": "32768"}):
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
