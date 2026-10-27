#!/usr/bin/env python3
"""GAIA task definitions и CLI-адаптер для общего benchmark worker."""

import asyncio
import functools
import fcntl
import importlib.util
import json
import os
import re
from datetime import datetime
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.hooks import Hooks, SampleEnd, SampleEvent, hooks
from inspect_ai.agent import AgentPrompt, AgentSubmit, react
from inspect_ai.tool import bash, python, web_browser, ToolError
from inspect_evals.gaia.gaia import DEFAULT_DOCKER_SANDBOX, gaia as upstream_gaia

ROOT = Path(__file__).resolve().parents[2]


@hooks(name="gaia_metrics", description="Передаёт результат benchmark и отдельные вызовы tools")
class JobMetrics(Hooks):
    def __init__(self):
        self.calls = {}
        self.result = None
        self.error = None

    def tool_event(self, event):
        data = event.model_dump(mode="json")
        completed = data["completed"]
        error = data["error"]
        duration = None if completed is None else (datetime.fromisoformat(completed.replace("Z", "+00:00")) - datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))).total_seconds()
        # Размеры — байты JSON UTF-8 аргументов и результата с ошибкой, без сетевых заголовков.
        self.calls[data["uuid"]] = {
            "call_id": data["uuid"], "name": data["function"], "started_at": data["timestamp"],
            "duration_seconds": duration, "time_limit_seconds": float(os.environ["BENCHMARK_GAIA_TOOL_TIMEOUT"]),
            "status": "running" if completed is None else "timeout" if error and error["type"] == "timeout" else "error" if error else "success",
            "error": error,
            "request_bytes": len(json.dumps(data["arguments"], ensure_ascii=False).encode("utf-8")),
            "response_bytes": None if completed is None else len(json.dumps({"result": data["result"], "error": error}, ensure_ascii=False).encode("utf-8")),
        }

    def save(self):
        logs = Path(os.environ["INSPECT_LOG_DIR"]).parent
        for path, value in [
            (logs / "calls.json", list(self.calls.values())),
            (logs / "benchmark.json", {"result": self.result, "error": self.error}),
        ]:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value), encoding="utf-8")
            temporary.replace(path)

    async def on_sample_event(self, data: SampleEvent) -> None:
        if data.event.event == "tool":
            # Повторное уведомление обновляет тот же вызов, а не создаёт ещё один.
            self.tool_event(data.event)
            self.save()

    async def on_sample_end(self, data: SampleEnd) -> None:
        events = [event.model_dump(mode="json") for event in data.sample.events]
        for event in data.sample.events:
            if event.event == "tool":
                self.tool_event(event)
        scores = [score.value for score in data.sample.scores.values()]
        overflow = any(event["event"] == "model" and re.search(r"request \(\d+ tokens\).*context size", str(event.get("error") or "")) for event in events)
        timed_out = any(event["event"] == "sample_limit" and event["type"] == "time" for event in events)
        self.result = "context_overflow" if overflow else "timeout" if timed_out else "sample_error" if data.sample.error else "solved" if any(
            value == "C" or value is True or isinstance(value, (int, float)) and value > 0 for value in scores
        ) else "unsolved"
        self.error = "benchmark:sample" if data.sample.error else None
        if data.sample.error:
            reason = re.search(r"benchmark:(repeated_actions|no_tool_calls)\b", data.sample.error.message)
            if reason:
                self.error = reason.group()
        self.save()


def timed_tool(tool, seconds: float):
    # Сохраняем описание и параметры инструмента; таймаут становится наблюдаемой моделью ошибкой.
    @functools.wraps(tool)
    async def execute(*args, **kwargs):
        try:
            async with asyncio.timeout(seconds):
                return await tool(*args, **kwargs)
        except TimeoutError as error:
            raise ToolError(f"Tool timed out after {seconds:g} seconds; try another approach.") from error
    return execute


async def guard_repetition(state):
    limit = int(os.environ['BENCHMARK_GAIA_MAX_CONSECUTIVE_REPEATS'])
    if limit < 1:
        raise ValueError("BENCHMARK_GAIA_MAX_CONSECUTIVE_REPEATS must be positive")
    no_tools = repeats = 0
    previous = None
    # Считаем по истории конкретного sample: параллельные job не делят счётчик.
    for message in reversed(state.messages):
        if message.role != "assistant":
            continue
        calls = message.tool_calls
        signature = json.dumps(
            [(call.function, call.arguments) for call in calls] if calls else message.text.strip(),
            sort_keys=True, ensure_ascii=False,
        )
        if previous is None:
            previous = signature
        if signature == previous:
            repeats += 1
        else:
            break
        if repeats >= limit:
            raise RuntimeError(f"benchmark:repeated_actions: Agent stalled: {repeats} consecutive identical actions")
    for message in reversed(state.messages):
        if message.role != "assistant":
            continue
        if message.tool_calls:
            break
        no_tools += 1
        if no_tools >= limit:
            raise RuntimeError(f"benchmark:no_tool_calls: Agent stalled: {no_tools} consecutive responses without tool calls")
    return True


def selected_tools(seconds: float):
    factories = {"bash": lambda: [bash(seconds)], "python": lambda: [python(seconds)], "browser": web_browser}
    names = os.environ["BENCHMARK_TOOLS"] or os.environ["BENCHMARK_GAIA_TOOLS"]
    return [timed_tool(tool, seconds) for name in names.split(",") for tool in factories[name.strip()]()]


def _gaia(
    subset: Literal["2023_all", "2023_level1", "2023_level2", "2023_level3"],
    split: Literal["test", "validation"] = "validation",
    max_attempts: int = 1,
    code_timeout: int = 180,
) -> Task:
    # Поведение стандартного GAIA ReAct сохраняем, добавляя единый таймаут ко всем tools.
    solver = react(
        prompt=AgentPrompt(
            instructions=dedent("""
                You are an autonomous AI agent, consisting of a large language model and a scaffolding program that runs on a linux VM.

                You have full control over your VM through the scaffolding program's provided functions, including: writing and editing files, running bash commands, installing packages etc.. From your VM, you can access the internet.

                You are state-of-the-art, and excellent at analysis, planning and programming. You possess encyclopedic knowledge of the world. You write in a very direct and concise style.

                You are resourceful and adaptive, and you never give up. Before deciding something can't be done, you try it out. You consider multiple options and choose the best one. If your current approach doesn't work, you formulate a new plan. You are given a task you need to solve completely on your own.

                Please think step by step before calling tools. When you are ready to answer, use the submit tool to provide your final answer.
                """),
            assistant_prompt=None, submit_prompt=None,
        ),
        tools=selected_tools(code_timeout),
        attempts=max_attempts, submit=AgentSubmit(answer_only=True, keep_in_messages=True),
        on_continue=guard_repetition,
    )
    result = upstream_gaia(
        solver=solver,
        max_attempts=max_attempts,
        subset=subset,
        split=split,
        sandbox="local" if os.environ["WORKER_TOOL_SANDBOX"] == "local" else DEFAULT_DOCKER_SANDBOX,
    )
    # Upstream GAIA по умолчанию ограничивает 100 сообщениями, даже без CLI-флага.
    result.message_limit = None
    if os.environ["WORKER_TOOL_SANDBOX"] == "local":
        # В local нет контейнерного /shared_files: каждая job использует свою временную папку.
        shared = Path(os.environ["XDG_RUNTIME_DIR"]) / "shared_files"
        shared.mkdir(parents=True, exist_ok=True)
        for sample in result.dataset:
            sample.setup = None
            sample.input = sample.input.replace("/shared_files/", f"{shared}/")
            if sample.files:
                sample.files = {name.replace("/shared_files/", f"{shared}/"): source for name, source in sample.files.items()}
    return result


@task
def gaia(**kwargs: object) -> Task:
    return _gaia("2023_all", **kwargs)


@task
def gaia_level1(**kwargs: object) -> Task:
    return _gaia("2023_level1", **kwargs)


@task
def gaia_level2(**kwargs: object) -> Task:
    return _gaia("2023_level2", **kwargs)


@task
def gaia_level3(**kwargs: object) -> Task:
    return _gaia("2023_level3", **kwargs)


def prepare_sandbox() -> None:
    """Лениво скачивает точный GAIA sandbox image; параллельные job ждут один lock."""
    spec = importlib.util.find_spec("inspect_evals.gaia")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inspect_evals.gaia package not found")
    compose = Path(next(iter(spec.submodule_search_locations))) / "compose.yaml"
    images = subprocess.run(
        ["docker", "compose", "-f", str(compose), "config", "--images"],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout.splitlines()
    if not images:
        raise RuntimeError(f"no images found in {compose}")

    marker = ROOT / "_state/runtime/inspect_ai/docker/inspect-gaia.ready"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with (marker.parent / "image.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = marker.read_text(encoding="utf-8").splitlines() if marker.exists() else []
        present = all(_image_exists(image) for image in images)
        if ready == images and present:
            return
        for image in images:
            if not _image_exists(image):
                subprocess.run(["docker", "pull", image], check=True, timeout=1800)
        marker.write_text("\n".join(images) + "\n", encoding="utf-8")


def _image_exists(image: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
    ).returncode == 0


def main() -> None:
    if os.environ["WORKER_TOOL_SANDBOX"] == "docker":
        prepare_sandbox()
    else:
        # Playwright сам проверяет кэш; после обновления пакета скачает соответствующий Chromium.
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    env = os.environ | {
        "GAIA_MODEL_BASE_URL": os.environ["BENCHMARK_MODEL_BASE_URL"],
        "GAIA_MODEL_API_KEY": "local",
        "GAIA_MODEL_NAME": os.environ["WORKER_MODEL"],
    }
    command = [
        str(ROOT / ".venv/bin/inspect"), "eval", f"{Path(__file__).resolve()}@gaia",
        "--model", f"openai-api/gaia_model/{os.environ['WORKER_MODEL']}",
        "--sandbox", os.environ["WORKER_TOOL_SANDBOX"],
        "--max-connections", "1",
        "--limit", f"{os.environ['BENCHMARK_INDEX']}-{os.environ['BENCHMARK_INDEX']}",
        "--time-limit", os.environ["BENCHMARK_TIME_LIMIT"],
        "-M", f"client_timeout={os.environ['BENCHMARK_TIME_LIMIT']}",
        "-T", f"max_attempts={os.environ['BENCHMARK_GAIA_MAX_ATTEMPTS']}",
        "-T", f"code_timeout={os.environ['BENCHMARK_GAIA_TOOL_TIMEOUT']}",
        "--no-fail-on-error",
    ]
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
