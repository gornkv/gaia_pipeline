#!/usr/bin/env python3
"""GAIA task definitions и CLI-адаптер для общего benchmark worker."""

import argparse
import asyncio
import functools
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import AgentPrompt, AgentSubmit, react
from inspect_ai.tool import bash, python, web_browser, ToolError
from inspect_evals.gaia.gaia import gaia as upstream_gaia

ROOT = Path(__file__).resolve().parents[2]


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
    limit = int(os.getenv("MAX_CONSECUTIVE_REPEATS", "5"))
    if limit < 1:
        raise ValueError("MAX_CONSECUTIVE_REPEATS must be positive")
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
            raise RuntimeError(f"Agent stalled: {repeats} consecutive identical actions")
    for message in reversed(state.messages):
        if message.role != "assistant":
            continue
        if message.tool_calls:
            break
        no_tools += 1
        if no_tools >= limit:
            raise RuntimeError(f"Agent stalled: {no_tools} consecutive responses without tool calls")
    return True


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
        tools=[timed_tool(tool, code_timeout) for tool in [bash(code_timeout), python(code_timeout), *web_browser()]],
        attempts=max_attempts, submit=AgentSubmit(answer_only=True, keep_in_messages=True),
        on_continue=guard_repetition,
    )
    result = upstream_gaia(
        solver=solver,
        max_attempts=max_attempts,
        subset=subset,
        split=split,
    )
    # Upstream GAIA по умолчанию ограничивает 100 сообщениями, даже без CLI-флага.
    result.message_limit = None
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

    cache = Path(os.environ["XDG_CACHE_HOME"])
    marker = cache / "docker" / "inspect-gaia.ready"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / "_state/inspect-gaia-image.lock").open("a+") as lock:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-no", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--time-limit", required=True)
    args = parser.parse_args()

    if args.sandbox == "docker":
        prepare_sandbox()
    env = os.environ | {
        "GAIA_MODEL_BASE_URL": os.environ["BENCHMARK_MODEL_BASE_URL"],
        "GAIA_MODEL_API_KEY": os.environ["BENCHMARK_MODEL_API_KEY"],
        "GAIA_MODEL_NAME": args.model,
    }
    command = [
        str(ROOT / ".venv/bin/inspect"), "eval", f"{Path(__file__).resolve()}@{args.task}",
        "--model", f"openai-api/gaia_model/{args.model}",
        "--sandbox", args.sandbox,
        "--max-connections", "1",
        "--limit", f"{args.sample_no}-{args.sample_no}",
        "--time-limit", args.time_limit,
        "-T", f"split={args.split}",
        "-T", f"max_attempts={os.environ.get('GAIA_MAX_ATTEMPTS', '1')}",
        "-T", f"code_timeout={os.environ.get('TOOL_TIMEOUT', '120')}",
        "--no-fail-on-error",
    ]
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
