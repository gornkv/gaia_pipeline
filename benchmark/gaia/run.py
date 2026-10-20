#!/usr/bin/env python3
"""GAIA task definitions и CLI-адаптер для общего benchmark worker."""

import argparse
import fcntl
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_evals.gaia.gaia import default_solver, gaia as upstream_gaia

ROOT = Path(__file__).resolve().parents[2]


def _gaia(
    subset: Literal["2023_all", "2023_level1", "2023_level2", "2023_level3"],
    split: Literal["test", "validation"] = "validation",
    max_attempts: int = 1,
    code_timeout: int = 180,
) -> Task:
    return upstream_gaia(
        solver=default_solver(max_attempts, code_timeout),
        max_attempts=max_attempts,
        subset=subset,
        split=split,
    )


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
    parser.add_argument("--message-limit", required=True)
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
        "--message-limit", args.message_limit,
        "-T", f"split={args.split}",
        "-T", f"max_attempts={os.environ.get('GAIA_MAX_ATTEMPTS', '1')}",
        "-T", f"code_timeout={os.environ.get('TOOL_TIMEOUT', '180')}",
        "--no-fail-on-error",
    ]
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
