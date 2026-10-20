"""Переносимый CPU-backend llama.cpp для локальных end-to-end тестов."""

import os


def sandbox() -> str:
    return "docker"


def command(model: str, port: int, parallel: int) -> list[str]:
    # Кэш на хосте переживает --rm, поэтому GGUF и mmproj не скачиваются повторно.
    cache = os.path.abspath(os.environ.get("LLAMA_CACHE", "_state/cache/llama"))
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{cache}:/root/.cache",
        "-p",
        f"127.0.0.1:{port}:8080",
        os.environ.get("LLAMA_DOCKER_IMAGE", "ghcr.io/ggml-org/llama.cpp:server"),
        "-hf",
        model,
        "--alias",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        # llama.cpp делит общий ctx-size между parallel slots.
        "--ctx-size",
        str(int(os.environ["MODEL_CONTEXT_LIMIT"]) * parallel),
        "--parallel",
        str(parallel),
        "--jinja",
    ]


def environment(gpu: str) -> dict[str, str]:
    # Устройства выбирает Docker Desktop; этот локальный backend намеренно не использует GPU.
    return {}
