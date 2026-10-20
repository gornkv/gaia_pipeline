"""Внешний OpenAI-compatible backend, которым worker не управляет."""

import os


def sandbox() -> str:
    return "docker"


def command(model: str, port: int, parallel: int) -> None:
    return None


def base_url(port: int) -> str:
    return os.environ.get("MODEL_API_BASE_URL", os.environ.get("BASE_MODEL_API_BASE_URL", "")).rstrip("/")


def api_key() -> str:
    return os.environ["MODEL_API_KEY"]


def environment(gpu: str) -> dict[str, str]:
    return {}
