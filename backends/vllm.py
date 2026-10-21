"""Собирает команду постоянного production-backend vLLM."""

import os
from typing import Any


def enrich_payload(payload: dict[str, Any]) -> dict[str, Any]:
    budget = int(os.getenv("MODEL_THINKING_BUDGET_LIMIT") or "0")
    if budget > 0:
        payload["thinking_token_budget"] = budget
    return payload


def sandbox() -> str:
    return "local"


def command(model: str, port: int, parallel: int) -> list[str]:
    # max-num-seqs фиксирует требуемую ёмкость: worker не уменьшает PARALLEL молча.
    return [
        "env",
        f"CUDA_VISIBLE_DEVICES={os.environ['GPU']}",
        os.environ.get("VLLM_BIN", "vllm"),
        "serve",
        model,
        "--served-model-name",
        model,
        "--host",
        "127.0.0.1",
        "--api-key",
        "local",
        "--port",
        str(port),
        "--max-num-seqs",
        str(parallel),
        "--max-model-len",
        os.environ["MODEL_CONTEXT_LIMIT"],
        "--gpu-memory-utilization",
        os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        os.environ.get("TOOL_CALL_PARSER", "qwen3_coder"),
        "--trust-remote-code",
    ]
