"""Собирает команду постоянного production-backend vLLM."""

import os


def sandbox() -> str:
    return "local"


def command(model: str, port: int, parallel: int) -> list[str]:
    # max-num-seqs фиксирует требуемую ёмкость: worker не уменьшает PARALLEL молча.
    return [
        os.environ.get("VLLM_BIN", "vllm"),
        "serve",
        model,
        "--served-model-name",
        model,
        "--host",
        "127.0.0.1",
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


def environment(gpu: str) -> dict[str, str]:
    # vLLM видит только физический GPU, назначенный этому worker.
    return {"CUDA_VISIBLE_DEVICES": gpu}
