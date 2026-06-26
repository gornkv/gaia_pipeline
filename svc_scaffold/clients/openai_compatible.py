from __future__ import annotations

import json
from typing import Any

import httpx

CTX_LIMIT = 131072
MAX_TOKENS_OUT = 8192
# leave headroom for output and special tokens
MAX_INPUT_CHARS = (CTX_LIMIT - MAX_TOKENS_OUT) * 3


def _truncate_messages(messages: list) -> list:
    """Drop old tool results from the middle when total chars exceed budget."""
    total = sum(len(json.dumps(m)) for m in messages)
    if total <= MAX_INPUT_CHARS:
        return messages

    # Always keep: system (index 0) and last N messages
    result = []
    kept_end = []
    budget = MAX_INPUT_CHARS

    for m in reversed(messages):
        s = len(json.dumps(m))
        if budget - s > 0:
            kept_end.insert(0, m)
            budget -= s
        else:
            break

    # Prepend system message if not already included
    if messages and messages[0].get("role") == "system" and messages[0] not in kept_end:
        kept_end.insert(0, messages[0])

    return kept_end


class OpenAICompatibleModelClient:
    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def health(self) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/models", headers=self.headers())
        response.raise_for_status()

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        request["model"] = self.model
        request["stream"] = False
        request.setdefault("max_tokens", MAX_TOKENS_OUT)

        if "messages" in request:
            request["messages"] = _truncate_messages(request["messages"])

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers(),
                json=request,
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            try:
                err = response.json().get("error", {}).get("message", response.text)[:300]
            except Exception:
                err = response.text[:300]
            return {
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Error: {err}"},
                    "finish_reason": "stop",
                }],
                "model": self.model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        return response.json()
