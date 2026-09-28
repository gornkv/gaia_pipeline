from __future__ import annotations

import os
import re
import sys
from typing import Any

import httpx
from svc_scaffold.usage import record


def _merge_system_messages(messages: Any) -> Any:
    """Collapse every system message into one at the front.

    The backend accepts exactly one system message and only in first position:
    two of them fail just as surely as one in the middle. Breakpoints that
    prepend an instruction block or splice in a summary would otherwise lose
    the whole sample to a 400, and the loss reads as a bad score for the
    technique rather than as a failure. Reported on stderr so the rate stays
    visible instead of silently disappearing.
    """
    if not isinstance(messages, list):
        return messages
    system = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    if len(system) < 2 and not (
        system and messages and messages[0] is not system[0]
    ):
        return messages
    rest = [m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")]
    parts = []
    simple = True
    for m in system:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        else:
            simple = False
            parts.append(c)
    if simple:
        merged = {"role": "system", "content": "\n\n".join(p for p in parts if p)}
    else:
        flat = []
        for p in parts:
            flat.extend(p if isinstance(p, list) else [{"type": "text", "text": str(p)}])
        merged = {"role": "system", "content": flat}
    print(
        f"[SCAFFOLD] merged {len(system)} system message(s) into one; "
        "the request would have been rejected with 400",
        file=sys.stderr,
        flush=True,
    )
    return [merged] + rest

# Тексты, которыми бэкенды сообщают об исчерпании окна модели. Это предел, а не
# сбой среды, и в отчёте он должен стоять отдельной строкой -- поэтому ответ
# помечается маркером, а не остаётся безымянной 400-й.
_CONTEXT_RE = re.compile(
    r"maximum context length|at least \d+ input tokens|context_length_exceeded|"
    r"longer than the maximum model length|decoder prompt \(length \d+\)",
    re.I,
)

LIMIT_CONTEXT = "LIMIT_CONTEXT"
ERROR_HANG = "ERROR_HANG"


class ModelClientHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


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
        request["messages"] = _merge_system_messages(request.get("messages"))
        request["model"] = self.model
        request["stream"] = False

        timeout = float(os.getenv("BASE_MODEL_REQUEST_TIMEOUT", "1800"))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers(),
                    json=request,
                )
        except httpx.TimeoutException as exc:
            raise ModelClientHTTPError(
                504,
                f"{ERROR_HANG}: base model request timed out after {timeout:g} seconds",
            ) from exc
        except httpx.RequestError as exc:
            raise ModelClientHTTPError(502, f"Base model connection failed: {exc}") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            if response.status_code == 400 and _CONTEXT_RE.search(detail or ""):
                # 413 отличает исчерпание окна от прочих 400-х на уровне кода,
                # чтобы разметка не зависела от формулировок бэкенда.
                raise ModelClientHTTPError(413, f"{LIMIT_CONTEXT}: {detail[:4000]}") from exc
            raise ModelClientHTTPError(response.status_code, detail[:4000]) from exc
        data = response.json()
        record(data.get("usage"))
        return data
