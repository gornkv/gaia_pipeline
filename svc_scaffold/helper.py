from __future__ import annotations

# Shared auxiliary-model client for scaffold breakpoints.
#
# Breakpoints that need a *helper* LLM (compressor / summariser / scorer) call
# helper_completion(). If HELPER_API_BASE_URL + HELPER_MODEL are set, the call
# goes to that strong, neutral helper endpoint (e.g. gpt-oss-120B on its own
# card, or an API model); otherwise it falls back to the eval model_client so
# the feature still runs (current behaviour).
#
# env:
#   HELPER_API_BASE_URL   e.g. http://127.0.0.1:18082/v1   (unset -> fallback)
#   HELPER_MODEL          e.g. gpt-oss-120b
#   HELPER_API_KEY        default EMPTY
#   HELPER_REASONING_EFFORT  low|medium|high  (default low; for gpt-oss so it
#                            doesn't burn tokens thinking on helper tasks)
#   HELPER_REQUEST_TIMEOUT   seconds (default 600)

import os
import httpx

import svc_scaffold.openai_helpers as h
from svc_scaffold.usage import record

_ANNOUNCED = False


def _base() -> str:
    return os.getenv("HELPER_API_BASE_URL", "").rstrip("/")


def _model() -> str:
    return os.getenv("HELPER_MODEL", "")


def helper_enabled() -> bool:
    return bool(_base() and _model())


def _text(resp: dict) -> str:
    msg = h.message(resp)
    out = (msg.get("content") or "").strip()
    if not out:
        out = (msg.get("reasoning_content") or "").strip()
    return out


async def _fallback(prompt: str, max_tokens: int, fallback_client) -> str:
    if fallback_client is None:
        return ""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = await fallback_client.chat_completions(payload)
        return _text(r)
    except Exception as e:  # noqa: BLE001
        print(f"[HELPER] fallback call failed: {e}", flush=True)
        return ""


async def helper_completion(prompt: str, max_tokens: int, fallback_client=None) -> str:
    """Run one auxiliary completion on the strong helper (or fall back)."""
    global _ANNOUNCED
    if not helper_enabled():
        if not _ANNOUNCED:
            print("[HELPER] not configured -> using eval model for aux calls", flush=True)
            _ANNOUNCED = True
        return await _fallback(prompt, max_tokens, fallback_client)

    if not _ANNOUNCED:
        print(f"[HELPER] using {_model()} @ {_base()} for aux calls", flush=True)
        _ANNOUNCED = True

    req = {
        "model": _model(),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        # gpt-oss: keep reasoning cheap on helper tasks; ignored by non-reasoning helpers
        "reasoning_effort": os.getenv("HELPER_REASONING_EFFORT", "low"),
        "chat_template_kwargs": {"reasoning_effort": os.getenv("HELPER_REASONING_EFFORT", "low")},
    }
    key = os.getenv("HELPER_API_KEY", "EMPTY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    timeout = float(os.getenv("HELPER_REQUEST_TIMEOUT", "600"))
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            resp = await c.post(f"{_base()}/chat/completions", headers=headers, json=req)
        resp.raise_for_status()
        data = resp.json()
        record(data.get("usage"), helper=True)
        out = _text(data)
        if out:
            return out
        # empty -> fall back to eval model
        return await _fallback(prompt, max_tokens, fallback_client)
    except Exception as e:  # noqa: BLE001
        print(f"[HELPER] call failed ({e}); falling back to eval model", flush=True)
        return await _fallback(prompt, max_tokens, fallback_client)
