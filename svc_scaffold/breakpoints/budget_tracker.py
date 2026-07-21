from __future__ import annotations

import os
from typing import Any

import svc_scaffold.openai_helpers as h

TOOL_LIMIT = int(os.getenv("BUDGET_TRACKER_TOOL_LIMIT", "30"))
CTX_LIMIT = int(os.getenv("BASE_MODEL_CTX_LIMIT", "262144"))


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(_content_text(item.get("text") or item.get("content") or ""))
            else:
                parts.append(_content_text(item))
        return "\n".join(part for part in parts if part)
    return "" if value is None else str(value)


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_BUDGET_TRACKER"

    def before_task_call(self, payload):
        self.store["tool_calls"] = 0
        self.store["prompt_tokens"] = 0
        self.store["last_tool_result_tokens"] = 0
        return payload

    def before_chat_message(self, payload):
        tool_calls = self.store.get("tool_calls", 0)
        prompt_tokens = self.store.get("prompt_tokens", 0)
        last_tool_result_tokens = self.store.get("last_tool_result_tokens", 0)
        status = (
            "[Budget status: "
            f"tool calls used: {tool_calls}; "
            f"tool calls remaining: {max(TOOL_LIMIT - tool_calls, 0)}; "
            f"context used: {prompt_tokens} tokens; "
            f"context remaining: {max(CTX_LIMIT - prompt_tokens, 0)} tokens"
        )
        if last_tool_result_tokens:
            status += f"; last tool result added about {last_tool_result_tokens} tokens"
        status += ".]"
        print(f"[BUDGET_TRACKER] injecting transient status: {status}", flush=True)

        payload = dict(payload)
        payload["messages"] = list(h.messages(payload)) + [
            {"role": "system", "content": status}
        ]
        return payload

    def after_chat_message(self, response):
        if response is not None:
            usage = response.get("usage") or {}
            pt = usage.get("prompt_tokens")
            if pt is None:
                pt = usage.get("input_tokens")
            if pt is not None:
                self.store["prompt_tokens"] = pt
        return response, None

    def after_tool_call(self, payload):
        total = 0
        for msg in reversed(h.messages(payload)):
            if not isinstance(msg, dict) or msg.get("role") not in {"tool", "function"}:
                break
            text = _content_text(msg.get("content"))
            if text:
                total += max(1, round(len(text) / 4))
        self.store["last_tool_result_tokens"] = total
        return payload

    def after_task_call(self, response):
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        tool_call_count = len(h.tool_calls(response)) or 1
        tool_calls = self.store.get("tool_calls", 0) + tool_call_count
        self.store["tool_calls"] = tool_calls
        self.store["last_tool_result_tokens"] = 0
        return response
