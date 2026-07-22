from __future__ import annotations

import os
from typing import Any

import svc_scaffold.openai_helpers as h

TOOL_BUDGET = int(os.getenv("VOI_TOOL_BUDGET", "30"))
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
        return "FEATURE_VOI"

    def before_task_call(self, payload):
        self.store["tool_calls"] = 0
        self.store["prompt_tokens"] = 0
        self.store["completion_tokens"] = 0
        self.store["last_tool_result_tokens"] = 0
        return payload

    def before_chat_message(self, payload):
        msgs = list(h.messages(payload))
        last_tool_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], dict) and msgs[i].get("role") in {"tool", "function"}:
                last_tool_idx = i
                break
            if isinstance(msgs[i], dict) and msgs[i].get("role") != "system":
                break
        if last_tool_idx is None:
            return payload

        tool_calls = self.store.get("tool_calls", 0)
        tools_remaining = max(TOOL_BUDGET - tool_calls, 0)
        context = (
            self.store.get("prompt_tokens", 0)
            + self.store.get("completion_tokens", 0)
            + self.store.get("last_tool_result_tokens", 0)
        )
        context_remaining = max(CTX_LIMIT - context, 0)
        context_fraction = context / CTX_LIMIT if CTX_LIMIT > 0 else 1.0

        if tools_remaining > 5 and context_fraction < 0.75:
            return payload

        if tools_remaining <= 0 or context_fraction >= 0.97:
            strategy = (
                "ANSWER now. The tool/context budget is effectively exhausted; "
                "do not call more tools; provide the best final answer from the evidence available."
            )
        elif tools_remaining <= 2 or context_fraction >= 0.9:
            strategy = (
                "FINALIZE unless one specific low-cost tool call is clearly likely "
                "to change the final answer."
            )
        else:
            strategy = (
                "Use another tool only if it resolves a concrete uncertainty that is "
                "likely to change the answer; keep the query narrow."
            )

        hint = (
            "[VOI guidance after latest tool result: "
            f"tools remaining {tools_remaining}; "
            f"context remaining {context_remaining} tokens; "
            f"{strategy}]"
        )
        print(f"[VOI] {hint}", flush=True)

        payload = dict(payload)
        tool_msg = dict(msgs[last_tool_idx])
        tool_msg["content"] = f"{_content_text(tool_msg.get('content'))}\n{hint}".strip()
        msgs[last_tool_idx] = tool_msg
        payload["messages"] = msgs
        return payload

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

    def after_chat_message(self, response):
        if response is not None:
            usage = response.get("usage") or {}
            pt = usage.get("prompt_tokens")
            if pt is None:
                pt = usage.get("input_tokens")
            if isinstance(pt, int):
                self.store["prompt_tokens"] = pt
            ct = usage.get("completion_tokens")
            if ct is None:
                ct = usage.get("output_tokens")
            if isinstance(ct, int):
                self.store["completion_tokens"] = ct
        return response, None

    def after_task_call(self, response):
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        tool_call_count = len(h.tool_calls(response)) or 1
        self.store["tool_calls"] = self.store.get("tool_calls", 0) + tool_call_count
        self.store["last_tool_result_tokens"] = 0
        return response
