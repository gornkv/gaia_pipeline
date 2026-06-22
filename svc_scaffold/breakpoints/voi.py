from __future__ import annotations

import os
from typing import Any

import svc_scaffold.openai_helpers as h

TOOL_BUDGET = int(os.getenv("VOI_TOOL_BUDGET", "10"))
TOKEN_BUDGET = int(os.getenv("VOI_TOKEN_BUDGET", "40000"))


def _strategy(tools_used: int, tokens_used: int) -> str:
    tools_remaining = TOOL_BUDGET - tools_used
    token_fraction_used = tokens_used / TOKEN_BUDGET if TOKEN_BUDGET > 0 else 1.0

    if tools_remaining <= 1 or token_fraction_used >= 0.9:
        return f"ANSWER — budget nearly exhausted (tools left: {tools_remaining}). Provide your best answer now without additional tool calls."
    if tools_remaining <= 3 or token_fraction_used >= 0.6:
        return f"RETRIEVE — budget is low (tools left: {tools_remaining}). Make at most one targeted retrieval, then answer."
    return f"DECOMPOSE — budget is sufficient (tools left: {tools_remaining}). Break the problem down and retrieve as needed."


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_VOI"

    def before_task_call(self, payload):
        self.store.setdefault("tool_calls", 0)
        self.store.setdefault("tokens_used", 0)
        return payload

    def before_chat_message(self, payload):
        strategy = _strategy(
            self.store.get("tool_calls", 0),
            self.store.get("tokens_used", 0),
        )
        hint = f"[VOI strategy: {strategy}]"
        print(f"[VOI] {hint}", flush=True)

        msgs = list(h.messages(payload))
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], dict) and msgs[i].get("role") == "user":
                msgs[i] = h.append_content(msgs[i], hint)
                break

        payload = dict(payload)
        payload["messages"] = msgs
        return payload

    def after_tool_call(self, payload):
        return payload

    def after_chat_message(self, response):
        if response is not None:
            usage = response.get("usage") or {}
            self.store["tokens_used"] = self.store.get("tokens_used", 0) + (usage.get("total_tokens") or 0)
        return response, None

    def after_task_call(self, response):
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        self.store["tool_calls"] = self.store.get("tool_calls", 0) + 1
        return response