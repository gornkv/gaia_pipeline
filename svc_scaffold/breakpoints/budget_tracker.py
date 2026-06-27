from __future__ import annotations

import os
from typing import Any

import svc_scaffold.openai_helpers as h

TOOL_LIMIT = int(os.getenv("BUDGET_TRACKER_TOOL_LIMIT", "20"))
CTX_LIMIT = 131072


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_BUDGET_TRACKER"

    def before_task_call(self, payload):
        self.store["tool_calls"] = 0
        self.store["prompt_tokens"] = 0
        return payload

    def before_chat_message(self, payload):
        tool_calls = self.store.get("tool_calls", 0)
        prompt_tokens = self.store.get("prompt_tokens", 0)
        ctx_remaining = CTX_LIMIT - prompt_tokens
        status = f"[Tools: {tool_calls}/{TOOL_LIMIT} | ctx: {prompt_tokens}/{CTX_LIMIT} ({ctx_remaining} remaining)]"
        print(f"[BUDGET_TRACKER] injecting: {status}", flush=True)

        msgs = list(h.messages(payload))
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], dict) and msgs[i].get("role") == "user":
                msgs[i] = h.append_content(msgs[i], status)
                break

        payload = dict(payload)
        payload["messages"] = msgs
        return payload

    def after_chat_message(self, response):
        if response is not None:
            usage = response.get("usage") or {}
            pt = usage.get("prompt_tokens")
            if pt:
                self.store["prompt_tokens"] = pt
        return response, None

    def after_tool_call(self, payload):
        return payload

    def after_task_call(self, response):
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        self.store["tool_calls"] = self.store.get("tool_calls", 0) + 1
        return response