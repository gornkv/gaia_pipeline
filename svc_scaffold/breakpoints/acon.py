from __future__ import annotations

# ACON without fine-tuning (arxiv 2510.00615)
# Injects pre-optimised compression guidelines into the system prompt.
# Guidelines are stored at module level so they can be iteratively refined
# across tasks within the same process (API-only, no fine-tuning).

import os
from typing import Any

import svc_scaffold.openai_helpers as h

TOKEN_BUDGET = int(os.getenv("ACON_TOKEN_BUDGET", "30000"))

# Globally shared guidelines – updated after over-budget tasks
_guidelines: list[str] = [
    "Be maximally concise; omit filler words and verbose preambles.",
    "Do not restate the question; go straight to reasoning.",
    "Use abbreviations and symbols where unambiguous.",
    "Chain reasoning steps inline; avoid bullet lists unless essential.",
    "Provide the final answer in one short sentence.",
]


def _guidelines_block() -> str:
    return "Compression guidelines:\n" + "\n".join(f"- {g}" for g in _guidelines)


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_ACON"

    def before_task_call(self, payload):
        self.store["tokens_used"] = 0
        # Inject guidelines into existing system message or prepend a new one
        msgs = list(h.messages(payload))
        block = _guidelines_block()
        injected = False
        for i, msg in enumerate(msgs):
            if isinstance(msg, dict) and msg.get("role") == "system":
                msgs[i] = h.append_content(msg, "\n\n" + block)
                injected = True
                break
        if not injected:
            msgs.insert(0, {"role": "system", "content": block})
        payload = dict(payload)
        payload["messages"] = msgs
        print(f"[ACON] injected guidelines ({'system msg updated' if injected else 'new system msg'}), budget={TOKEN_BUDGET}", flush=True)
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_tool_call(self, payload):
        return payload

    def after_chat_message(self, response):
        if response is not None:
            usage = response.get("usage") or {}
            self.store["tokens_used"] = self.store.get("tokens_used", 0) + (usage.get("total_tokens") or 0)
        return response, None

    def after_task_call(self, response):
        tokens = self.store.get("tokens_used", 0)
        if tokens > TOKEN_BUDGET and _guidelines:
            _guidelines[-1] = _guidelines[-1].rstrip(".") + " — be even briefer."
            print(f"[ACON] over budget ({tokens}/{TOKEN_BUDGET}), tightened guidelines", flush=True)
        else:
            print(f"[ACON] task done, tokens_used={tokens}/{TOKEN_BUDGET}", flush=True)
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        return response