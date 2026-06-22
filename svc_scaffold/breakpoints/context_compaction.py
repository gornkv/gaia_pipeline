from __future__ import annotations

# Context Compaction (arxiv 2308.15022 / arxiv 2605.23296)
# When context exceeds 70% of limit, keeps system messages and recent
# messages, inserting a compaction marker. Prevents context rot.

import os
from typing import Any

import svc_scaffold.openai_helpers as h

COMPACTION_RATIO = float(os.getenv("CONTEXT_COMPACTION_RATIO", "0.7"))
MAX_CONTEXT = int(os.getenv("CONTEXT_COMPACTION_MAX_TOKENS", "28000"))


def _estimate_tokens(messages: list) -> int:
    return sum(len(str(m)) for m in messages) // 4


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_CONTEXT_COMPACTION"

    def before_task_call(self, payload):
        self.store["compaction_count"] = 0
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_chat_message(self, response):
        return response, None

    def after_tool_call(self, payload):
        msgs = h.messages(payload)
        tokens = _estimate_tokens(msgs)
        threshold = int(MAX_CONTEXT * COMPACTION_RATIO)
        if tokens > threshold:
            count = self.store.get("compaction_count", 0) + 1
            self.store["compaction_count"] = count
            print(f"[CONTEXT_COMPACTION] {tokens} tokens exceeds {threshold} threshold (compaction #{count})", flush=True)
            system_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
            recent_msgs = msgs[-6:]
            compacted = system_msgs + [
                {
                    "role": "system",
                    "content": f"[Context compacted #{count}: earlier conversation history has been summarized to conserve space.]",
                }
            ] + recent_msgs
            payload = dict(payload)
            payload["messages"] = compacted
        return payload

    def after_task_call(self, response):
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        return response
