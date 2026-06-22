from __future__ import annotations

# ReSum (arxiv 2509.13313)
# When context approaches limit, triggers a structured summarization
# producing evidence/gaps/next_steps before continuing. Training-free.

import os
from typing import Any

import svc_scaffold.openai_helpers as h

CONTEXT_LIMIT = int(os.getenv("RESUM_CONTEXT_LIMIT", "30000"))


def _estimate_tokens(messages: list) -> int:
    return sum(len(str(m)) for m in messages) // 4


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_RESUM"

    def before_task_call(self, payload):
        self.store["summarized"] = False
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_chat_message(self, response):
        return response, None

    def after_tool_call(self, payload):
        msgs = h.messages(payload)
        tokens = _estimate_tokens(msgs)
        if tokens > CONTEXT_LIMIT and not self.store.get("summarized"):
            self.store["summarized"] = True
            summary_prompt = (
                "The conversation history is becoming very long. "
                "Please provide a structured summary of what has been established so far, including:\n"
                "EVIDENCE: key facts and findings\n"
                "GAPS: what remains unknown or uncertain\n"
                "NEXT_STEPS: what should be done next\n"
                "Then continue with the task."
            )
            msgs = list(msgs)
            msgs.append({"role": "user", "content": summary_prompt})
            payload = dict(payload)
            payload["messages"] = msgs
            print(f"[RESUM] context {tokens} tokens exceeds limit, injected summary request", flush=True)
        return payload

    def after_task_call(self, response):
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        return response
