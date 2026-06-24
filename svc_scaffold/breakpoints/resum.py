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


def _find_first_user_message(messages: list[Any]) -> dict[str, Any] | None:
    for item in messages:
        if isinstance(item, dict) and item.get("role") == "user":
            return item
    return None


class Breakpoints:
    def __init__(self, model_client=None):
        self.store: dict[str, Any] = {}
        self.store["resum_count"] = 0
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "FEATURE_RESUM"

    async def _summarize(self, messages: list[Any], count: int) -> str:
        prompt = (
            "Compress the following conversation history into a structured summary with three sections:\n"
            "EVIDENCE: what is known and established\n"
            "GAPS: what remains unknown or uncertain\n"
            "NEXT_STEPS: what should be done next to continue working on the task\n\n"
            "Only include the state necessary to continue. Do not repeat the full conversation. "
            "Use concise bullet points where appropriate.\n\n"
            "Conversation history:\n"
        )
        for item in messages:
            if isinstance(item, dict):
                prompt += f"{item.get('role', '').upper()}: {item.get('content', '')}\n"
            else:
                prompt += f"UNKNOWN: {item}\n"

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        if not self.model_client:
            return (
                f"[ReSum #{count}] Conversation history has been summarized. "
                "Original details have been compacted to conserve context."
            )
        result = await self.model_client.chat_completions(payload)
        content = (h.message(result).get("content") or "").strip()
        if not content:
            return (
                f"[ReSum #{count}] Conversation history has been summarized. "
                "Original details have been compacted to conserve context."
            )
        return f"[ReSum #{count}] {content}"

    def before_task_call(self, payload):
        self.store["resum_count"] = 0
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_chat_message(self, response):
        return response, None

    async def after_tool_call(self, payload):
        msgs = list(h.messages(payload))
        tokens = _estimate_tokens(msgs)
        if tokens <= CONTEXT_LIMIT:
            return payload

        self.store["resum_count"] += 1
        count = self.store["resum_count"]
        print(f"[RESUM] {tokens} tokens exceeds {CONTEXT_LIMIT} limit, compressing history #{count}", flush=True)

        first_user = _find_first_user_message(msgs)
        system_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
        history_msgs = [m for m in msgs if m not in system_msgs and m is not first_user]

        summary_text = await self._summarize(history_msgs, count)
        compacted = list(system_msgs)
        if first_user:
            compacted.append(first_user)
        compacted.append({"role": "system", "content": summary_text})
        payload = dict(payload)
        payload["messages"] = compacted
        print(f"[RESUM] history replaced with compressed state ({len(compacted)} messages)", flush=True)
        return payload

    def after_task_call(self, response):
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        return response
