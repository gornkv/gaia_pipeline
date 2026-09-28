from __future__ import annotations

# Context Compaction (arxiv 2308.15022 / 2605.23296) — at a fraction of the
# context window, LLM-summarise the middle history and keep system + recent tail.
# The chunk-summariser runs on the strong HELPER model (helper.py) when
# configured, else the eval model.
# Set CONTEXT_COMPACTION_MAX_TOKENS to the real model window (e.g. 131072).

import asyncio
import os
from typing import Any

import svc_scaffold.openai_helpers as h
from svc_scaffold.helper import helper_completion

COMPACTION_RATIO = float(os.getenv("CONTEXT_COMPACTION_RATIO", "0.7"))
MAX_CONTEXT = int(os.getenv("CONTEXT_COMPACTION_MAX_TOKENS", "28000"))


def _estimate_tokens(messages: list) -> int:
    return sum(len(str(m)) for m in messages) // 4


class Breakpoints:
    def __init__(self, model_client=None):
        self.store: dict[str, Any] = {}
        self.store["compaction_count"] = 0
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "FEATURE_CONTEXT_COMPACTION"

    async def _summarize_chunk(self, messages: list[Any], index: int) -> str:
        prompt = (
            f"Summarize the following conversation chunk #{index} into a concise context note. "
            "Include only the key facts, decisions, and relevant state needed to continue.\n\n"
            "Chunk:\n"
        )
        for item in messages:
            if isinstance(item, dict):
                prompt += f"{item.get('role', '').upper()}: {item.get('content', '')}\n"
            else:
                prompt += f"UNKNOWN: {item}\n"
        content = await helper_completion(prompt, 1024, self.model_client)
        return content or f"[ContextCompaction #{index}] conversation chunk summarized."

    def before_task_call(self, payload):
        self.store["compaction_count"] = 0
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_chat_message(self, response):
        return response, None

    async def after_tool_call(self, payload):
        msgs = list(h.messages(payload))
        tokens = _estimate_tokens(msgs)
        threshold = int(MAX_CONTEXT * COMPACTION_RATIO)
        if tokens <= threshold:
            return payload

        self.store["compaction_count"] += 1
        count = self.store["compaction_count"]
        print(
            f"[CONTEXT_COMPACTION] {tokens} tokens exceeds {threshold} threshold (compaction #{count})",
            flush=True,
        )

        system_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"]
        # summaries written by an earlier compaction can fall inside the last
        # six messages; keeping them in the tail would put a system message
        # after a non-system one (they are preserved in system_msgs anyway)
        tail_msgs = [
            m for m in msgs[-6:]
            if not (isinstance(m, dict) and m.get("role") == "system")
        ]
        middle_msgs = msgs[len(system_msgs) : -6]

        if not middle_msgs:
            return payload

        tail_has_user = any(
            isinstance(m, dict) and m.get("role") == "user" for m in tail_msgs
        )
        if not tail_has_user:
            last_user_idx = next(
                (i for i in range(len(msgs) - 1, -1, -1)
                 if isinstance(msgs[i], dict) and msgs[i].get("role") == "user"),
                None,
            )
            if last_user_idx is not None and msgs[last_user_idx] not in tail_msgs:
                tail_msgs = [msgs[last_user_idx]] + tail_msgs
                middle_msgs = msgs[len(system_msgs) : last_user_idx]

        if not middle_msgs:
            return payload

        chunk_size = max(1, len(middle_msgs) // int(os.getenv("CONTEXT_COMPACTION_CHUNKS", "2")))
        chunks = [middle_msgs[i : i + chunk_size] for i in range(0, len(middle_msgs), chunk_size)]
        summary_texts = await asyncio.gather(
            *[self._summarize_chunk(chunk, idx) for idx, chunk in enumerate(chunks, start=1)]
        )
        summaries = [
            {"role": "system", "content": f"[Context summary #{count}.{idx}] {text}"}
            for idx, text in enumerate(summary_texts, start=1)
        ]

        compacted = list(system_msgs) + summaries + tail_msgs
        payload = dict(payload)
        payload["messages"] = compacted
        print(
            f"[CONTEXT_COMPACTION] replaced middle history with {len(summaries)} summaries; total messages now {len(compacted)}",
            flush=True,
        )
        return payload

    def after_task_call(self, response):
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        return response
