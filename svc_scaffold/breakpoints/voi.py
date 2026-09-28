from __future__ import annotations

# Value-of-Information budget control (arXiv 2605.05701).
# At each step a controller scores each action (RETRIEVE/DECOMPOSE/ANSWER) by
# value-of-information given the current search state + remaining dual budget,
# then steers the agent to the highest-VOI action. The VOI-scoring call runs on
# the strong HELPER model (helper.py) when configured, else the eval model.

import os
import re
from typing import Any

import svc_scaffold.openai_helpers as h
from svc_scaffold.helper import helper_completion

TOOL_BUDGET = int(os.getenv("VOI_TOOL_BUDGET", "10"))
TOKEN_BUDGET = int(os.getenv("VOI_TOKEN_BUDGET", "40000"))
STATE_CHARS = int(os.getenv("VOI_STATE_CHARS", "1500"))

_ACTIONS = ("RETRIEVE", "DECOMPOSE", "ANSWER")
_DIRECTIVE = {
    "RETRIEVE": "Make at most ONE targeted tool call that maximally reduces uncertainty, then reassess.",
    "DECOMPOSE": "Break the task into explicit sub-questions and tackle the single highest-value one next.",
    "ANSWER": "Do NOT make further tool calls. Commit your best final answer now, concisely.",
}


class Breakpoints:
    def __init__(self, model_client=None):
        self.store: dict[str, Any] = {}
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "FEATURE_VOI"

    def before_task_call(self, payload):
        self.store.setdefault("tool_calls", 0)
        self.store.setdefault("tokens_used", 0)
        return payload

    def _latest_state(self, msgs: list) -> str:
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") in ("tool", "function", "user", "assistant"):
                c = m.get("content")
                text = c if isinstance(c, str) else str(c)
                if text.strip():
                    return text[-STATE_CHARS:]
        return ""

    async def _score_actions(self, state: str, tools_left: int, tok_frac_left: float):
        prompt = (
            "You are a Value-of-Information (VOI) controller for a search agent under a strict dual budget.\n"
            f"Remaining budget: {tools_left} tool calls and {int(tok_frac_left * 100)}% of the token budget.\n\n"
            "Current state (most recent context):\n"
            f"{state}\n\n"
            "For each next action, estimate its Value-of-Information = expected marginal task progress "
            "per unit of remaining budget, on a 0-10 scale:\n"
            "- RETRIEVE: gather more evidence via a tool call\n"
            "- DECOMPOSE: split the problem into sub-questions\n"
            "- ANSWER: commit the final answer now\n"
            "When little budget is left, ANSWER's relative VOI rises.\n"
            "Return ONLY this line: RETRIEVE=<n> DECOMPOSE=<n> ANSWER=<n>"
        )
        txt = await helper_completion(prompt, 64, self.model_client)
        scores: dict[str, float] = {}
        for a in _ACTIONS:
            m = re.search(a + r"\s*=\s*([0-9]+(?:\.[0-9]+)?)", txt)
            if m:
                scores[a] = float(m.group(1))
        return scores or None

    async def before_chat_message(self, payload):
        msgs = list(h.messages(payload))
        tools_used = self.store.get("tool_calls", 0)
        tokens_used = self.store.get("tokens_used", 0)
        tools_left = max(0, TOOL_BUDGET - tools_used)
        tok_frac_left = max(0.0, 1.0 - (tokens_used / TOKEN_BUDGET if TOKEN_BUDGET > 0 else 1.0))

        scores = await self._score_actions(self._latest_state(msgs), tools_left, tok_frac_left)
        if not scores:
            return payload
        action = max(scores, key=scores.get)
        hint = f"[VOI controller — next action: {action} (scores {scores})] {_DIRECTIVE[action]}"
        print(f"[VOI] {hint}", flush=True)

        target = None
        for i in range(len(msgs) - 1, -1, -1):
            role = msgs[i].get("role") if isinstance(msgs[i], dict) else None
            if role in ("tool", "function"):
                target = i
                break
            if role == "user" and target is None:
                target = i
        if target is not None:
            msgs[target] = h.append_content(msgs[target], "\n" + hint)
            payload = dict(payload)
            payload["messages"] = msgs
        return payload

    def after_tool_call(self, payload):
        return payload

    def after_chat_message(self, response):
        if response is not None:
            usage = response.get("usage") or {}
            ct = usage.get("completion_tokens")
            if ct is None:
                ct = usage.get("output_tokens")
            if isinstance(ct, int):
                self.store["tokens_used"] = self.store.get("tokens_used", 0) + ct
        return response, None

    def after_task_call(self, response):
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        self.store["tool_calls"] = self.store.get("tool_calls", 0) + 1
        return response
