from __future__ import annotations

# BAVT – Budget-Aware Verification Tree (arxiv 2603.12634)
# Implements beam search of width B at each tool-call step. The effective
# beam width is reduced to 1 (greedy) when the tool budget runs low,
# reproducing the paper's low-budget collapse behaviour.
# Candidates are scored by a heuristic: shorter tool arguments signal a
# more targeted, confident action (higher score). An LLM-based value
# function would require injecting the model client into Breakpoints,
# which would require modifying core.py, so the heuristic is used instead.

import os
from typing import Any

import svc_scaffold.openai_helpers as h

BEAM_WIDTH = int(os.getenv("BAVT_BEAM_WIDTH", "2"))
TOOL_BUDGET = int(os.getenv("BAVT_TOOL_BUDGET", "15"))


def _heuristic_score(response: dict) -> float:
    calls = h.tool_calls(response)
    if not calls:
        return 0.0
    # Shorter argument strings → more targeted call → higher score
    total_arg_len = sum(len(c.get("function", {}).get("arguments", "")) for c in calls)
    return -float(total_arg_len)


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_BAVT"

    def before_task_call(self, payload):
        self.store.setdefault("tool_calls_done", 0)
        return payload

    def before_chat_message(self, payload):
        self.store["last_payload"] = payload
        return payload

    def after_tool_call(self, payload):
        return payload

    def after_chat_message(self, response):
        if response is None or not h.has_tool_call(response):
            self.store.pop("candidates", None)
            return response, None

        # Collapse to greedy when budget is tight
        tools_remaining = TOOL_BUDGET - self.store.get("tool_calls_done", 0)
        effective_b = BEAM_WIDTH if tools_remaining > BEAM_WIDTH + 2 else 1

        candidates: list = self.store.setdefault("candidates", [])
        candidates.append(response)
        print(f"[BAVT] candidate {len(candidates)}/{effective_b}, tools_done={self.store.get('tool_calls_done', 0)}, remaining={tools_remaining}", flush=True)

        if len(candidates) < effective_b:
            self.store["retry_payload"] = self.store["last_payload"]
            return None, None

        scores = [_heuristic_score(r) for r in candidates]
        best = max(candidates, key=_heuristic_score)
        print(f"[BAVT] selected best candidate, scores={scores}", flush=True)
        self.store["candidates"] = []
        return best, None

    def after_task_call(self, response):
        retry = self.store.pop("retry_payload", None)
        if retry is not None:
            return None, retry
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        self.store["tool_calls_done"] = self.store.get("tool_calls_done", 0) + 1
        return response