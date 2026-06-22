from __future__ import annotations

# Complexity-Based Consistency (arxiv 2210.00720)
# Extends Self-Consistency: votes only among the most complex (longest)
# reasoning chains, keeping a configurable fraction of the longest.

import os
from collections import Counter
from typing import Any

import svc_scaffold.openai_helpers as h

BRANCHES = int(os.getenv("COMPLEXITY_CONSISTENCY_BRANCHES", "5"))
RATIO = float(os.getenv("COMPLEXITY_CONSISTENCY_RATIO", "0.5"))


def _complexity(response: dict) -> int:
    return len(h.message(response).get("content", "") or "")


def _extract_final_answer(response: dict) -> str:
    content = h.message(response).get("content", "") or ""
    if isinstance(content, str):
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        return lines[-1] if lines else content.strip()
    return str(content).strip()


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_COMPLEXITY_CONSISTENCY"

    def before_task_call(self, payload):
        if self.store.get("responses") is None:
            self.store["responses"] = []
            self.store["branch_payload"] = payload
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_chat_message(self, response):
        return response, None

    def after_tool_call(self, payload):
        return payload

    def after_task_call(self, response):
        if response is not None:
            self.store["responses"].append(response)
        branch_num = len(self.store["responses"])
        print(f"[COMPLEXITY_CONSISTENCY] branch {branch_num}/{BRANCHES}", flush=True)
        if len(self.store["responses"]) < BRANCHES:
            return None, self.store["branch_payload"]
        responses = [r for r in self.store["responses"] if r is not None]
        responses.sort(key=_complexity, reverse=True)
        keep_count = max(1, int(len(responses) * RATIO))
        complex_responses = responses[:keep_count]
        answers = [_extract_final_answer(r) for r in complex_responses]
        counts = Counter(answers)
        winner = counts.most_common(1)[0][0]
        print(
            f"[COMPLEXITY_CONSISTENCY] voting among {keep_count}/{len(responses)} most complex, winner={winner[:80]}",
            flush=True,
        )
        best = next(
            (r for r in responses if _extract_final_answer(r) == winner),
            responses[0],
        )
        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response
