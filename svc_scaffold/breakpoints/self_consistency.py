from __future__ import annotations

import os
from collections import Counter
from typing import Any

import svc_scaffold.openai_helpers as h

BRANCHES = int(os.getenv("SELF_CONSISTENCY_BRANCHES", "3"))


def _extract_answer(response: dict) -> str:
    content = (h.message(response).get("content") or "").strip()
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    return lines[-1] if lines else content


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_SELF_CONSISTENCY"

    def before_task_call(self, payload):
        if self.store.get("responses") is None:
            self.store["responses"] = []
            self.store["branch_payload"] = payload
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_tool_call(self, payload):
        return payload

    def after_chat_message(self, response):
        return response, None

    def after_task_call(self, response):
        if response is not None:
            self.store["responses"].append(response)

        branch_num = len(self.store["responses"])
        print(f"[SELF_CONSISTENCY] branch {branch_num}/{BRANCHES} collected", flush=True)

        if len(self.store["responses"]) < BRANCHES:
            return None, self.store["branch_payload"]

        answers = [_extract_answer(r) for r in self.store["responses"]]
        counts = Counter(answers)
        winner = counts.most_common(1)[0][0]
        print(f"[SELF_CONSISTENCY] answers: {answers}", flush=True)
        print(f"[SELF_CONSISTENCY] winner: {winner}", flush=True)
        best = next(
            (r for r in self.store["responses"] if _extract_answer(r) == winner),
            self.store["responses"][0],
        )
        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response