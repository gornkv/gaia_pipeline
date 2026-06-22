from __future__ import annotations

import os
from collections import Counter
from typing import Any

import svc_scaffold.openai_helpers as h

BRANCHES = int(os.getenv("SHORT_MAK_BRANCHES", "3"))
print(f"[SHORT_MAK] module loaded, BRANCHES={BRANCHES}", flush=True)


def _extract_answer(response: dict) -> str:
    content = (h.message(response).get("content") or "").strip()
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    return lines[-1] if lines else content


def _chain_length(response: dict) -> int:
    return len(h.message(response).get("content") or "")


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_SHORT_MAK"

    def before_task_call(self, payload):
        if self.store.get("responses") is None:
            self.store["responses"] = []
            self.store["branch_payload"] = payload
            print(f"[SHORT_MAK] NEW TASK initialized, BRANCHES={BRANCHES}", flush=True)
        else:
            print(f"[SHORT_MAK] CONTINUING task, have {len(self.store['responses'])} responses", flush=True)
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
        print(f"[SHORT_MAK] after_task_call: have {branch_num}/{BRANCHES} branches", flush=True)

        if len(self.store["responses"]) < BRANCHES:
            print(f"[SHORT_MAK] returning branch_payload to trigger re-run", flush=True)
            return None, self.store["branch_payload"]

        answers = [_extract_answer(r) for r in self.store["responses"]]
        counts = Counter(answers)
        majority_answer, majority_count = counts.most_common(1)[0]

        candidates = (
            [r for r in self.store["responses"] if _extract_answer(r) == majority_answer]
            if majority_count > 1
            else self.store["responses"]
        )
        lengths = [_chain_length(r) for r in candidates]
        best = min(candidates, key=_chain_length)
        print(f"[SHORT_MAK] DONE: majority='{majority_answer}' ({majority_count}/{BRANCHES}), shortest={min(lengths)} chars", flush=True)
        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response