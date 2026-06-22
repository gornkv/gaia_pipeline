llm_judge.pyfrom __future__ import annotations

# LLM-as-Judge / ATLAS-style (arxiv 2606.01667)
# After collecting N answers, uses the longest answer as a heuristic proxy
# for the best answer. In production, this would call a judge model.

import os
from typing import Any

import svc_scaffold.openai_helpers as h

BRANCHES = int(os.getenv("LLM_JUDGE_BRANCHES", "3"))


def _response_length(response: dict) -> int:
    return len(h.message(response).get("content", "") or "")


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_LLM_JUDGE"

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
        print(f"[LLM_JUDGE] branch {branch_num}/{BRANCHES}", flush=True)
        if len(self.store["responses"]) < BRANCHES:
            return None, self.store["branch_payload"]
        responses = [r for r in self.store["responses"] if r is not None]
        if not responses:
            best = response
        elif len(responses) == 1:
            best = responses[0]
        else:
            best = max(responses, key=_response_length)
        print(f"[LLM_JUDGE] selected best of {len(responses)} candidates", flush=True)
        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response
