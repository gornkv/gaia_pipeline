from __future__ import annotations

# Adaptive Best-of-N / AdaBoN-style (arxiv 2505.12050)
# Phase 1 – probe: run one rollout and estimate task complexity from
#            response length (token count proxy).
# Phase 2 – branches: run K additional rollouts, where K scales with
#            complexity. Pick the longest (most complete) final response.

import os
from typing import Any

import svc_scaffold.openai_helpers as h

MAX_K = int(os.getenv("ADAPTIVE_BON_MAX_K", "4"))
COMPLEXITY_THRESHOLD = int(os.getenv("ADAPTIVE_BON_COMPLEXITY_THRESHOLD", "400"))


def _complexity(response: dict) -> int:
    return len(h.message(response).get("content") or "")


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_ADAPTIVE_BON"

    def before_task_call(self, payload):
        if self.store.get("phase") is None:
            self.store["phase"] = "probe"
            self.store["branch_payload"] = payload
            self.store["responses"] = []
            self.store["k"] = 1
            print(f"[ADAPTIVE_BON] NEW TASK, MAX_K={MAX_K}, threshold={COMPLEXITY_THRESHOLD}", flush=True)
        else:
            print(f"[ADAPTIVE_BON] CONTINUING phase={self.store['phase']} responses={len(self.store['responses'])}/{self.store['k']}", flush=True)
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

        if self.store["phase"] == "probe":
            if response is not None:
                complexity = _complexity(response)
                self.store["k"] = MAX_K if complexity > COMPLEXITY_THRESHOLD else max(1, MAX_K // 2)
                print(f"[ADAPTIVE_BON] probe done, complexity={complexity}, k={self.store['k']}", flush=True)
            self.store["phase"] = "branches"

            if len(self.store["responses"]) < self.store["k"]:
                print(f"[ADAPTIVE_BON] branching → need {self.store['k']} total", flush=True)
                return None, self.store["branch_payload"]
        else:
            if len(self.store["responses"]) < self.store["k"]:
                print(f"[ADAPTIVE_BON] branching → {len(self.store['responses'])}/{self.store['k']}", flush=True)
                return None, self.store["branch_payload"]

        candidates = [r for r in self.store["responses"] if r is not None]
        best = max(candidates, key=_complexity) if candidates else response
        print(f"[ADAPTIVE_BON] DONE: picked best of {len(candidates)}, length={_complexity(best)}", flush=True)
        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response