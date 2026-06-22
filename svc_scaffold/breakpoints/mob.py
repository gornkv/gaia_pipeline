from __future__ import annotations

# Majority of the Bests (MoB) – arxiv 2511.18630
# Bootstraps over generated outputs: creates many subsets of size m,
# picks the best by reward in each, then majority-votes among winners.

import os
import random
from collections import Counter
from typing import Any

import svc_scaffold.openai_helpers as h

BRANCHES = int(os.getenv("MOB_BRANCHES", "8"))
BOOTSTRAP = int(os.getenv("MOB_BOOTSTRAP_SAMPLES", "1000"))
SUBSET = int(os.getenv("MOB_SUBSET_SIZE", "3"))


def _reward(response: dict) -> float:
    return float(len(h.message(response).get("content", "") or ""))


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_MOB"

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
        print(f"[MOB] branch {branch_num}/{BRANCHES}", flush=True)
        if len(self.store["responses"]) < BRANCHES:
            return None, self.store["branch_payload"]
        responses = [r for r in self.store["responses"] if r is not None]
        if not responses:
            self.store = {}
            return response, None
        rewards = [_reward(r) for r in responses]
        winners = []
        for _ in range(BOOTSTRAP):
            idx = random.choices(range(len(responses)), k=min(SUBSET, len(responses)))
            best_idx = max(idx, key=lambda i: rewards[i])
            winners.append(str(h.message(responses[best_idx]).get("content", "")).strip())
        winner = Counter(winners).most_common(1)[0][0]
        print(f"[MOB] bootstrapped {BOOTSTRAP} subsets, winner from {len(responses)} responses", flush=True)
        best = next(
            (r for r in responses if str(h.message(r).get("content", "")).strip() == winner),
            responses[0],
        )
        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response
