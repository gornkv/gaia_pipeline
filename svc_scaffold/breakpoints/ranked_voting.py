from __future__ import annotations

# Ranked Voting Self-Consistency (arxiv 2505.10772)
# Prompts model to output top-3 ranked answers, then aggregates
# via IRV, Borda Count (BCV), or Mean Reciprocal Rank Voting (MRRV).
# Modes: RANKED_VOTING_MODE=irv|bcv|mrrv (default bcv)

import os
from collections import Counter
from typing import Any

import svc_scaffold.openai_helpers as h

BRANCHES = int(os.getenv("RANKED_VOTING_BRANCHES", "5"))
MODE = os.getenv("RANKED_VOTING_MODE", "bcv")
NUM_CANDIDATES = int(os.getenv("RANKED_VOTING_CANDIDATES", "3"))


def _extract_ranked(response: dict, n: int) -> list[str]:
    content = h.message(response).get("content", "")
    if not isinstance(content, str):
        return []
    candidates = []
    for line in content.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and ". " in line[:4]:
            candidates.append(line.split(". ", 1)[1].strip())
            if len(candidates) >= n:
                break
    return candidates[:n]


def _irv(rankings: list[list[str]]) -> str:
    active = list(set(c for r in rankings for c in r))
    while len(active) > 1:
        first = Counter({c: 0 for c in active})
        for r in rankings:
            for c in r:
                if c in active:
                    first[c] += 1
                    break
        total = sum(first.values())
        for c, cnt in first.items():
            if cnt > total / 2:
                return c
        active.remove(min(first, key=first.get))
    return active[0] if active else ""


def _bcv(rankings: list[list[str]]) -> str:
    m = max(len(r) for r in rankings)
    scores = {}
    for r in rankings:
        for i, c in enumerate(r):
            scores[c] = scores.get(c, 0) + (m - i)
    return max(scores, key=scores.get)


def _mrrv(rankings: list[list[str]]) -> str:
    scores = {}
    for r in rankings:
        for i, c in enumerate(r):
            scores[c] = scores.get(c, 0.0) + 1.0 / (i + 1)
    return max(scores, key=scores.get)


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_RANKED_VOTING"

    def before_task_call(self, payload):
        if self.store.get("responses") is None:
            self.store["responses"] = []
            self.store["branch_payload"] = payload
        return payload

    def before_chat_message(self, payload):
        msgs = list(h.messages(payload))
        prompt = (
            f"\n\nAfter reasoning, provide the top {NUM_CANDIDATES} most likely answers "
            "ranked best to worst:\nRANKED ANSWERS:\n"
            + "\n".join(f"{i+1}. [answer]" for i in range(NUM_CANDIDATES))
        )
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], dict) and msgs[i].get("role") == "user":
                msgs[i] = h.append_content(msgs[i], prompt)
                break
        payload = dict(payload)
        payload["messages"] = msgs
        return payload

    def after_chat_message(self, response):
        return response, None

    def after_tool_call(self, payload):
        return payload

    def after_task_call(self, response):
        if response is not None:
            self.store["responses"].append(response)
        branch_num = len(self.store["responses"])
        print(f"[RANKED_VOTING] branch {branch_num}/{BRANCHES}", flush=True)
        if len(self.store["responses"]) < BRANCHES:
            return None, self.store["branch_payload"]
        rankings = [_extract_ranked(r, NUM_CANDIDATES) for r in self.store["responses"]]
        rankings = [r for r in rankings if r]
        if not rankings:
            best = self.store["responses"][0]
        else:
            if MODE == "irv":
                winner = _irv(rankings)
            elif MODE == "mrrv":
                winner = _mrrv(rankings)
            else:
                winner = _bcv(rankings)
            print(f"[RANKED_VOTING] mode={MODE}, winner={winner[:80]}", flush=True)
            best = next(
                (r for r in self.store["responses"] if winner in str(h.message(r).get("content", ""))),
                self.store["responses"][0],
            )
        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response
