from __future__ import annotations

# PiCSAR – Probabilistic Confidence Selection And Ranking (arxiv 2508.21787)
# Scores reasoning chains by joint log-likelihood of reasoning + answer.
# Requires logprobs from the API. Falls back gracefully if unavailable.

import math
import os
from typing import Any

import svc_scaffold.openai_helpers as h

BRANCHES = int(os.getenv("PICSAR_BRANCHES", "6"))
USE_NORMALIZED = os.getenv("PICSAR_NORMALIZED", "0") == "1"
ANSWER_PROMPT = "\n\nBased on the reasoning above, what is the final answer?"


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}
        self._logprobs_available = True  # Will be set to False if unavailable

    @staticmethod
    def feature_name():
        return "FEATURE_PICSAR"

    # ─── BP2: before_task_call ───
    def before_task_call(self, payload):
        if self.store.get("responses") is None:
            self.store["responses"] = []
            self.store["branch_payload"] = payload
            self.store["logprobs_scores"] = []
        return payload

    # ─── BP3: before_chat_message ───
    def before_chat_message(self, payload):
        # Request logprobs from the model
        payload = dict(payload)
        payload["logprobs"] = True
        payload["top_logprobs"] = 1  # We only need the chosen token's logprob
        return payload

    # ─── BP4: after_chat_message ───
    def after_chat_message(self, response):
        """Extract reasoning confidence from logprobs."""
        if response is None:
            return response, None

        # Try to extract logprobs
        logprobs_score = None
        try:
            choices = response.get("choices", [])
            if choices:
                logprobs = choices[0].get("logprobs")
                if logprobs and "token_logprobs" in logprobs:
                    # Sum of all token logprobs = reasoning confidence
                    token_logprobs = logprobs["token_logprobs"]
                    if token_logprobs:
                        logprobs_score = sum(
                            lp for lp in token_logprobs if lp is not None
                        )
        except Exception:
            pass

        if logprobs_score is None and self._logprobs_available:
            print(
                "[PICSAR] WARNING: logprobs not available in response. "
                "Model may not support logprobs. Falling back to Self-Consistency.",
                flush=True,
            )
            self._logprobs_available = False

        self.store["_last_logprobs_score"] = logprobs_score
        return response, None

    # ─── BP1: after_tool_call ───
    def after_tool_call(self, payload):
        return payload

    # ─── BP5: after_task_call ───
    def after_task_call(self, response):
        if response is not None:
            reasoning_score = self.store.get("_last_logprobs_score", 0.0)
            if reasoning_score is None:
                reasoning_score = 0.0
            self.store["logprobs_scores"].append(reasoning_score)
            self.store["responses"].append(response)

        branch_num = len(self.store["responses"])
        print(
            f"[PICSAR] branch {branch_num}/{BRANCHES} "
            f"(logprobs={'available' if self._logprobs_available else 'unavailable'})",
            flush=True,
        )

        if len(self.store["responses"]) < BRANCHES:
            return None, self.store["branch_payload"]

        # Select best response
        responses = [r for r in self.store["responses"] if r is not None]
        scores = self.store["logprobs_scores"]

        if not responses:
            best = response
        elif not self._logprobs_available or all(s == 0.0 for s in scores):
            # Fallback: pick the longest response (heuristic)
            print("[PICSAR] Using fallback: selecting longest response", flush=True)
            best = max(responses, key=lambda r: len(h.message(r).get("content", "")))
        else:
            # PiCSAR: select by highest reasoning confidence
            if USE_NORMALIZED:
                lengths = [
                    len(h.message(r).get("content", "").split()) for r in responses
                ]
                normalized = [
                    s / max(1, l) for s, l in zip(scores, lengths)
                ]
                best_idx = max(range(len(normalized)), key=lambda i: normalized[i])
                print(
                    f"[PICSAR] Selected best by normalized score: "
                    f"{normalized[best_idx]:.2f}",
                    flush=True,
                )
            else:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                print(
                    f"[PICSAR] Selected best by reasoning confidence: "
                    f"{scores[best_idx]:.1f}",
                    flush=True,
                )
            best = responses[best_idx]

        self.store = {}
        return best, None

    # ─── BP6: before_tool_call ───
    def before_tool_call(self, response):
        return response
