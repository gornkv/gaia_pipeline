from __future__ import annotations

# PiCSAR – Probabilistic Confidence Selection And Ranking (arxiv 2508.21787)
# Scores reasoning chains by joint log-likelihood of reasoning + answer.
# Requires logprobs from the API. Falls back gracefully if unavailable.
#
#   Score = Reasoning Confidence + Answer Confidence
#   Reasoning Confidence = sum of token logprobs in the reasoning chain
#   Answer Confidence = logprob of the final answer given reasoning + prompt

import math, os, re
from typing import Any

import requests
import svc_scaffold.openai_helpers as h

BRANCHES = int(os.getenv("PICSAR_BRANCHES", "6"))
USE_NORMALIZED = os.getenv("PICSAR_NORMALIZED", "0") == "1"
BASE_URL = os.environ.get("BASE_MODEL_API_BASE_URL", "http://127.0.0.1:18080/v1")

ANSWER_CONFIDENCE_PROMPT = (
    "Based on the reasoning above, what is the final answer? "
    "Provide ONLY the final answer, nothing else."
)


def _extract_final_answer(response: dict) -> str:
    """Extract the final answer from the reasoning chain."""
    content = h.message(response).get("content", "") or ""
    if not isinstance(content, str):
        return ""
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    for line in reversed(lines):
        if "answer" in line.lower() and any(c.isdigit() for c in line):
            return line
    return lines[-1] if lines else content.strip()


def _compute_answer_confidence(response: dict) -> float:
    """
    Compute answer confidence by asking the model to state the final answer
    given the reasoning chain, and taking the logprob of that answer.
    """
    reasoning = h.message(response).get("content", "") or ""
    if not reasoning.strip():
        return 0.0

    # Truncate reasoning to avoid context overflow
    truncated = reasoning[-2000:] if len(reasoning) > 2000 else reasoning

    try:
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": truncated},
                    {"role": "user", "content": ANSWER_CONFIDENCE_PROMPT},
                ],
                "max_tokens": 50,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 1,
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]

        # Extract token logprobs
        logprobs = choice.get("logprobs")
        if logprobs and "token_logprobs" in logprobs:
            token_logprobs = logprobs["token_logprobs"]
            if token_logprobs:
                return sum(lp for lp in token_logprobs if lp is not None)

        return 0.0
    except Exception as e:
        print(f"[PICSAR] Answer confidence failed: {e}", flush=True)
        return 0.0


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}
        self._logprobs_available = True

    @staticmethod
    def feature_name():
        return "FEATURE_PICSAR"

    def before_task_call(self, payload):
        if self.store.get("responses") is None:
            self.store["responses"] = []
            self.store["branch_payload"] = payload
            self.store["reasoning_scores"] = []  # Reasoning Confidence
            self.store["answer_scores"] = []      # Answer Confidence
        return payload

    def before_chat_message(self, payload):
        payload = dict(payload)
        payload["logprobs"] = True
        payload["top_logprobs"] = 1
        return payload

    def after_chat_message(self, response):
        """Extract reasoning confidence from logprobs."""
        if response is None:
            return response, None

        logprobs_score = None
        try:
            choices = response.get("choices", [])
            if choices:
                logprobs = choices[0].get("logprobs")
                if logprobs:
                    content_probs = logprobs.get("content")
                    if isinstance(content_probs, list) and content_probs:
                        logprobs_score = sum(
                            item["logprob"] for item in content_probs
                            if item.get("logprob") is not None
                        )
                    # Old format: token_logprobs is a flat list of floats
                    elif "token_logprobs" in logprobs:
                        token_logprobs = logprobs["token_logprobs"]
                        if token_logprobs:
                            logprobs_score = sum(
                                lp for lp in token_logprobs if lp is not None
                            )
        except Exception:
            pass

        if logprobs_score is None and self._logprobs_available:
            print(
                "[PICSAR] WARNING: logprobs not available. "
                "Falling back to LLM-judge scoring.",
                flush=True,
            )
            self._logprobs_available = False

        self.store["_last_reasoning_score"] = logprobs_score
        return response, None

    def after_tool_call(self, payload):
        return payload

    def after_task_call(self, response):
        if response is not None:
            # Reasoning confidence
            reasoning_score = self.store.get("_last_reasoning_score", 0.0)
            if reasoning_score is None:
                reasoning_score = 0.0
            self.store["reasoning_scores"].append(reasoning_score)

            # Answer confidence
            if self._logprobs_available:
                answer_score = _compute_answer_confidence(response)
            else:
                answer_score = 0.0
            self.store["answer_scores"].append(answer_score)

            self.store["responses"].append(response)

        branch_num = len(self.store["responses"])
        print(
            f"[PICSAR] branch {branch_num}/{BRANCHES} "
            f"(logprobs={'✓' if self._logprobs_available else '✗'})",
            flush=True,
        )

        if len(self.store["responses"]) < BRANCHES:
            return None, self.store["branch_payload"]

        responses = [r for r in self.store["responses"] if r is not None]
        reasoning_scores = self.store["reasoning_scores"]
        answer_scores = self.store["answer_scores"]

        if not responses:
            best = response
        elif not self._logprobs_available or all(s == 0.0 for s in reasoning_scores):
            # Fallback: LLM-judge scoring
            print("[PICSAR] Using fallback: LLM-judge scoring", flush=True)
            best = max(responses, key=lambda r: len(h.message(r).get("content", "")))
        else:
            # PiCSAR: joint score = reasoning + answer confidence
            if USE_NORMALIZED:
                lengths = [
                    len(h.message(r).get("content", "").split()) for r in responses
                ]
                joint_scores = [
                    (rs + ans) / max(1, l)
                    for rs, ans, l in zip(reasoning_scores, answer_scores, lengths)
                ]
            else:
                joint_scores = [
                    rs + ans
                    for rs, ans in zip(reasoning_scores, answer_scores)
                ]

            best_idx = max(range(len(joint_scores)), key=lambda i: joint_scores[i])
            print(
                f"[PICSAR] Scores: reasoning={[f'{s:.1f}' for s in reasoning_scores]}, "
                f"answer={[f'{s:.1f}' for s in answer_scores]}, "
                f"joint={[f'{s:.1f}' for s in joint_scores]}",
                flush=True,
            )
            print(
                f"[PICSAR] Selected candidate {best_idx+1}/{len(responses)} "
                f"(joint_score={joint_scores[best_idx]:.1f})",
                flush=True,
            )
            best = responses[best_idx]

        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response
