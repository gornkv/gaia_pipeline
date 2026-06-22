from __future__ import annotations

# CATTS (arxiv 2602.12276) – per-step action sampling with entropy gate.
# At each intermediate tool-call step, N_SAMPLES responses are collected
# by re-running the same context. If all samples agree on the same tool
# name (low entropy / consensus), any is taken. If they disagree (high
# entropy), majority vote is used as the implicit arbiter (no separate LLM
# call needed, keeping this training- and fine-tuning-free).

import math
import os
from collections import Counter
from typing import Any

import svc_scaffold.openai_helpers as h

N_SAMPLES = int(os.getenv("CATTS_N_SAMPLES", "3"))


def _tool_sig(response: dict) -> tuple:
    return tuple(
        (c.get("function", {}).get("name", ""), c.get("function", {}).get("arguments", ""))
        for c in h.tool_calls(response)
    )


def _entropy(counts: Counter, total: int) -> float:
    e = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            e -= p * math.log2(p)
    return e


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_CATTS"

    def before_task_call(self, payload):
        return payload

    def before_chat_message(self, payload):
        # Capture the context used for this LLM call so we can re-run it
        self.store["last_payload"] = payload
        # Use temperature > 0 to get diverse samples
        payload = dict(payload)
        payload.setdefault("temperature", 0.7)
        return payload

    def after_tool_call(self, payload):
        return payload

    def after_chat_message(self, response):
        # Only sample at tool-call (intermediate) steps
        if response is None or not h.has_tool_call(response):
            self.store.pop("step_samples", None)
            return response, None

        samples: list = self.store.setdefault("step_samples", [])
        samples.append(response)
        print(f"[CATTS] step sample {len(samples)}/{N_SAMPLES}", flush=True)

        if len(samples) < N_SAMPLES:
            self.store["retry_payload"] = self.store["last_payload"]
            return None, None

        sigs = [_tool_sig(r) for r in samples]
        counts: Counter = Counter(sigs)
        ent = _entropy(counts, len(sigs))

        majority_sig = counts.most_common(1)[0][0]
        best = next(r for r in samples if _tool_sig(r) == majority_sig)
        print(f"[CATTS] entropy={ent:.3f}, majority tool={majority_sig[0][0] if majority_sig else '?'}", flush=True)

        self.store["step_samples"] = []
        return best, None

    def after_task_call(self, response):
        # Relay retry payload set by after_chat_message
        retry = self.store.pop("retry_payload", None)
        if retry is not None:
            return None, retry
        return response, None

    def before_tool_call(self, response):
        return response