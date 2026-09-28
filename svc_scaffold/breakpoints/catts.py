from __future__ import annotations

# CATTS (arxiv 2602.12276) – per-step action sampling with entropy gate.
# At each intermediate tool-call step, N_SAMPLES responses are collected
# by re-running the same context. If entropy is below threshold (consensus),
# the majority candidate is taken directly. If entropy is high (uncertain),
# an LLM-Arbiter is called to pick the best candidate.

import math
import os
import re
from collections import Counter
from typing import Any

import svc_scaffold.openai_helpers as h

N_SAMPLES = int(os.getenv("CATTS_N_SAMPLES", "3"))
ENTROPY_THRESHOLD = float(os.getenv("CATTS_ENTROPY_THRESHOLD", "0.5"))


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
    def __init__(self, model_client=None):
        self.store: dict[str, Any] = {}
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "FEATURE_CATTS"

    def before_task_call(self, payload):
        return payload

    def before_chat_message(self, payload):
        self.store["last_payload"] = payload
        payload = dict(payload)
        payload.setdefault("temperature", 0.7)
        return payload

    def after_tool_call(self, payload):
        return payload

    async def _arbitrate(self, candidates: list[dict]) -> dict:
        lines = []
        for i, r in enumerate(candidates):
            calls = h.tool_calls(r)
            parts = "; ".join(
                f"{c.get('function', {}).get('name', '?')}({(c.get('function', {}).get('arguments') or '')[:120]})"
                for c in calls
            )
            lines.append(f"Candidate {i + 1}: {parts}")
        prompt = (
            "You are a tool-call arbiter. The agent must decide which tool call to execute next.\n\n"
            + "\n".join(lines)
            + "\n\nSelect the best candidate. Reply ONLY with its number."
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 5,
        }
        result = await self.model_client.chat_completions(payload)
        content = (h.message(result).get("content") or "").strip()
        match = re.search(r"\d+", content)
        idx = int(match.group()) - 1 if match else 0
        idx = max(0, min(idx, len(candidates) - 1))
        print(f"[CATTS] arbiter chose candidate {idx + 1}/{len(candidates)}", flush=True)
        return candidates[idx]

    async def after_chat_message(self, response):
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

        if ent <= ENTROPY_THRESHOLD or not self.model_client:
            majority_sig = counts.most_common(1)[0][0]
            best = next(r for r in samples if _tool_sig(r) == majority_sig)
            print(
                f"[CATTS] entropy={ent:.3f} ≤ {ENTROPY_THRESHOLD} → consensus, "
                f"tool={majority_sig[0][0] if majority_sig else '?'}",
                flush=True,
            )
        else:
            print(f"[CATTS] entropy={ent:.3f} > {ENTROPY_THRESHOLD} → LLM-Arbiter", flush=True)
            best = await self._arbitrate(samples)

        self.store["step_samples"] = []
        return best, None

    def after_task_call(self, response):
        retry = self.store.pop("retry_payload", None)
        if retry is not None:
            return None, retry
        return response, None

    def before_tool_call(self, response):
        return response
