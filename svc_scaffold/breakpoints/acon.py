from __future__ import annotations

# ACON without fine-tuning (arXiv 2510.00615)
# Compresses tool observations via a COMPRESSOR model guided by natural-language
# compression guidelines, iteratively refined from over-budget tasks.
# The compressor runs on the strong HELPER model (helper.py) when configured
# (the paper used gpt-4.1) and falls back to the eval model otherwise.

import hashlib
import os
from typing import Any

import svc_scaffold.openai_helpers as h
from svc_scaffold.helper import helper_completion

TOKEN_BUDGET = int(os.getenv("ACON_TOKEN_BUDGET", "30000"))
COMPRESS_THRESHOLD = int(os.getenv("ACON_COMPRESS_THRESHOLD", "1500"))
COMPRESS_MAX_TOKENS = int(os.getenv("ACON_COMPRESS_MAX_TOKENS", "2048"))

_guidelines: list[str] = [
    "Preserve every fact, number, name, date, URL, code snippet and state needed to continue the task.",
    "Drop navigation chrome, boilerplate, ads, and repeated content.",
    "Keep exact quotes/values verbatim; never paraphrase numbers or identifiers.",
    "Summarize long prose into terse bullet points.",
    "Remove anything already stated earlier in the conversation.",
]
_compress_cache: dict[str, str] = {}


def _guidelines_block() -> str:
    return "Compression guidelines:\n" + "\n".join(f"- {g}" for g in _guidelines)


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)


class Breakpoints:
    def __init__(self, model_client=None):
        self.store: dict[str, Any] = {}
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "FEATURE_ACON"

    def before_task_call(self, payload):
        self.store["tokens_used"] = 0
        return payload

    def before_chat_message(self, payload):
        return payload

    async def _compress_observation(self, text: str) -> str:
        key = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
        if key in _compress_cache:
            return _compress_cache[key]
        prompt = (
            "Compress the following tool observation for an LLM agent, following these guidelines:\n"
            + _guidelines_block()
            + "\n\nOutput ONLY the compressed observation, nothing else.\n\nObservation:\n"
            + text
        )
        out = await helper_completion(prompt, COMPRESS_MAX_TOKENS, self.model_client)
        if not out or len(out) >= len(text):
            _compress_cache[key] = text
            return text
        _compress_cache[key] = out
        return out

    async def after_tool_call(self, payload):
        msgs = list(h.messages(payload))
        changed = 0
        saved = 0
        for i, m in enumerate(msgs):
            if not (isinstance(m, dict) and m.get("role") in ("tool", "function")):
                continue
            text = _as_text(m.get("content"))
            if len(text) <= COMPRESS_THRESHOLD:
                continue
            compressed = await self._compress_observation(text)
            if compressed != text:
                nm = dict(m)
                nm["content"] = compressed
                msgs[i] = nm
                changed += 1
                saved += len(text) - len(compressed)
        if changed:
            payload = dict(payload)
            payload["messages"] = msgs
            print(f"[ACON] compressed {changed} observation(s), saved ~{saved} chars", flush=True)
        return payload

    def after_chat_message(self, response):
        if response is not None:
            usage = response.get("usage") or {}
            ct = usage.get("completion_tokens")
            if ct is None:
                ct = usage.get("output_tokens")
            if isinstance(ct, int):
                self.store["tokens_used"] = self.store.get("tokens_used", 0) + ct
        return response, None

    async def _optimize_guidelines(self, tokens_used: int) -> None:
        prompt = (
            f"An agent's context exceeded the token budget ({tokens_used} used, limit {TOKEN_BUDGET}) "
            "despite these compression guidelines being applied to its tool observations.\n\n"
            "Current guidelines:\n" + "\n".join(f"- {g}" for g in _guidelines) + "\n\n"
            "Rewrite them to compress more aggressively while never dropping facts needed to solve the task. "
            "Output ONLY the updated guidelines as a bullet list, one per line starting with '- '."
        )
        content = await helper_completion(prompt, 400, self.model_client)
        if not content:
            return
        new_lines = [ln.lstrip("- ").strip() for ln in content.split("\n") if ln.strip().startswith("-")]
        if new_lines:
            _guidelines.clear()
            _guidelines.extend(new_lines[:6])
            _compress_cache.clear()
            print(f"[ACON] guidelines refined via helper ({len(_guidelines)} rules)", flush=True)

    async def after_task_call(self, response):
        tokens = self.store.get("tokens_used", 0)
        if tokens > TOKEN_BUDGET:
            await self._optimize_guidelines(tokens)
        else:
            print(f"[ACON] task done, tokens_used={tokens}/{TOKEN_BUDGET}", flush=True)
        self.store = {}
        return response, None

    def before_tool_call(self, response):
        return response
