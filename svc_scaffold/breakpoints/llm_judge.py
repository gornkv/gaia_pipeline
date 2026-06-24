from __future__ import annotations

# LLM-as-Judge / ATLAS-style (arxiv 2606.01667)

import os, re
from typing import Any
import requests
import svc_scaffold.openai_helpers as h


BRANCHES = int(os.getenv("LLM_JUDGE_BRANCHES", "3"))
BASE_URL = os.environ.get("BASE_MODEL_API_BASE_URL", "http://127.0.0.1:18080/v1")
MODEL = os.environ.get("BASE_MODEL_NAME", "")
API_KEY = os.environ.get("BASE_MODEL_API_KEY", "")
MAX_CHARS = int(os.getenv("LLM_JUDGE_MAX_CHARS", "2000"))


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

    def before_chat_message(self, payload): return payload
    def after_chat_message(self, response): return response, None
    def after_tool_call(self, payload): return payload

    def after_task_call(self, response):
        if response is not None:
            self.store["responses"].append(response)
        print(f"[LLM_JUDGE] branch {len(self.store['responses'])}/{BRANCHES}", flush=True)
        if len(self.store["responses"]) < BRANCHES:
            return None, self.store["branch_payload"]
        responses = [r for r in self.store["responses"] if r is not None]
        if len(responses) < 2:
            best = responses[0] if responses else response
        else:
            judge_prompt = (
                "You are an expert evaluator. Below are candidate answers to the same "
                "question. Select the BEST answer. Output ONLY the number of the best "
                "candidate.\n\n"
            )
            for i, r in enumerate(responses):
                content = (h.message(r).get("content") or "")[-MAX_CHARS:]
                judge_prompt += f"Candidate {i+1}:\n{content}\n\n"
            judge_prompt += "BEST candidate number:"
            headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
            req = {
                "model": MODEL,
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": 5,
                "temperature": 0,
            }
            try:
                judge_resp = requests.post(
                    f"{BASE_URL.rstrip('/')}/chat/completions",
                    json=req, headers=headers, timeout=30,
                )
                judge_resp.raise_for_status()
                judge_answer = judge_resp.json()["choices"][0]["message"]["content"].strip()
                match = re.search(r"\d+", judge_answer)
                best_idx = int(match.group()) - 1 if match else 0
                best_idx = max(0, min(best_idx, len(responses) - 1))
                best = responses[best_idx]
                print(f"[LLM_JUDGE] Judge selected {best_idx+1}/{len(responses)}", flush=True)
            except Exception as e:
                print(f"[LLM_JUDGE] Judge call failed: {e}, using fallback", flush=True)
                best = max(responses, key=lambda r: len(h.message(r).get("content") or ""))
        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response