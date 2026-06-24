from __future__ import annotations

# Majority of the Bests (MoB) – arxiv 2511.18630
# Bootstraps over generated outputs: creates many subsets of size m,
# picks the best by LLM-judge reward in each, then majority-votes among winners.

import os, re, random
from collections import Counter
from typing import Any

import requests
import svc_scaffold.openai_helpers as h

BRANCHES = int(os.getenv("MOB_BRANCHES", "8"))
BOOTSTRAP = int(os.getenv("MOB_BOOTSTRAP_SAMPLES", "1000"))
SUBSET = int(os.getenv("MOB_SUBSET_SIZE", "3"))
BASE_URL = os.environ.get("BASE_MODEL_API_BASE_URL", "http://127.0.0.1:18080/v1")


def _reward(response: dict) -> float:
    """
    Оценка ответа через LLM-judge (1-10), с fallback на длину.
    В продакшене заменить на внешний reward model (ORM).
    """
    content = h.message(response).get("content", "") or ""
    if not content.strip():
        return 0.0

    # Берём первые 500 символов для скорости
    truncated = content[:500]

    try:
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Rate the quality and correctness of this answer from 1 to 10. "
                            f"Output ONLY the number.\n\nAnswer:\n{truncated}"
                        ),
                    }
                ],
                "max_tokens": 3,
                "temperature": 0,
            },
            timeout=15,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        match = re.search(r"\d+", text)
        score = float(match.group()) if match else 5.0
        return max(1.0, min(10.0, score))
    except Exception:
        # Fallback: длина ответа (чем длиннее, тем выше оценка)
        return float(len(content))


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

        # Вычисляем reward для каждого ответа (один раз)
        rewards = [_reward(r) for r in responses]
        print(f"[MOB] rewards: {[f'{s:.1f}' for s in rewards]}", flush=True)

        # Бутстреп: создаём BOOTSTRAP подмножеств, в каждом выбираем лучшего
        winners = []
        for _ in range(BOOTSTRAP):
            # Случайное подмножество размера SUBSET
            idx = random.choices(range(len(responses)), k=min(SUBSET, len(responses)))
            # Лучший в подмножестве по reward
            best_idx = max(idx, key=lambda i: rewards[i])
            content = str(h.message(responses[best_idx]).get("content", "")).strip()
            winners.append(content)

        # Majority vote среди победителей бутстрепа
        winner_content = Counter(winners).most_common(1)[0][0]
        votes = Counter(winners).most_common(1)[0][1]
        print(
            f"[MOB] bootstrapped {BOOTSTRAP} subsets (size={SUBSET}), "
            f"winner votes={votes}/{BOOTSTRAP}",
            flush=True,
        )

        # Находим ответ, соответствующий победителю
        best = next(
            (
                r
                for r in responses
                if str(h.message(r).get("content", "")).strip() == winner_content
            ),
            responses[0],
        )

        self.store = {}
        return best, None

    def before_tool_call(self, response):
        return response
