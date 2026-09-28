from __future__ import annotations

# HiAgent (arxiv 2408.09559) — subgoal prompting + observation summarization when
# the subgoal changes. The subgoal-summariser runs on the strong HELPER model
# (helper.py) when configured, else the eval model.

from typing import Any

import svc_scaffold.openai_helpers as h
from svc_scaffold.helper import helper_completion

SUBGOAL_PROMPT = (
    "Before taking any action, clearly state your current subgoal in one sentence. "
    "Format: SUBGOAL: <one sentence description>"
)


class Breakpoints:
    def __init__(self, model_client=None):
        self.store: dict[str, Any] = {}
        self._current_subgoal: str = ""
        self._subgoal_steps: int = 0
        self._subgoal_start_idx: int = 0
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "FEATURE_HIAGENT"

    def before_task_call(self, payload):
        self._current_subgoal = ""
        self._subgoal_steps = 0
        self._subgoal_start_idx = 0
        msgs = list(h.messages(payload))
        injected = False
        for i, msg in enumerate(msgs):
            if isinstance(msg, dict) and msg.get("role") == "system":
                msgs[i] = h.append_content(msg, "\n\n" + SUBGOAL_PROMPT)
                injected = True
                break
        if not injected:
            msgs.insert(0, {"role": "system", "content": SUBGOAL_PROMPT})
        payload = dict(payload)
        payload["messages"] = msgs
        print("[HIAGENT] injected subgoal prompt", flush=True)
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_chat_message(self, response):
        return response, None

    async def _summarize_subgoal(self, subgoal: str, messages: list[Any]) -> str:
        prompt = (
            f"Summarize what was accomplished and what was learned while working on the subgoal '{subgoal}'. "
            "Keep the summary concise and include only the information needed to continue the task.\n\n"
            "Conversation fragment:\n"
        )
        for item in messages:
            if isinstance(item, dict):
                prompt += f"{item.get('role', '').upper()}: {item.get('content', '')}\n"
            else:
                prompt += f"UNKNOWN: {item}\n"
        content = await helper_completion(prompt, 1024, self.model_client)
        return content or f"[HiAgent] Completed subgoal '{subgoal}'."

    async def after_tool_call(self, payload):
        msgs = list(h.messages(payload))

        new_subgoal = ""
        new_subgoal_start = 0
        prev_subgoal = ""
        prev_subgoal_start = 0
        for i, msg in enumerate(msgs):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and "SUBGOAL:" in content:
                    candidate = content.split("SUBGOAL:")[-1].split("\n")[0].strip()
                    if candidate != new_subgoal:
                        prev_subgoal = new_subgoal
                        prev_subgoal_start = new_subgoal_start
                        new_subgoal = candidate
                        new_subgoal_start = i

        if new_subgoal and new_subgoal != self._current_subgoal:
            if self._current_subgoal and prev_subgoal == self._current_subgoal:
                fragment = msgs[prev_subgoal_start:new_subgoal_start]
                if fragment:
                    summary_text = await self._summarize_subgoal(self._current_subgoal, fragment)
                    head = list(msgs[:prev_subgoal_start])
                    lead = 0
                    while (
                        lead < len(head)
                        and isinstance(head[lead], dict)
                        and head[lead].get("role") == "system"
                    ):
                        lead += 1
                    # the summary is context, not dialogue: it belongs in the
                    # leading system block, not where the fragment used to be
                    compacted = head[:lead]
                    compacted.append({"role": "system", "content": f"[HiAgent summary] {summary_text}"})
                    compacted.extend(head[lead:])
                    compacted.extend(msgs[new_subgoal_start:])
                    payload = dict(payload)
                    payload["messages"] = compacted
                    new_subgoal_start = prev_subgoal_start + 1
                    print(
                        f"[HIAGENT] subgoal changed: {self._current_subgoal} -> {new_subgoal}, compressed previous fragment",
                        flush=True,
                    )
            self._current_subgoal = new_subgoal
            self._subgoal_steps = 0
            self._subgoal_start_idx = new_subgoal_start
        else:
            self._subgoal_steps += 1
        return payload

    def after_task_call(self, response):
        self._current_subgoal = ""
        self._subgoal_steps = 0
        self._subgoal_start_idx = 0
        return response, None

    def before_tool_call(self, response):
        return response
