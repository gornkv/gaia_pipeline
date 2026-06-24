from __future__ import annotations

# HiAgent (arxiv 2408.09559)
# Hierarchical working memory management via subgoals.
# Prompts model to formulate subgoals before acting, and summarizes
# observations when subgoals change.

from typing import Any

import svc_scaffold.openai_helpers as h

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

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        if not self.model_client:
            return f"[HiAgent] Completed subgoal '{subgoal}'." 
        result = await self.model_client.chat_completions(payload)
        content = (h.message(result).get("content") or "").strip()
        return content or f"[HiAgent] Completed subgoal '{subgoal}'."

    async def after_tool_call(self, payload):
        msgs = list(h.messages(payload))

        # Scan full history to find actual subgoal boundary positions.
        # GAIA sends the full uncompacted history on every call, so persistent
        # _subgoal_start_idx becomes stale across turns, rescan each time.
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
                    compacted = list(msgs[:prev_subgoal_start])
                    compacted.append({"role": "system", "content": f"[HiAgent summary] {summary_text}"})
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
