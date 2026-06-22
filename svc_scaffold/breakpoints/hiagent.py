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
    def __init__(self):
        self.store: dict[str, Any] = {}
        self._current_subgoal: str = ""
        self._subgoal_steps: int = 0

    @staticmethod
    def feature_name():
        return "FEATURE_HIAGENT"

    def before_task_call(self, payload):
        self._current_subgoal = ""
        self._subgoal_steps = 0
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

    def after_tool_call(self, payload):
        msgs = h.messages(payload)
        new_subgoal = ""
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and "SUBGOAL:" in content:
                    new_subgoal = content.split("SUBGOAL:")[-1].split("\n")[0].strip()
                    break
        if new_subgoal and new_subgoal != self._current_subgoal:
            if self._current_subgoal:
                summary = f"[Previous subgoal completed: {self._current_subgoal}. Steps taken: {self._subgoal_steps}]"
                msgs = list(msgs)
                msgs.append({"role": "system", "content": summary})
                payload = dict(payload)
                payload["messages"] = msgs
                print(f"[HIAGENT] subgoal changed: {self._current_subgoal} -> {new_subgoal}", flush=True)
            self._current_subgoal = new_subgoal
            self._subgoal_steps = 0
        else:
            self._subgoal_steps += 1
        return payload

    def after_task_call(self, response):
        self._current_subgoal = ""
        self._subgoal_steps = 0
        return response, None

    def before_tool_call(self, response):
        return response
