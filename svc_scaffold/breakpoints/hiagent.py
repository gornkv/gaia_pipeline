from __future__ import annotations

# HiAgent (arxiv 2408.09559)
# Hierarchical working memory management via subgoals.
# Prompts model to formulate subgoals before acting, and summarizes
# observations when subgoals change.

import hashlib
import json
import re
from typing import Any

from svc_scaffold.breakpoints import common as h

SUBGOAL_PROMPT = (
    "Before taking any action, clearly state your current subgoal in one sentence. "
    "Choose a bounded step that you expect to complete in roughly 1 to 15 tool calls; split broader work "
    "into smaller subgoals. Format: SUBGOAL: <one sentence description>. Reuse the exact same SUBGOAL "
    "line while continuing the same subgoal; change it only after that subgoal is completed or abandoned."
)

SUBGOAL_PATTERN = re.compile(r"(?:^|\n)\s*SUBGOAL:\s*([^\r\n]+)", re.IGNORECASE)
SUMMARY_MESSAGE_NAME = "hiagent_summary"


class Breakpoints:
    def __init__(self, model_client=None):
        self.store: dict[str, Any] = {}
        self._current_subgoal: str = ""
        self._subgoal_steps: int = 0
        self._subgoal_start_idx: int = 0
        self._summary_cache: dict[str, str] = {}
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "hiagent"

    def _ensure_subgoal_prompt(self, payload):
        msgs = list(h.messages(payload))
        if any(
            isinstance(msg, dict)
            and msg.get("role") == "system"
            and SUBGOAL_PROMPT in str(msg.get("content") or "")
            for msg in msgs
        ):
            return payload

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

    def before_task_call(self, payload):
        self._current_subgoal = ""
        self._subgoal_steps = 0
        self._subgoal_start_idx = 0
        self._summary_cache = {}
        return self._ensure_subgoal_prompt(payload)

    def before_chat_message(self, payload):
        return self._ensure_subgoal_prompt(payload)

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
                role = str(item.get("role") or "unknown").upper()
                prompt += f"{role}: {item.get('content', '')}\n"
                if isinstance(item.get("tool_calls"), list):
                    prompt += (
                        "TOOL_CALLS: "
                        + json.dumps(item["tool_calls"], ensure_ascii=False, sort_keys=True, default=str)
                        + "\n"
                    )
                if isinstance(item.get("function_call"), dict):
                    prompt += (
                        "FUNCTION_CALL: "
                        + json.dumps(item["function_call"], ensure_ascii=False, sort_keys=True, default=str)
                        + "\n"
                    )
                if item.get("tool_call_id"):
                    prompt += f"TOOL_CALL_ID: {item['tool_call_id']}\n"
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

    @staticmethod
    def _subgoal_transitions(messages: list[Any]) -> list[tuple[int, str]]:
        transitions: list[tuple[int, str]] = []
        for i, msg in enumerate(messages):
            if (
                not isinstance(msg, dict)
                or msg.get("role") != "assistant"
                or str(msg.get("name") or "").endswith("_summary")
            ):
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            matches = SUBGOAL_PATTERN.findall(content)
            if not matches:
                continue
            candidate = matches[-1].strip()
            if candidate and (not transitions or candidate != transitions[-1][1]):
                transitions.append((i, candidate))
        return transitions

    @staticmethod
    def _summary_key(subgoal: str, fragment: list[Any]) -> str:
        serialized = json.dumps(
            {"subgoal": subgoal, "messages": fragment},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def after_tool_call(self, payload):
        msgs = list(h.messages(payload))

        transitions = self._subgoal_transitions(msgs)
        if not transitions:
            self._subgoal_steps += 1
            return payload

        new_subgoal_start, new_subgoal = transitions[-1]
        created_summaries = 0
        if len(transitions) > 1:
            compacted = list(msgs[: transitions[0][0]])
            for transition_idx in range(len(transitions) - 1):
                fragment_start, completed_subgoal = transitions[transition_idx]
                fragment_end = transitions[transition_idx + 1][0]
                fragment = msgs[fragment_start:fragment_end]
                cache_key = self._summary_key(completed_subgoal, fragment)
                summary_text = self._summary_cache.get(cache_key)
                if summary_text is None:
                    summary_text = await self._summarize_subgoal(completed_subgoal, fragment)
                    self._summary_cache[cache_key] = summary_text
                    created_summaries += 1
                compacted.append(
                    {
                        "role": "assistant",
                        "name": SUMMARY_MESSAGE_NAME,
                        "content": f"[HiAgent summary] {summary_text}",
                    }
                )
            compacted.extend(msgs[new_subgoal_start:])
            payload = dict(payload)
            payload["messages"] = compacted
            new_subgoal_start = len(compacted) - len(msgs[new_subgoal_start:])

        if new_subgoal != self._current_subgoal:
            previous_subgoal = self._current_subgoal
            self._current_subgoal = new_subgoal
            self._subgoal_steps = 0
            self._subgoal_start_idx = new_subgoal_start
            if previous_subgoal:
                print(
                    f"[HIAGENT] subgoal changed: {previous_subgoal} -> {new_subgoal}; "
                    f"created {created_summaries} new summary(s)",
                    flush=True,
                )
        else:
            self._subgoal_steps += 1
        return payload

    def after_task_call(self, response):
        self._current_subgoal = ""
        self._subgoal_steps = 0
        self._subgoal_start_idx = 0
        self._summary_cache = {}
        return response, None

    def before_tool_call(self, response):
        return response
