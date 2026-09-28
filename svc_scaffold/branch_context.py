from __future__ import annotations

from typing import Any

import svc_scaffold.openai_helpers as h


class BranchContextManager:
    """
    Prevents context accumulation across branches in multi-rollout techniques.

    When inspect runs a multi-branch technique, it keeps appending messages to
    its conversation state. On each branch restart the scaffold returns the
    initial branch_payload, but inspect's next request already includes all
    previous branches' history. This class strips that accumulated history so
    each branch only sees the initial prompt + its own messages.
    """

    def __init__(self) -> None:
        self._initial: list[Any] = []
        self._tail: list[Any] = []
        self._active: bool = False

    def start(self, branch_payload: dict[str, Any]) -> None:
        self._initial = list(h.messages(branch_payload))
        self._tail = []
        self._active = True

    def stop(self) -> None:
        self._active = False
        self._tail = []

    def before_chat_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._active:
            return payload
        p = dict(payload)
        p["messages"] = list(self._initial) + list(self._tail)
        return p

    def after_chat_message(self, response: dict[str, Any] | None) -> None:
        if not self._active or not response:
            return
        msg = h.message(response)
        if msg:
            self._tail.append(msg)

    def after_tool_call(self, payload: dict[str, Any]) -> None:
        if not self._active or not self._tail:
            return
        last = self._tail[-1]
        if not isinstance(last, dict) or last.get("role") != "assistant":
            return

        # Find tool results for pending tool call IDs
        pending = {tc["id"] for tc in last.get("tool_calls", []) if isinstance(tc, dict) and "id" in tc}
        received = {
            m.get("tool_call_id")
            for m in self._tail
            if isinstance(m, dict) and m.get("role") in {"tool", "function"}
        }
        needed = pending - received

        all_msgs = h.messages(payload)

        if needed:
            for msg in all_msgs:
                if isinstance(msg, dict) and msg.get("role") in {"tool", "function"}:
                    if msg.get("tool_call_id") in needed:
                        self._tail.append(msg)
        else:
            # Fallback for function_call format without IDs: take the last tool message
            for msg in reversed(all_msgs):
                if isinstance(msg, dict) and msg.get("role") in {"tool", "function"}:
                    if msg not in self._tail:
                        self._tail.append(msg)
                    break
