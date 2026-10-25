"""Общая библиотека OpenAI-сообщений и контекста для всех фич."""

from __future__ import annotations

from typing import Any


def messages(payload: dict[str, Any]) -> list[Any]:
    return payload.get("messages") if isinstance(payload.get("messages"), list) else []


def message(response: dict[str, Any]) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    value = choice.get("message") if isinstance(choice, dict) else None
    return value if isinstance(value, dict) else {}


def tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    value = message(response)
    if isinstance(value.get("tool_calls"), list):
        calls.extend(call for call in value["tool_calls"] if isinstance(call, dict))
    if isinstance(value.get("function_call"), dict):
        calls.append(value["function_call"])
    return calls


def has_tool_call(response: dict[str, Any]) -> bool:
    return bool(tool_calls(response))


def has_tool_result(payload: dict[str, Any]) -> bool:
    return any(
        isinstance(item, dict) and item.get("role") in {"tool", "function"}
        for item in messages(payload)
    )


def is_task_start(payload: dict[str, Any]) -> bool:
    roles = [item.get("role") for item in messages(payload) if isinstance(item, dict)]
    return roles.count("user") == 1 and not {"assistant", "tool", "function"}.intersection(roles)


def is_visible_task_response(response: dict[str, Any] | None) -> bool:
    return isinstance(response, dict) and not has_tool_call(response) and bool(message(response).get("content"))


def set_message(response: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    response = dict(response)
    response["choices"] = list(response.get("choices") or [{}])
    response["choices"][0] = dict(response["choices"][0])
    response["choices"][0]["message"] = value
    return response


def append_content(item: dict[str, Any], value: Any) -> dict[str, Any]:
    item = dict(item)
    item["content"] = f"{item.get('content') or ''}\n{value}".strip()
    return item


class BranchContextManager:
    """Не даёт истории предыдущих веток попадать в текущую ветку."""

    def __init__(self) -> None:
        self._initial: list[Any] = []
        self._tail: list[Any] = []
        self._active = False

    def start(self, branch_payload: dict[str, Any]) -> None:
        self._initial = list(messages(branch_payload))
        self._tail = []
        self._active = True

    def stop(self) -> None:
        self._active = False
        self._tail = []

    def before_chat_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._active:
            return payload
        payload = dict(payload)
        payload["messages"] = self._initial + self._tail
        return payload

    def after_chat_message(self, response: dict[str, Any] | None) -> None:
        if self._active and response:
            value = message(response)
            if value:
                self._tail.append(value)

    def after_tool_call(self, payload: dict[str, Any]) -> None:
        if not self._active or not self._tail:
            return
        last = self._tail[-1]
        if not isinstance(last, dict) or last.get("role") != "assistant":
            return

        pending = {
            call["id"] for call in last.get("tool_calls", [])
            if isinstance(call, dict) and "id" in call
        }
        received = {
            item.get("tool_call_id") for item in self._tail
            if isinstance(item, dict) and item.get("role") in {"tool", "function"}
        }
        needed = pending - received
        results = [
            item for item in messages(payload)
            if isinstance(item, dict) and item.get("role") in {"tool", "function"}
        ]
        if needed:
            self._tail.extend(item for item in results if item.get("tool_call_id") in needed)
        elif results and results[-1] not in self._tail:
            self._tail.append(results[-1])
