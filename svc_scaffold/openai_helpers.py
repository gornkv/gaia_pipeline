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
    message_payload = message(response)
    if isinstance(message_payload.get("tool_calls"), list):
        calls.extend(c for c in message_payload.get("tool_calls") if isinstance(c, dict))
    if isinstance(message_payload.get("function_call"), dict):
        calls.append(message_payload.get("function_call"))
    return calls


def has_tool_call(response: dict[str, Any]) -> bool:
    return bool(tool_calls(response))


def has_tool_result(payload: dict[str, Any]) -> bool:
    for item in messages(payload):
        if isinstance(item, dict) and item.get("role") in {"tool", "function"}:
            return True
    return False


def is_task_start(payload: dict[str, Any]) -> bool:
    user_messages = 0
    for item in messages(payload):
        if not isinstance(item, dict):
            continue
        if item.get("role") in {"assistant", "tool", "function"}:
            return False
        if item.get("role") == "user":
            user_messages += 1
    return user_messages == 1


def is_visible_task_response(response: dict[str, Any] | None) -> bool:
    if not isinstance(response, dict) or has_tool_call(response):
        return False
    return bool(message(response).get("content"))


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
