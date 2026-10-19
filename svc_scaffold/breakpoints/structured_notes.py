from __future__ import annotations

# Structured Note-taking (Kaggle agents-intensive-capstone 2025)
# Adds write_note and read_notes tools to the system prompt for
# persistent just-in-time memory without growing context.

import json
import uuid
from typing import Any

from svc_scaffold.breakpoints import common as h

NOTE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": (
                "Save important facts for later steps of the task. You may save several related facts in one "
                "call. Write each fact as one short, self-contained sentence. Call this tool alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "One or more important facts, with each fact written as a short, "
                            "self-contained sentence"
                        ),
                    }
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": (
                "Return all notes saved for the current task. "
                "Call this tool alone."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

SYSTEM_INSTRUCTION = (
    "You have persistent notes for the current task. These notes survive context compression.\n"
    "Use write_note when you learn information that will probably be important later.\n"
    "- You may save several related facts in one call.\n"
    "- Write each fact as one short, self-contained sentence.\n"
    "- Call write_note alone. Do not call another tool in the same response.\n"
    "Use read_notes when you need information saved earlier.\n"
    "- Call read_notes alone. It returns all saved notes."
)

NOTE_TOOL_NAMES = {"write_note", "read_notes"}


class Breakpoints:
    def __init__(self, model_client=None):
        self.store: dict[str, Any] = {}
        self.store["notes"] = []
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "structured_notes"

    @staticmethod
    def _tool_name(tool: Any) -> str:
        if not isinstance(tool, dict):
            return ""
        function = tool.get("function")
        if isinstance(function, dict):
            return str(function.get("name") or "")
        return str(tool.get("name") or "")

    def _ensure_note_capabilities(self, payload):
        payload = dict(payload)
        raw_tools = payload.get("tools")
        tools = list(raw_tools) if isinstance(raw_tools, list) else []
        tool_names = {self._tool_name(tool) for tool in tools}
        for tool in NOTE_TOOLS:
            tool_name = self._tool_name(tool)
            if tool_name not in tool_names:
                tools.append(tool)
                tool_names.add(tool_name)
        payload["tools"] = tools

        msgs = list(h.messages(payload))
        if any(
            isinstance(msg, dict)
            and msg.get("role") == "system"
            and SYSTEM_INSTRUCTION in str(msg.get("content") or "")
            for msg in msgs
        ):
            payload["messages"] = msgs
            return payload

        injected = False
        for i, msg in enumerate(msgs):
            if isinstance(msg, dict) and msg.get("role") == "system":
                msgs[i] = h.append_content(msg, "\n\n" + SYSTEM_INSTRUCTION)
                injected = True
                break
        if not injected:
            msgs.insert(0, {"role": "system", "content": SYSTEM_INSTRUCTION})
        payload["messages"] = msgs
        print("[STRUCTURED_NOTES] restored note capabilities", flush=True)
        return payload

    def before_task_call(self, payload):
        self.store["notes"] = []
        return self._ensure_note_capabilities(payload)

    def before_chat_message(self, payload):
        return self._ensure_note_capabilities(payload)

    def after_chat_message(self, response):
        return response, None

    def after_tool_call(self, payload):
        return payload

    def after_task_call(self, response):
        return response, None

    def _execute_note_call(self, call: dict[str, Any]) -> str:
        function = call.get("function")
        if not isinstance(function, dict):
            return "ERROR: missing function payload."
        raw_arguments = function.get("arguments", {})
        try:
            arguments = raw_arguments if isinstance(raw_arguments, dict) else json.loads(raw_arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            return "ERROR: arguments contain invalid JSON."
        if not isinstance(arguments, dict):
            return "ERROR: arguments must be a JSON object."

        name = str(function.get("name") or "")
        if name == "write_note":
            content = arguments.get("content")
            if not isinstance(content, str) or not content.strip():
                return 'ERROR: write_note requires a non-empty string field "content".'
            self.store.setdefault("notes", []).append(content.strip())
            return "OK"

        note_text = "\n".join(str(note) for note in self.store.get("notes", []))
        return note_text or "No notes available."

    def before_tool_call(self, response, payload=None):
        assistant_message = dict(h.message(response))
        tool_calls = [
            call for call in assistant_message.get("tool_calls", []) if isinstance(call, dict)
        ]
        internal_calls = [call for call in tool_calls if self._tool_name(call) in NOTE_TOOL_NAMES]
        if not internal_calls:
            return response

        messages = list(h.messages(payload)) if payload is not None else []
        normalized_calls = []
        internal_results: list[dict[str, Any]] = []
        for call in internal_calls:
            normalized = {
                **call,
                "id": str(call.get("id") or f"structured-notes-{uuid.uuid4().hex}"),
            }
            normalized_calls.append(normalized)
            internal_results.append(
                {
                    "role": "tool",
                    "tool_call_id": normalized["id"],
                    "content": self._execute_note_call(call),
                }
            )
        assistant_message["tool_calls"] = normalized_calls
        assistant_message.pop("function_call", None)

        # Run the model again using only internally handled calls. Any external
        # calls from a mixed response are deliberately deferred: after seeing
        # the note results, the model can reissue or revise them.
        new_msgs = list(messages)
        assistant_message.setdefault("role", "assistant")
        if assistant_message:
            new_msgs.append(assistant_message)
        new_msgs.extend(internal_results)
        new_payload = dict(payload) if payload is not None else {}
        new_payload["messages"] = new_msgs
        internal_response = h.set_message(response, assistant_message)
        print(
            f"[STRUCTURED_NOTES] handled {len(internal_calls)} note tool call(s) internally; "
            f"deferred {len(tool_calls) - len(internal_calls)} external call(s)",
            flush=True,
        )
        return internal_response, new_payload
