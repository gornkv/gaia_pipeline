from __future__ import annotations

# Structured Note-taking (Kaggle agents-intensive-capstone 2025)
# Adds write_note and read_notes tools to the system prompt for
# persistent just-in-time memory without growing context.

from typing import Any

import svc_scaffold.openai_helpers as h

NOTE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Write a note to persistent memory. Use this to record important facts.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "Content to write"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": "Read all previously written notes from persistent memory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_INSTRUCTION = (
    "You have access to write_note and read_notes tools for persistent memory. "
    "Use write_note to record key facts, numbers, or intermediate conclusions. "
    "Use read_notes at the beginning of complex tasks to recall previously recorded information."
)


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_STRUCTURED_NOTES"

    def before_task_call(self, payload):
        payload = dict(payload)
        tools = list(payload.get("tools", []))
        for tool in NOTE_TOOLS:
            if tool not in tools:
                tools.append(tool)
        payload["tools"] = tools
        msgs = list(h.messages(payload))
        injected = False
        for i, msg in enumerate(msgs):
            if isinstance(msg, dict) and msg.get("role") == "system":
                msgs[i] = h.append_content(msg, "\n\n" + SYSTEM_INSTRUCTION)
                injected = True
                break
        if not injected:
            msgs.insert(0, {"role": "system", "content": SYSTEM_INSTRUCTION})
        payload["messages"] = msgs
        print("[STRUCTURED_NOTES] injected note tools and instructions", flush=True)
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_chat_message(self, response):
        return response, None

    def after_tool_call(self, payload):
        return payload

    def after_task_call(self, response):
        return response, None

    def before_tool_call(self, response):
        return response
