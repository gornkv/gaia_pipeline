from __future__ import annotations

# Structured Note-taking (Kaggle agents-intensive-capstone 2025)
# Adds write_note and read_notes tools to the system prompt for
# persistent just-in-time memory without growing context.

import json
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
    def __init__(self, model_client=None):
        self.store: dict[str, Any] = {}
        self.store["notes"] = []
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "FEATURE_STRUCTURED_NOTES"

    def before_task_call(self, payload):
        self.store["notes"] = []
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

    def before_tool_call(self, response, payload=None):
        tool_calls = h.tool_calls(response)
        if not tool_calls:
            return response

        handled = []
        messages = list(h.messages(payload)) if payload is not None else []
        for call in tool_calls:
            # OpenAI tool_calls format: {"id": "...", "type": "function", "function": {"name": ..., "arguments": ...}}
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = call.get("name") or fn.get("name")
            tool_call_id = call.get("id", "")
            if name == "write_note":
                raw_args = call.get("arguments") or fn.get("arguments") or {}
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args if isinstance(raw_args, dict) else {})
                content = args.get("content") if isinstance(args, dict) else None
                if content:
                    self.store["notes"].append(str(content))
                handled.append((call, {"role": "tool", "tool_call_id": tool_call_id, "content": "OK"}))
            elif name == "read_notes":
                note_text = "\n".join(self.store.get("notes", [])) or "No notes available."
                handled.append((call, {"role": "tool", "tool_call_id": tool_call_id, "content": note_text}))

        if not handled:
            return response

        # Synthesize tool results and rerun the model on the augmented history.
        new_msgs = list(messages)
        assistant_message = dict(h.message(response))
        if assistant_message:
            new_msgs.append({"role": "assistant", **assistant_message})
        else:
            new_msgs.append({"role": "assistant", "content": h.message(response).get("content", "")})
        for _, tool_result in handled:
            new_msgs.append(tool_result)
        new_payload = dict(payload) if payload is not None else {}
        new_payload["messages"] = new_msgs
        print(f"[STRUCTURED_NOTES] handled {len(handled)} note tool call(s) internally", flush=True)
        return response, new_payload
