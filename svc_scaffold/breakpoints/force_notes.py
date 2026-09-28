from __future__ import annotations

# Always-on nudge (experiment A): FORCE a weak model to actually USE the
# structured_notes write_note/read_notes tools. Two fixes over the soft version:
#   1) re-inject the note TOOLS every step (structured_notes only adds them at
#      task start, so after step 1 they vanish from Inspect-rebuilt payloads and
#      the model literally cannot call them);
#   2) inject a strong imperative into the last user message every step.
# Pair with FEATURE_STRUCTURED_NOTES (which still HANDLES the calls) + TIT.

import svc_scaffold.openai_helpers as h

NOTE_TOOLS = [
    {"type": "function", "function": {
        "name": "write_note",
        "description": "Write a note to persistent memory. Use this to record important facts.",
        "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "Content to write"}}, "required": ["content"]}}},
    {"type": "function", "function": {
        "name": "read_notes",
        "description": "Read all previously written notes from persistent memory.",
        "parameters": {"type": "object", "properties": {}}}},
]

MARK = "[SAVE-NOTES]"
INSTR = (
    "\n\n" + MARK + " You have write_note and read_notes tools for persistent memory. "
    "Whenever you discover a concrete fact, number, name, date, URL, or intermediate result, "
    "you MUST immediately call write_note to save it BEFORE doing anything else. "
    "Before you submit your final answer, call read_notes to review everything you saved. "
    "Do not rely on memory alone - save every useful fact the moment you find it."
)


class Breakpoints:
    def __init__(self, model_client=None):
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "FEATURE_FORCE_NOTES"

    def before_task_call(self, payload):
        return payload

    def before_chat_message(self, payload):
        payload = dict(payload)
        # 1) ensure note tools are present EVERY step
        tools = list(payload.get("tools", []))
        names = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
        for t in NOTE_TOOLS:
            if t["function"]["name"] not in names:
                tools.append(t)
        payload["tools"] = tools
        # 2) inject the imperative into the last user message
        msgs = list(h.messages(payload))
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str) and MARK in content:
                    break
                msgs[i] = h.append_content(m, INSTR)
                break
        payload["messages"] = msgs
        return payload

    def after_chat_message(self, response):
        return response, None

    def after_tool_call(self, payload):
        return payload

    def after_task_call(self, response):
        return response, None

    def before_tool_call(self, response, payload=None):
        return response
