from __future__ import annotations

# Always-on helper: force a weak tool-calling model (Gemma) to emit plain-text
# reasoning each step, so text-dependent features (ranked_voting, structured_notes)
# have content to work with. Injects into the last user message every step.

import svc_scaffold.openai_helpers as h

MARK = "[THINK-FIRST]"
INSTR = (
    "\n\n" + MARK + " Before your next tool call, FIRST write one or two sentences of "
    "plain text: what you have learned so far and what you will do next and why. "
    "You MUST include this text every turn. Never respond with only a tool call and no text."
)


class Breakpoints:
    def __init__(self, model_client=None):
        self.model_client = model_client

    @staticmethod
    def feature_name():
        return "FEATURE_THINK_IN_TEXT"

    def before_task_call(self, payload):
        return payload

    def before_chat_message(self, payload):
        msgs = list(h.messages(payload))
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str) and MARK in content:
                    break  # already injected, idempotent
                msgs[i] = h.append_content(m, INSTR)
                break
        payload = dict(payload)
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
