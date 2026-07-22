from __future__ import annotations

from typing import Any

import svc_scaffold.openai_helpers as h

toolcalls_count = 0
BRANCHES = 3


class Breakpoints:
    def __init__(self):
        self.store: dict[str, Any] = {}

    @staticmethod
    def feature_name():
        return "FEATURE_EXAMPLE"

    def after_tool_call(self, payload):
        return payload

    def before_task_call(self, payload):
        if self.store.get("responses") is None:
            self.store["need_new_branch"] = True
            self.store["new_branch_payload"] = payload
            self.store["responses"] = []

        return payload

    def before_chat_message(self, payload):
        return payload

    def after_chat_message(self, response):
        if response is None:
            return response, None
        return h.set_message(
            response,
            h.append_content(h.message(response), "\nдумай экономичнее и качественнее, пожалуйста\n"),
        ), None

    def after_task_call(self, response):
        if response is not None:
            self.store["responses"].append(response)

        if len(self.store["responses"]) >= BRANCHES:
            self.store["need_new_branch"] = False

        if self.store["need_new_branch"]:
            return None, self.store["new_branch_payload"]
        else:
            response = self.store["responses"][0]  # имитация выбора лучшего ответа
            self.store = {}
            return response, None

    def before_tool_call(self, response):
        global toolcalls_count
        toolcalls_count += 1
        response.setdefault("scaffold", {})["toolcalls_count"] = toolcalls_count

        return response
