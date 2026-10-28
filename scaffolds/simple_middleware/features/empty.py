"""Контрольная техника: сохраняет запросы и ответы без изменений."""


class Breakpoints:
    @staticmethod
    def feature_name():
        return "empty"

    def before_task_call(self, payload):
        return payload

    def before_chat_message(self, payload):
        return payload

    def after_tool_call(self, payload):
        return payload

    def after_chat_message(self, response):
        return response, None

    def after_task_call(self, response):
        return response, None

    def before_tool_call(self, response):
        return response
