from __future__ import annotations

import importlib
import hashlib
import inspect
import os
import pkgutil
from typing import Any, Protocol

import svc_scaffold.breakpoints as breakpoint_modules
import svc_scaffold.openai_helpers as h


class ModelClient(Protocol):
    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class Middleware(Protocol):
    @staticmethod
    def feature_name() -> str:
        ...

    def after_tool_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def before_task_call(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        ...

    def before_chat_message(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        ...

    def after_chat_message(self, response: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        ...

    def after_task_call(self, response: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        ...

    def before_tool_call(self, response: dict[str, Any]) -> dict[str, Any]:
        ...


enabled_breakpoints_handlers: dict[str, list[Middleware]] = {}


def enabled_breakpoints(key: str, model_client: ModelClient | None = None) -> list[Middleware]:
    if key not in enabled_breakpoints_handlers:
        enabled_breakpoints_handlers[key] = []
        for module_info in pkgutil.iter_modules(breakpoint_modules.__path__, breakpoint_modules.__name__ + "."):
            breakpoint_class = importlib.import_module(module_info.name).Breakpoints
            if os.getenv(breakpoint_class.feature_name()) == "1":
                try:
                    enabled_breakpoints_handlers[key].append(breakpoint_class(model_client))
                except TypeError:
                    enabled_breakpoints_handlers[key].append(breakpoint_class())

    return enabled_breakpoints_handlers[key]


def task_key(payload: dict[str, Any]) -> str:
    for item in h.messages(payload):
        if isinstance(item, dict) and item.get("role") == "user":
            return hashlib.md5(str(item.get("content") or "").encode()).hexdigest()
    return "default"


class Scaffold:
    def __init__(self, model_client: ModelClient) -> None:
        self.model_client = model_client
        self.enabled_breakpoints = enabled_breakpoints

    async def _maybe_await(self, result):
        if inspect.isawaitable(result):
            return await result
        return result

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = task_key(payload)

        if h.has_tool_result(payload):
            for middleware in self.enabled_breakpoints(key, self.model_client):
                result = middleware.after_tool_call(payload)
                payload = await self._maybe_await(result)

        if h.is_task_start(payload):
            for middleware in self.enabled_breakpoints(key, self.model_client):
                result = middleware.before_task_call(payload)
                payload = await self._maybe_await(result)
                if payload is None:
                    break

        if payload is not None:
            for middleware in self.enabled_breakpoints(key, self.model_client):
                result = middleware.before_chat_message(payload)
                payload = await self._maybe_await(result)
                if payload is None:
                    break

        response = None if payload is None else await self.model_client.chat_completions(payload)

        new_payload = None
        for middleware in self.enabled_breakpoints(key, self.model_client):
            result = middleware.after_chat_message(response)
            response, maybe_payload = await self._maybe_await(result)
            if maybe_payload is not None:
                new_payload = maybe_payload

        if response is None or h.is_visible_task_response(response):
            for middleware in self.enabled_breakpoints(key, self.model_client):
                result = middleware.after_task_call(response)
                response, maybe_payload = await self._maybe_await(result)
                if maybe_payload is not None:
                    new_payload = maybe_payload

        if new_payload is not None:
            response = await self.chat_completions(new_payload)

        if response is not None and h.has_tool_call(response):
            for middleware in self.enabled_breakpoints(key, self.model_client):
                before_tool_call_fn = middleware.before_tool_call
                try:
                    result = before_tool_call_fn(response, payload)
                except TypeError:
                    result = before_tool_call_fn(response)
                result = await self._maybe_await(result)
                if isinstance(result, tuple) and len(result) == 2:
                    response, new_payload = result
                else:
                    response = result
            if new_payload is not None:
                response = await self.chat_completions(new_payload)

        if new_payload is None and h.is_visible_task_response(response):
            enabled_breakpoints_handlers.pop(key, None)

        return response
