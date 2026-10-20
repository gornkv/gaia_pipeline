from __future__ import annotations

import importlib
import inspect
import os
from typing import Any, Protocol

from svc_scaffold.features import common as h


class ModelClient(Protocol):
    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


def enabled_features(model_client: ModelClient | None = None) -> list[Any]:
    handlers = []
    for feature in filter(None, os.getenv("FEATURES", "").split(",")):
        module_name = f"svc_scaffold.features.{feature}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            raise ValueError(f"Unknown feature: {feature}") from error
        try:
            feature_class = module.Breakpoints
        except AttributeError as error:
            raise ValueError(f"Feature module {feature} has no Breakpoints class") from error
        if feature_class.feature_name() != feature:
            raise ValueError(f"Feature module {feature} declares name {feature_class.feature_name()}")
        try:
            handlers.append(feature_class(model_client))
        except TypeError:
            handlers.append(feature_class())
    return handlers


class Scaffold:
    def __init__(self, model_client: ModelClient) -> None:
        self.model_client = model_client
        # Runner создаёт отдельный scaffold-процесс на job, поэтому фичи живут до завершения процесса.
        self.features = enabled_features(model_client)

    async def _maybe_await(self, result):
        if inspect.isawaitable(result):
            return await result
        return result

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_start = h.is_task_start(payload)
        handlers = self.features

        if h.has_tool_result(payload):
            for middleware in handlers:
                result = middleware.after_tool_call(payload)
                payload = await self._maybe_await(result)

        if task_start:
            for middleware in handlers:
                result = middleware.before_task_call(payload)
                payload = await self._maybe_await(result)
                if payload is None:
                    break

        if payload is not None:
            for middleware in handlers:
                result = middleware.before_chat_message(payload)
                payload = await self._maybe_await(result)
                if payload is None:
                    break

        response = None if payload is None else await self.model_client.chat_completions(payload)

        new_payload = None
        for middleware in handlers:
            result = middleware.after_chat_message(response)
            response, maybe_payload = await self._maybe_await(result)
            if maybe_payload is not None:
                new_payload = maybe_payload

        if response is None or h.is_visible_task_response(response):
            for middleware in handlers:
                result = middleware.after_task_call(response)
                response, maybe_payload = await self._maybe_await(result)
                if maybe_payload is not None:
                    new_payload = maybe_payload

        if new_payload is not None:
            response = await self.chat_completions(new_payload)

        if response is not None and h.has_tool_call(response):
            for middleware in handlers:
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

        return response
