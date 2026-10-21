"""Минимальный OpenAI-compatible HTTP-прокси для цепочки фич."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import time
import uuid
from typing import Any
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from features import common as h


class ModelClientHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ModelClient:
    name = "openai_compatible"

    def __init__(self) -> None:
        self.base_url = os.environ["BASE_MODEL_API_BASE_URL"].rstrip("/")
        self.api_key = "local"
        self.model = os.environ["BASE_MODEL_NAME"]
        self.backend = importlib.import_module(f"backends.{os.environ['MODEL_BACKEND']}")
        self.context_limit = int(os.environ["MODEL_CONTEXT_LIMIT"])
        self.avg_tokens_per_second = float(os.getenv("AVG_TOKENS_PER_SECOND", "30"))
        if self.context_limit < 1 or not math.isfinite(self.avg_tokens_per_second) or self.avg_tokens_per_second <= 0:
            raise ValueError("model context and average token speed must be positive")
        self.request_timeout = math.ceil(self.context_limit / self.avg_tokens_per_second)
        self.avg_stats = {"avg_output_tokens": 0, "avg_response_time": 0.0, "avg_usage_complete": True}

    def save_avg_stats(self) -> None:
        if directory := os.getenv("JOB_STATE_DIR"):
            path = Path(directory) / "avg.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.avg_stats), encoding="utf-8")
            temporary.replace(path)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def health(self) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/models", headers=self.headers)
        response.raise_for_status()

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload, model=self.model, stream=False)
        payload = self.backend.enrich_payload(payload)
        timeout = self.request_timeout
        avg_started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", headers=self.headers, json=payload
                )
        except httpx.TimeoutException as error:
            self.avg_stats["avg_usage_complete"] = False
            self.save_avg_stats()
            raise ModelClientHTTPError(504, f"Base model request timed out after {timeout:g} seconds") from error
        except httpx.RequestError as error:
            self.avg_stats["avg_usage_complete"] = False
            self.save_avg_stats()
            raise ModelClientHTTPError(502, f"Base model connection failed: {error}") from error

        # Записываем тело до разбора JSON и обработки фичами, без заголовков с секретами.
        if directory := os.getenv("JOB_STATE_DIR"):
            with (Path(directory) / "model-responses.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"timestamp": time.time(), "status": response.status_code,
                                      "request": payload, "response_body": response.text}, ensure_ascii=False) + "\n")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            self.avg_stats["avg_usage_complete"] = False
            self.save_avg_stats()
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            raise ModelClientHTTPError(response.status_code, detail[:4000]) from error
        result = response.json()
        usage = result.get("usage") or {}
        # Сохраняем пик входного контекста для промежуточного heartbeat job.
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        if isinstance(input_tokens, int):
            self.avg_stats["max_input_tokens"] = max(self.avg_stats.get("max_input_tokens", 0), input_tokens)
        # completion_tokens уже включает reasoning: отдельно его не прибавляем.
        avg_output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        self.avg_stats["avg_usage_complete"] &= isinstance(avg_output_tokens, int) and avg_output_tokens >= 0
        if isinstance(avg_output_tokens, int) and avg_output_tokens >= 0:
            self.avg_stats["avg_output_tokens"] += avg_output_tokens
        self.avg_stats["avg_response_time"] += time.monotonic() - avg_started
        self.save_avg_stats()
        return result


def avg_job_time_limit(handlers: list[Any], context_limit: int) -> int:
    avg_tps = float(os.getenv("AVG_TOKENS_PER_SECOND", "30"))
    avg_tool_count = float(os.getenv("AVG_TOOL_COUNT", "50"))
    avg_tool_time = float(os.getenv("AVG_TOOL_TIME", "30"))
    avg_factors = [float(getattr(handler, "avg_token_factor", lambda: 1)()) for handler in handlers]
    if not all(math.isfinite(value) and value > 0 for value in [avg_tps, *avg_factors]):
        raise ValueError("AVG_TOKENS_PER_SECOND and feature avg_token_factor must be finite and positive")
    if not all(math.isfinite(value) and value >= 0 for value in [avg_tool_count, avg_tool_time]):
        raise ValueError("AVG_TOOL_COUNT and AVG_TOOL_TIME must be finite and nonnegative")
    return math.ceil(context_limit * math.prod(avg_factors) / avg_tps + avg_tool_count * avg_tool_time)


def enabled_features(model_client: ModelClient | None = None) -> list[Any]:
    handlers = []
    for feature in filter(None, os.getenv("FEATURES", "").split(",")):
        module_name = f"features.{feature}"
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
        self.avg_job_time_limit = avg_job_time_limit(self.features, model_client.context_limit)
        model_client.request_timeout = self.avg_job_time_limit
        model_client.avg_stats["avg_job_time_limit"] = self.avg_job_time_limit
        model_client.save_avg_stats()

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



def create_app() -> FastAPI:
    model = os.environ["SCAFFOLD_MODEL_NAME"]
    model_client = ModelClient()
    scaffold = Scaffold(model_client)
    app = FastAPI(title="Benchmark Scaffold OpenAI-compatible Proxy")


    @app.get("/health")
    async def health() -> dict[str, Any]:
        await model_client.health()
        return {"ok": True, "adapter": "openai_compatible", "model_client": model_client.name}


    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "scaffold",
            }],
        }


    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        if payload.get("stream"):
            raise HTTPException(400, "Streaming is not supported")
        try:
            response = await scaffold.chat_completions(payload)
        except ModelClientHTTPError as error:
            raise HTTPException(error.status_code, error.detail) from error

        response = dict(response)
        response.update(id=f"scaffold-{uuid.uuid4().hex}", model=model)
        response["scaffold"] = dict(response.get("scaffold") or {})
        response["scaffold"].update(algorithm="features", model_client=model_client.name)
        return JSONResponse(response)
    return app
