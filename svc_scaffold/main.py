"""Минимальный OpenAI-compatible HTTP-прокси для цепочки фич."""

from __future__ import annotations

import math
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from svc_scaffold.core import Scaffold


class ModelClientHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ModelClient:
    name = "openai_compatible"

    def __init__(self) -> None:
        self.base_url = os.environ["BASE_MODEL_API_BASE_URL"].rstrip("/")
        self.api_key = os.environ["BASE_MODEL_API_KEY"]
        self.model = os.environ["BASE_MODEL_NAME"]
        self.context_limit = int(os.environ["MODEL_CONTEXT_LIMIT"])
        self.average_tokens_per_second = float(os.getenv("MODEL_AVERAGE_TOKENS_PER_SECOND", "20"))
        if self.context_limit < 1 or self.average_tokens_per_second <= 0:
            raise ValueError("model context and average token speed must be positive")
        self.request_timeout = math.ceil(self.context_limit / self.average_tokens_per_second)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def health(self) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/models", headers=self.headers)
        response.raise_for_status()

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload, model=self.model, stream=False)
        requested_max = payload.get("max_tokens")
        payload["max_tokens"] = (
            min(requested_max, self.context_limit)
            if isinstance(requested_max, int)
            else self.context_limit
        )
        timeout = self.request_timeout
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", headers=self.headers, json=payload
                )
        except httpx.TimeoutException as error:
            raise ModelClientHTTPError(504, f"Base model request timed out after {timeout:g} seconds") from error
        except httpx.RequestError as error:
            raise ModelClientHTTPError(502, f"Base model connection failed: {error}") from error

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            raise ModelClientHTTPError(response.status_code, detail[:4000]) from error
        return response.json()


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
            "owned_by": "svc_scaffold",
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
