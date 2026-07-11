from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from svc_scaffold.clients.openai_compatible import (
    ModelClientHTTPError,
    OpenAICompatibleModelClient,
)
from svc_scaffold.core import Scaffold


def openai_response(
    response: dict[str, Any],
    model: str,
    scaffold: Scaffold,
    model_client: OpenAICompatibleModelClient,
) -> dict[str, Any]:
    result = dict(response)
    result["id"] = f"scaffold-{uuid.uuid4().hex}"
    result["model"] = model
    if not isinstance(result.get("scaffold"), dict):
        result["scaffold"] = {}
    result["scaffold"]["algorithm"] = "breakpoints"
    result["scaffold"]["model_client"] = model_client.name

    return result


def create_app(scaffold: Scaffold, model_client: OpenAICompatibleModelClient) -> FastAPI:
    model = os.environ["SCAFFOLD_MODEL_NAME"]
    app = FastAPI(title="GAIA Scaffold OpenAI-compatible Proxy")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        await model_client.health()
        return {"ok": True, "adapter": "openai_compatible", "model_client": model_client.name}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "svc_scaffold",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming is not supported by the BoN proxy yet")

        try:
            response = await scaffold.chat_completions(payload)
        except ModelClientHTTPError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        return JSONResponse(openai_response(response, model, scaffold, model_client))

    return app
