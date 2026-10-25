"""Локальный прокси к внешнему OpenAI-compatible API."""

import os
from typing import Any


def enrich_payload(payload: dict[str, Any]) -> dict[str, Any]:
    budget = os.environ["BACKEND_EXTERNAL_THINKING_BUDGET_LIMIT"]
    if budget and int(budget) > 0:
        # AITunnel преобразует reasoning.max_tokens во внутренний thinking_budget модели.
        payload["reasoning"] = {"max_tokens": int(budget)}
    return payload


def run(port: int) -> None:
    # Внешний секрет и адрес проверяются только движком, который их использует.
    for name in ("BACKEND_EXTERNAL_API_KEY", "BACKEND_EXTERNAL_API_URL"):
        if not os.environ[name]:
            raise ValueError(f"{name} must not be empty")
    import httpx
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI()
    upstream = os.environ["BACKEND_EXTERNAL_API_URL"].rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ['BACKEND_EXTERNAL_API_KEY']}"}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        if request.headers.get("authorization") != "Bearer local":
            return Response(status_code=401)
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.request(
                    request.method, f"{upstream}/{path}", params=request.query_params,
                    content=await request.body(), headers=headers | {"Content-Type": "application/json"},
                )
            return Response(response.content, status_code=response.status_code,
                            media_type=response.headers.get("content-type"))
        except httpx.RequestError:
            return Response('External model connection failed', status_code=502)

    uvicorn.run(app, host="127.0.0.1", port=port, ws="none")
