from __future__ import annotations

import os
from typing import Any

import httpx


class ModelClientHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class OpenAICompatibleModelClient:
    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def health(self) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/models", headers=self.headers())
        response.raise_for_status()

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        request["model"] = self.model
        request["stream"] = False

        timeout = float(os.getenv("BASE_MODEL_REQUEST_TIMEOUT", "1800"))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers(),
                    json=request,
                )
        except httpx.TimeoutException as exc:
            raise ModelClientHTTPError(
                504,
                f"Base model request timed out after {timeout:g} seconds",
            ) from exc
        except httpx.RequestError as exc:
            raise ModelClientHTTPError(502, f"Base model connection failed: {exc}") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            raise ModelClientHTTPError(response.status_code, detail[:4000]) from exc
        return response.json()
