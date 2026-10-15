"""Provider for Ollama's native API (``POST {base_url}/api/chat``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from app.core.http import UpstreamError, client_timeout, post_json
from app.llm.base import ChatMessage, JsonSchema, LLMError, LLMResponse


class OllamaProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or httpx.Client(timeout=client_timeout(timeout_seconds))

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: JsonSchema | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.as_dict() for message in messages],
            "stream": False,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": max_tokens or self._max_tokens,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema.schema
        try:
            data = post_json(
                self._client,
                self._url,
                payload,
                max_retries=self._max_retries,
                service="llm",
                retry_read_timeout=False,
            )
        except UpstreamError as exc:
            raise LLMError(str(exc), status_code=exc.status_code) from exc

        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMError("llm: empty response")
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        done_reason = data.get("done_reason")
        return LLMResponse(
            content=content.strip(),
            model=str(data.get("model") or self._model),
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            finish_reason=done_reason if isinstance(done_reason, str) else None,
        )
