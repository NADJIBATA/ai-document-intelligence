"""Provider for servers exposing ``POST {base_url}/chat/completions``:
OpenAI, Azure OpenAI, vLLM, Groq, Together, LM Studio, llama.cpp server..."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import structlog

from app.core.http import UpstreamError, client_timeout, post_json
from app.llm.base import ChatMessage, JsonSchema, LLMError, LLMResponse

logger = structlog.get_logger(__name__)


class OpenAICompatibleProvider:
    # Current OpenAI models only accept "max_completion_tokens"; subclasses targeting
    # servers that still expect the historical name override this.
    token_limit_field = "max_completion_tokens"  # noqa: S105 - request field name
    provider_name = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        # Some models (reasoning models) reject any non-default temperature: the
        # parameter is dropped after the first rejection.
        self._send_temperature = True
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            timeout=client_timeout(timeout_seconds), headers=headers
        )

    @property
    def name(self) -> str:
        return self.provider_name

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
            self.token_limit_field: max_tokens or self._max_tokens,
        }
        if self._send_temperature:
            payload["temperature"] = self._temperature if temperature is None else temperature
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema.name, "schema": json_schema.schema},
            }

        try:
            data = self._post(payload)
        except UpstreamError as exc:
            if not self._temperature_rejected(exc, payload):
                raise LLMError(str(exc), status_code=exc.status_code) from exc
            logger.warning("llm_temperature_not_supported", model=self._model)
            self._send_temperature = False
            payload.pop("temperature")
            try:
                data = self._post(payload)
            except UpstreamError as retry_exc:
                raise LLMError(str(retry_exc), status_code=retry_exc.status_code) from retry_exc
        return self._parse(data)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        return post_json(
            self._client,
            self._url,
            payload,
            max_retries=self._max_retries,
            service="llm",
            retry_read_timeout=False,
        )

    @staticmethod
    def _temperature_rejected(error: UpstreamError, payload: dict[str, Any]) -> bool:
        return error.status_code == 400 and "temperature" in payload and "temperature" in str(error)

    def _parse(self, data: dict[str, Any]) -> LLMResponse:
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content")
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError("llm: malformed response") from exc
        if not isinstance(content, str) or not content.strip():
            # Typically a refusal, or a reasoning model that spent the whole token budget.
            raise LLMError(f"llm: empty response (finish_reason={finish_reason})")
        raw_usage = data.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return LLMResponse(
            content=content.strip(),
            model=str(data.get("model") or self._model),
            prompt_tokens=_as_int(usage.get("prompt_tokens")),
            completion_tokens=_as_int(usage.get("completion_tokens")),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
