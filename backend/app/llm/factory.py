"""Builds the configured LLM provider. Switching provider or model is configuration only."""

from __future__ import annotations

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.fake import FakeLLMProvider
from app.llm.ollama import OllamaProvider
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.vllm import VLLMProvider


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "fake":
        return FakeLLMProvider()
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    provider_class = VLLMProvider if settings.llm_provider == "vllm" else OpenAICompatibleProvider
    api_key = settings.llm_api_key
    return provider_class(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=api_key.get_secret_value() if api_key else None,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
