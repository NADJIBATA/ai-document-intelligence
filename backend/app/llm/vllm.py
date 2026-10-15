"""vLLM's OpenAI-compatible server (``vllm serve <model>``).

Same wire format as OpenAI, including ``response_format: json_schema`` for structured
outputs (guided decoding); only the name of the token limit differs.
"""

from __future__ import annotations

from app.llm.openai_compatible import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    token_limit_field = "max_tokens"  # noqa: S105 - request field name
    provider_name = "vllm"
