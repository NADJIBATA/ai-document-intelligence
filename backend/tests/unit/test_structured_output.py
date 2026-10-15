from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from app.llm.base import ChatMessage, LLMError
from app.llm.fake import FakeLLMProvider
from app.llm.structured import StructuredOutputError, extract_json, generate_structured

MESSAGES = [ChatMessage("user", "Extract the parties.")]


class Party(BaseModel):
    name: str
    role: str | None = None


class Extraction(BaseModel):
    parties: list[Party] = Field(default_factory=list)
    amount: float | None = None


def test_extract_json_tolerates_fences_and_prose() -> None:
    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Here it is:\n{"a": {"b": 2}}\nDone.') == '{"a": {"b": 2}}'
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("no json here")


def test_valid_output_is_parsed_and_the_schema_is_sent_to_the_provider() -> None:
    seen = []

    def handler(messages, schema):  # type: ignore[no-untyped-def]
        seen.append(schema)
        return '{"parties": [{"name": "ACME", "role": "supplier"}], "amount": 1200.5}'

    result = generate_structured(FakeLLMProvider(handler=handler), MESSAGES, Extraction)
    assert result == Extraction(parties=[Party(name="ACME", role="supplier")], amount=1200.5)
    assert seen[0].name == "Extraction"
    assert seen[0].schema == Extraction.model_json_schema()


def test_invalid_output_gets_one_corrective_attempt() -> None:
    provider = FakeLLMProvider(
        responses=['{"parties": [{"role": "supplier"}]}', '```json\n{"parties": []}\n```']
    )
    result = generate_structured(provider, MESSAGES, Extraction)
    assert result == Extraction()

    assert len(provider.calls) == 2
    correction = provider.calls[1]
    assert [message.role for message in correction] == ["user", "assistant", "user"]
    assert correction[1].content == '{"parties": [{"role": "supplier"}]}'
    assert "parties.0.name" in correction[2].content  # the validation error is explained


def test_non_json_output_is_also_corrected() -> None:
    provider = FakeLLMProvider(responses=["I cannot do that.", '{"amount": 3}'])
    assert generate_structured(provider, MESSAGES, Extraction).amount == 3


def test_gives_up_after_the_allowed_corrections() -> None:
    provider = FakeLLMProvider(responses=["nope", "still nope", "never"])
    with pytest.raises(StructuredOutputError, match="Extraction"):
        generate_structured(provider, MESSAGES, Extraction, max_corrections=1)
    assert len(provider.calls) == 2

    assert issubclass(StructuredOutputError, LLMError)


def test_provider_errors_propagate() -> None:
    provider = FakeLLMProvider()
    provider.fail_with = LLMError("llm: HTTP 503")
    with pytest.raises(LLMError, match="503"):
        generate_structured(provider, MESSAGES, Extraction)
