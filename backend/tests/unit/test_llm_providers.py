from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.core import http as http_module
from app.core.config import Settings
from app.llm.base import ChatMessage, JsonSchema, LLMError
from app.llm.factory import build_llm
from app.llm.fake import FakeLLMProvider
from app.llm.ollama import OllamaProvider
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.vllm import VLLMProvider

MESSAGES = [ChatMessage("system", "Be brief."), ChatMessage("user", "Hello?")]
SCHEMA = JsonSchema(name="Answer", schema={"type": "object", "properties": {"a": {}}})


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_module.time, "sleep", lambda seconds: None)


class Server:
    """Records requests and replays the queued responses (the last one is repeated)."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    def payload(self, index: int = -1) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(self.requests[index].content)
        return body

    @property
    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def completion(content: str | None = "Hi!", **extra: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "gpt-4.1-mini-2025-04-14",
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            **extra,
        },
    )


def openai(server: Server, **kwargs: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1/", model="gpt-4.1-mini", client=server.client, **kwargs
    )


# ----------------------------------------------------------- OpenAI-compatible
def test_openai_request_and_response() -> None:
    server = Server(completion("  Hi!  "))
    provider = openai(server, temperature=0.1, max_tokens=512)
    response = provider.generate(MESSAGES)

    assert str(server.requests[0].url) == "https://api.openai.com/v1/chat/completions"
    assert server.payload() == {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello?"},
        ],
        "max_completion_tokens": 512,
        "temperature": 0.1,
    }
    assert response.content == "Hi!"
    assert response.model == "gpt-4.1-mini-2025-04-14"
    assert (response.prompt_tokens, response.completion_tokens) == (12, 3)
    assert response.finish_reason == "stop"
    assert (provider.name, provider.model_name) == ("openai", "gpt-4.1-mini")


def test_openai_per_call_overrides_and_json_schema() -> None:
    server = Server(completion('{"a": 1}'))
    openai(server).generate(MESSAGES, temperature=0.0, max_tokens=64, json_schema=SCHEMA)
    payload = server.payload()
    assert payload["temperature"] == 0.0
    assert payload["max_completion_tokens"] == 64
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Answer", "schema": SCHEMA.schema},
    }


def test_openai_api_key_is_sent_as_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    server = Server(completion())
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(server), **kwargs),
    )
    OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1", model="gpt-4.1-mini", api_key="k-test"
    ).generate(MESSAGES)
    assert server.requests[0].headers["authorization"] == "Bearer k-test"


def test_openai_drops_the_temperature_when_the_model_rejects_it() -> None:
    rejection = httpx.Response(
        400,
        json={
            "error": {
                "message": "Unsupported value: 'temperature' does not support 0.1 with this "
                "model. Only the default (1) value is supported.",
                "param": "temperature",
            }
        },
    )
    server = Server(rejection, completion("ok"))
    provider = openai(server)
    assert provider.generate(MESSAGES).content == "ok"
    assert "temperature" in server.payload(0)
    assert "temperature" not in server.payload(1)

    provider.generate(MESSAGES)  # remembered: no rejected call any more
    assert len(server.requests) == 3
    assert "temperature" not in server.payload(2)


def test_openai_other_client_errors_are_not_retried() -> None:
    server = Server(httpx.Response(400, json={"error": {"message": "model not found"}}))
    with pytest.raises(LLMError, match="HTTP 400") as error:
        openai(server).generate(MESSAGES)
    assert error.value.status_code == 400
    assert len(server.requests) == 1


def test_openai_authentication_error_does_not_leak_the_response_body() -> None:
    server = Server(httpx.Response(401, text="Incorrect API key provided: k-secret"))
    with pytest.raises(LLMError) as error:
        openai(server).generate(MESSAGES)
    assert str(error.value) == "llm: HTTP 401"


def test_openai_server_errors_are_retried() -> None:
    server = Server(httpx.Response(503, text="overloaded"), completion("after retry"))
    assert openai(server, max_retries=2).generate(MESSAGES).content == "after retry"
    assert len(server.requests) == 2


def test_openai_read_timeout_is_not_retried() -> None:
    server = Server(httpx.ReadTimeout("timed out"))
    with pytest.raises(LLMError, match="ReadTimeout"):
        openai(server, max_retries=3).generate(MESSAGES)
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        completion(None),
        completion("   "),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"unexpected": True}),
    ],
)
def test_openai_unusable_responses(response: httpx.Response) -> None:
    with pytest.raises(LLMError):
        openai(Server(response)).generate(MESSAGES)


def test_vllm_uses_max_tokens() -> None:
    server = Server(completion())
    provider = VLLMProvider(
        base_url="http://vllm-llm:8000/v1", model="Qwen/Qwen2.5-3B-Instruct", client=server.client
    )
    provider.generate(MESSAGES, max_tokens=128)
    payload = server.payload()
    assert payload["max_tokens"] == 128
    assert "max_completion_tokens" not in payload
    assert provider.name == "vllm"


# ---------------------------------------------------------------------- Ollama
def ollama(server: Server, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(
        base_url="http://ollama:11434/", model="qwen2.5:3b", client=server.client, **kwargs
    )


def test_ollama_request_and_response() -> None:
    server = Server(
        httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {"role": "assistant", "content": "Bonjour !"},
                "done_reason": "stop",
                "prompt_eval_count": 20,
                "eval_count": 4,
            },
        )
    )
    provider = ollama(server, temperature=0.2, max_tokens=256)
    response = provider.generate(MESSAGES, json_schema=SCHEMA)

    assert str(server.requests[0].url) == "http://ollama:11434/api/chat"
    assert server.payload() == {
        "model": "qwen2.5:3b",
        "messages": [message.as_dict() for message in MESSAGES],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 256},
        "format": SCHEMA.schema,
    }
    assert response.content == "Bonjour !"
    assert (response.prompt_tokens, response.completion_tokens) == (20, 4)
    assert (provider.name, provider.model_name) == ("ollama", "qwen2.5:3b")


def test_ollama_errors() -> None:
    with pytest.raises(LLMError, match="HTTP 404"):
        ollama(Server(httpx.Response(404, text="model not found"))).generate(MESSAGES)
    with pytest.raises(LLMError, match="empty response"):
        ollama(Server(httpx.Response(200, json={"message": {"content": ""}}))).generate(MESSAGES)
    with pytest.raises(LLMError, match="ConnectError"):
        ollama(Server(httpx.ConnectError("refused")), max_retries=0).generate(MESSAGES)


# ------------------------------------------------------------------------ fake
RAG_PROMPT = """<sources>
<source id="S1" document="a.pdf" page="1">
Invoices are payable within sixty days. Late fees apply.
</source>
<source id="S2" document="a.pdf" page="2">
The notice period is thirty days &amp; must be sent by mail. It starts on receipt.
</source>
</sources>

<question>
{question}
</question>"""


def ask_fake(question: str) -> str:
    prompt = RAG_PROMPT.format(question=question)
    return FakeLLMProvider().generate([ChatMessage("user", prompt)]).content


def test_fake_answers_from_the_most_relevant_source() -> None:
    assert ask_fake("What is the notice period?") == (
        "The notice period is thirty days & must be sent by mail. [S2]"
    )
    assert ask_fake("When are invoices payable?") == "Invoices are payable within sixty days. [S1]"


def test_fake_says_when_the_sources_do_not_answer() -> None:
    assert ask_fake("Who is the chief executive?") == "[NO_ANSWER]"
    assert FakeLLMProvider().generate([ChatMessage("user", "no sources")]).content == "[NO_ANSWER]"


def test_fake_scripted_responses_handler_and_failure() -> None:
    provider = FakeLLMProvider(responses=["first", "second"])
    assert provider.generate(MESSAGES).content == "first"
    assert provider.generate(MESSAGES).content == "second"
    assert provider.calls == [MESSAGES, MESSAGES]
    assert (provider.name, provider.model_name) == ("fake", "fake-llm")

    echo = FakeLLMProvider(handler=lambda messages, schema: messages[-1].content.upper())
    assert echo.generate(MESSAGES).content == "HELLO?"

    provider.fail_with = LLMError("llm: HTTP 503", status_code=503)
    with pytest.raises(LLMError, match="503"):
        provider.generate(MESSAGES)


# --------------------------------------------------------------------- factory
def test_factory(settings_factory: Callable[..., Settings]) -> None:
    assert isinstance(build_llm(settings_factory(llm_provider="fake")), FakeLLMProvider)

    openai_provider = build_llm(
        settings_factory(llm_provider="openai", llm_model="gpt-4.1-mini", llm_api_key="k-test")
    )
    assert type(openai_provider) is OpenAICompatibleProvider
    assert openai_provider.model_name == "gpt-4.1-mini"
    assert openai_provider._url == "https://api.openai.com/v1/chat/completions"
    assert openai_provider._client.headers["authorization"] == "Bearer k-test"

    vllm_provider = build_llm(
        settings_factory(llm_provider="vllm", llm_base_url="http://vllm-llm:8000/v1")
    )
    assert isinstance(vllm_provider, VLLMProvider)
    assert "authorization" not in vllm_provider._client.headers

    ollama_provider = build_llm(
        settings_factory(
            llm_provider="ollama", llm_base_url="http://ollama:11434", llm_model="qwen2.5:3b"
        )
    )
    assert isinstance(ollama_provider, OllamaProvider)
    assert ollama_provider._url == "http://ollama:11434/api/chat"
