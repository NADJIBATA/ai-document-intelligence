from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from app.core import http as http_module
from app.rag.embeddings import (
    EmbeddingError,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    l2_normalize,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_module.time, "sleep", lambda seconds: None)


class EmbeddingServer:
    """Minimal OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self.status = 200
        self.reverse_order = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        payload = json.loads(request.content)
        self.requests.append(payload)
        self.headers.append(request.headers)
        if self.status != 200:
            return httpx.Response(self.status, text="boom")
        data = [
            {"index": i, "embedding": [float(len(text)), 1.0, 0.0, 0.0][: self.dimension]}
            for i, text in enumerate(payload["input"])
        ]
        if self.reverse_order:
            data.reverse()
        return httpx.Response(200, json={"data": data, "model": payload["model"]})


def make_embedder(server: EmbeddingServer, **kwargs: Any) -> OpenAICompatibleEmbedder:
    options: dict[str, Any] = {
        "base_url": "http://vllm-embedding:8000/v1/",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "dimension": 4,
        "client": httpx.Client(transport=httpx.MockTransport(server)),
    }
    options.update(kwargs)
    return OpenAICompatibleEmbedder(**options)


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_l2_normalize() -> None:
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_documents_are_embedded_in_batches_and_normalised() -> None:
    server = EmbeddingServer()
    embedder = make_embedder(server, batch_size=2)
    vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [len(request["input"]) for request in server.requests] == [2, 2, 1]
    assert server.requests[0] == {"model": "Qwen/Qwen3-Embedding-0.6B", "input": ["a", "bb"]}
    assert len(vectors) == 5
    assert all(norm(vector) == pytest.approx(1.0) for vector in vectors)
    assert embedder.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert embedder.dimension == 4


def test_vectors_follow_the_input_order_even_if_the_server_shuffles() -> None:
    server = EmbeddingServer()
    server.reverse_order = True
    vectors = make_embedder(server).embed_documents(["a", "bbbbbbbb"])
    assert vectors[0][0] < vectors[1][0]  # first component grows with the text length


def test_query_instruction_is_applied_to_queries_only() -> None:
    server = EmbeddingServer()
    embedder = make_embedder(server, query_instruction="Retrieve relevant passages")
    embedder.embed_query("notice period?")
    embedder.embed_documents(["The notice period is thirty days."])
    assert server.requests[0]["input"] == [
        "Instruct: Retrieve relevant passages\nQuery: notice period?"
    ]
    assert server.requests[1]["input"] == ["The notice period is thirty days."]


def test_no_instruction_by_default() -> None:
    server = EmbeddingServer()
    make_embedder(server).embed_query("notice period?")
    assert server.requests[0]["input"] == ["notice period?"]


def test_empty_input_does_not_call_the_server() -> None:
    server = EmbeddingServer()
    assert make_embedder(server).embed_documents([]) == []
    assert server.requests == []


def test_dimension_is_probed_when_not_configured() -> None:
    server = EmbeddingServer(dimension=3)
    embedder = make_embedder(server, dimension=None)
    assert embedder.dimension == 3
    assert embedder.dimension == 3
    assert len(server.requests) == 1  # probed once


def test_dimension_mismatch_is_an_error() -> None:
    server = EmbeddingServer(dimension=3)
    with pytest.raises(EmbeddingError, match="unexpected vector size"):
        make_embedder(server, dimension=4).embed_query("x")


def test_server_errors_become_embedding_errors() -> None:
    server = EmbeddingServer()
    server.status = 503
    with pytest.raises(EmbeddingError, match="HTTP 503"):
        make_embedder(server, max_retries=1).embed_query("x")
    assert len(server.requests) == 2  # one retry


def test_malformed_response() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"oops": []}))
    )
    with pytest.raises(EmbeddingError, match="malformed response"):
        make_embedder(EmbeddingServer(), client=client).embed_query("x")


def test_missing_vectors_are_detected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0, 0, 0]}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="expected 2 vectors"):
        make_embedder(EmbeddingServer(), client=client).embed_documents(["a", "b"])


def test_api_key_is_sent_as_a_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    server = EmbeddingServer()
    real_client = httpx.Client

    def client_with_mock_transport(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(server), **kwargs)

    monkeypatch.setattr(httpx, "Client", client_with_mock_transport)
    embedder = OpenAICompatibleEmbedder(
        base_url="https://api.openai.com/v1", model="text-embedding-3-small", api_key="k-test"
    )
    embedder.embed_query("x")
    assert server.headers[0]["authorization"] == "Bearer k-test"


# -------------------------------------------------------------- HashingEmbedder
def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dimension=64)
    first = embedder.embed_query("termination notice period")
    assert first == embedder.embed_documents(["termination notice period"])[0]
    assert len(first) == 64
    assert norm(first) == pytest.approx(1.0)
    assert embedder.model_name == "hashing-64"
    assert embedder.dimension == 64


def test_hashing_embedder_similarity_reflects_shared_words() -> None:
    embedder = HashingEmbedder(dimension=256)

    def cosine(a: str, b: str) -> float:
        return sum(
            x * y for x, y in zip(embedder.embed_query(a), embedder.embed_query(b), strict=True)
        )

    related = cosine("contract termination notice", "notice of termination of the contract")
    unrelated = cosine("contract termination notice", "quarterly revenue forecast")
    assert related > 0.8
    assert related > unrelated + 0.5


def test_hashing_embedder_handles_empty_text() -> None:
    assert HashingEmbedder(dimension=8).embed_query("") == [0.0] * 8
