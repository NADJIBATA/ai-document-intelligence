from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest

from app.core import http as http_module
from app.rag.reranker import (
    LexicalOverlapReranker,
    PassthroughReranker,
    RerankApiReranker,
    RerankerError,
)
from app.rag.types import RetrievedChunk


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_module.time, "sleep", lambda seconds: None)


def chunk(number: int, text: str, section: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.UUID(int=number),
        document_id=uuid.UUID(int=500),
        filename="contract.pdf",
        text=text,
        page=1,
        section=section,
        chunk_index=number,
    )


def numbers(chunks: list[RetrievedChunk]) -> list[int]:
    return [c.chunk_id.int for c in chunks]


def api_reranker(handler: Any, **kwargs: Any) -> RerankApiReranker:
    return RerankApiReranker(
        base_url="http://vllm-reranker:8000/",
        model="BAAI/bge-reranker-v2-m3",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_api_reranker_sorts_by_relevance_and_truncates() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rerank"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.97},
                    {"index": 0, "relevance_score": 0.40},
                    {"index": 1, "relevance_score": 0.02},
                ]
            },
        )

    chunks = [
        chunk(1, "payment terms"),
        chunk(2, "governing law"),
        chunk(3, "thirty days notice", section="Termination"),
    ]
    reranker = api_reranker(handler)
    ranked = reranker.rerank("notice period", chunks, top_k=2)

    assert numbers(ranked) == [3, 1]
    assert [c.rerank_score for c in ranked] == [0.97, 0.40]
    assert ranked[0].score == 0.97  # the reranker score takes precedence over RRF
    assert requests[0] == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "notice period",
        "documents": ["payment terms", "governing law", "Termination\nthirty days notice"],
    }
    assert reranker.model_name == "BAAI/bge-reranker-v2-m3"


def test_api_reranker_without_candidates_does_not_call_the_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not be called")

    assert api_reranker(handler).rerank("q", [], top_k=5) == []


def test_api_reranker_server_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="model loading")

    with pytest.raises(RerankerError, match="HTTP 503"):
        api_reranker(handler, max_retries=1).rerank("q", [chunk(1, "a")], top_k=5)
    assert calls == 2


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": []},
        {"results": [{"index": 0}]},
        {"results": [{"index": 7, "relevance_score": 0.5}]},
        {"results": [{"index": 0, "relevance_score": 0.5}]},  # second document missing
    ],
)
def test_api_reranker_malformed_response(body: dict[str, Any]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(RerankerError):
        api_reranker(handler).rerank("q", [chunk(1, "a"), chunk(2, "b")], top_k=5)


def test_passthrough_keeps_the_fusion_order() -> None:
    chunks = [chunk(1, "a"), chunk(2, "b"), chunk(3, "c")]
    ranked = PassthroughReranker().rerank("q", chunks, top_k=2)
    assert numbers(ranked) == [1, 2]
    assert all(c.rerank_score is None for c in ranked)


def test_lexical_overlap_reranker() -> None:
    chunks = [
        chunk(1, "The supplier delivers the goods."),
        chunk(2, "The notice period for termination is thirty days."),
        chunk(3, "A notice must be sent by registered mail."),
    ]
    ranked = LexicalOverlapReranker().rerank("termination notice period", chunks, top_k=3)
    assert numbers(ranked) == [2, 3, 1]
    assert ranked[0].rerank_score == pytest.approx(1.0)
    assert ranked[1].rerank_score == pytest.approx(1 / 3)
    assert ranked[2].rerank_score == 0.0


def test_equal_scores_keep_the_input_order() -> None:
    chunks = [chunk(1, "notice"), chunk(2, "notice"), chunk(3, "notice")]
    assert numbers(LexicalOverlapReranker().rerank("notice", chunks, top_k=3)) == [1, 2, 3]
