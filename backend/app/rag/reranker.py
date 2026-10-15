"""Neural reranking of the fused candidates.

``RerankApiReranker`` calls a cross-encoder (``BAAI/bge-reranker-v2-m3`` by default)
served behind the de-facto standard rerank API (``POST /v1/rerank``, as exposed by
vLLM, Jina, Cohere-compatible servers):

    request : {"model": ..., "query": ..., "documents": [...]}
    response: {"results": [{"index": 3, "relevance_score": 0.97}, ...]}
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from app.core.http import UpstreamError, client_timeout, post_json
from app.rag.analyzer import TextAnalyzer
from app.rag.types import RetrievedChunk


class RerankerError(Exception):
    """The reranking service is unavailable or returned an invalid response."""


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most relevant chunks, best first, with ``rerank_score`` set."""
        ...


def _passage(chunk: RetrievedChunk) -> str:
    """Text scored by the cross-encoder: the section title gives useful context."""
    return f"{chunk.section}\n{chunk.text}" if chunk.section else chunk.text


def _ranked(
    chunks: Sequence[RetrievedChunk], scores: Sequence[float], top_k: int
) -> list[RetrievedChunk]:
    for chunk, score in zip(chunks, scores, strict=True):
        chunk.rerank_score = float(score)
    # Stable sort: equal scores keep the fusion order.
    return sorted(chunks, key=lambda c: -(c.rerank_score or 0.0))[:top_k]


class RerankApiReranker:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/rerank"
        self._model = model
        self._max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            timeout=client_timeout(timeout_seconds), headers=headers
        )

    @property
    def model_name(self) -> str:
        return self._model

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        try:
            data = post_json(
                self._client,
                self._url,
                {
                    "model": self._model,
                    "query": query,
                    "documents": [_passage(chunk) for chunk in chunks],
                },
                max_retries=self._max_retries,
                service="reranker",
            )
        except UpstreamError as exc:
            raise RerankerError(str(exc)) from exc

        scores: list[float | None] = [None] * len(chunks)
        try:
            for item in data["results"]:
                scores[int(item["index"])] = float(item["relevance_score"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RerankerError("reranker: malformed response") from exc
        if any(score is None for score in scores):
            raise RerankerError("reranker: missing scores in response")
        return _ranked(chunks, [score for score in scores if score is not None], top_k)


class PassthroughReranker:
    """Keeps the fusion order (``RERANKER_PROVIDER=none``)."""

    model_name = "none"

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return list(chunks[:top_k])


class LexicalOverlapReranker:
    """Deterministic stand-in for a cross-encoder (``RERANKER_PROVIDER=fake``).

    Score = share of the query terms found in the passage, in [0, 1].
    """

    model_name = "lexical-overlap"

    def __init__(self, analyzer: TextAnalyzer | None = None) -> None:
        self._analyzer = analyzer or TextAnalyzer()

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        query_terms = set(self._analyzer.tokenize(query))
        scores = []
        for chunk in chunks:
            terms = set(self._analyzer.tokenize(_passage(chunk)))
            scores.append(len(query_terms & terms) / len(query_terms) if query_terms else 0.0)
        return _ranked(chunks, scores, top_k)
