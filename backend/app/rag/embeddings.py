"""Embedding providers.

* ``OpenAICompatibleEmbedder`` talks to any server exposing ``POST /v1/embeddings``
  (vLLM, OpenAI, TEI, Ollama...). Changing the model is a configuration change.
* ``HashingEmbedder`` is a deterministic, dependency-free embedder used by tests and
  offline evaluation (``EMBEDDING_PROVIDER=fake``).

Vectors are always L2-normalised, so cosine similarity equals the dot product.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

import httpx

from app.core.http import UpstreamError, client_timeout, post_json
from app.rag.analyzer import TextAnalyzer


class EmbeddingError(Exception):
    """The embedding service is unavailable or returned an invalid response."""


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


class OpenAICompatibleEmbedder:
    """Client for ``POST {base_url}/embeddings``.

    Instruction-aware models (Qwen3-Embedding, E5-instruct...) expect *queries* to be
    prefixed with a task description while *documents* are embedded as-is:

        Instruct: {query_instruction}\\nQuery: {question}
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dimension: int | None = None,
        batch_size: int = 32,
        timeout_seconds: float = 60.0,
        query_instruction: str = "",
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._query_instruction = query_instruction.strip()
        self._max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            timeout=client_timeout(timeout_seconds), headers=headers
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Ask the server once: the collection size must match the model output.
            self._dimension = len(self._embed(["dimension probe"])[0])
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed(list(texts[start : start + self._batch_size])))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if self._query_instruction:
            text = f"Instruct: {self._query_instruction}\nQuery: {text}"
        return self._embed([text])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            data = post_json(
                self._client,
                self._url,
                {"model": self._model, "input": inputs},
                max_retries=self._max_retries,
                service="embeddings",
            )
        except UpstreamError as exc:
            raise EmbeddingError(str(exc)) from exc

        try:
            items = sorted(data["data"], key=lambda item: item["index"])
            vectors = [l2_normalize([float(v) for v in item["embedding"]]) for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("embeddings: malformed response") from exc
        if len(vectors) != len(inputs):
            raise EmbeddingError(
                f"embeddings: expected {len(inputs)} vectors, received {len(vectors)}"
            )
        if self._dimension is not None and any(len(v) != self._dimension for v in vectors):
            raise EmbeddingError(
                f"embeddings: unexpected vector size (configured dimension: {self._dimension})"
            )
        return vectors


class HashingEmbedder:
    """Bag-of-words hashed into a fixed-size vector (feature hashing).

    Not a semantic model, but texts sharing words get similar vectors, which is enough
    to test the dense retrieval path deterministically and without a model server.
    """

    def __init__(self, dimension: int = 256, analyzer: TextAnalyzer | None = None) -> None:
        self._dimension = dimension
        self._analyzer = analyzer or TextAnalyzer()

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in self._analyzer.tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return l2_normalize(vector)
