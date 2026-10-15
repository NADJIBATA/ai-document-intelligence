from __future__ import annotations

from collections.abc import Callable

import pytest

from app.core.config import Settings
from app.rag.embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from app.rag.factory import build_embedder, build_reranker, build_vector_store
from app.rag.reranker import LexicalOverlapReranker, PassthroughReranker, RerankApiReranker
from app.rag.vector_store import QdrantVectorStore


def test_fake_providers(settings_factory: Callable[..., Settings]) -> None:
    settings = settings_factory(
        embedding_provider="fake", embedding_dimension=64, reranker_provider="fake"
    )
    embedder = build_embedder(settings)
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dimension == 64
    assert isinstance(build_reranker(settings), LexicalOverlapReranker)


def test_vllm_embedder_uses_the_query_instruction(
    settings_factory: Callable[..., Settings],
) -> None:
    settings = settings_factory(
        embedding_provider="vllm",
        embedding_base_url="http://vllm-embedding:8000/v1",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_dimension=1024,
    )
    embedder = build_embedder(settings)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder.model_name == "Qwen/Qwen3-Embedding-0.6B"
    assert embedder.dimension == 1024  # configured: no call to the server
    assert embedder._query_instruction == settings.embedding_query_instruction
    assert embedder._url == "http://vllm-embedding:8000/v1/embeddings"


def test_openai_embedder_has_no_query_instruction(
    settings_factory: Callable[..., Settings],
) -> None:
    settings = settings_factory(
        embedding_provider="openai",
        embedding_base_url="https://api.openai.com/v1",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        embedding_api_key="k-test",
    )
    embedder = build_embedder(settings)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder._query_instruction == ""
    assert embedder._client.headers["authorization"] == "Bearer k-test"


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("none", PassthroughReranker), ("fake", LexicalOverlapReranker), ("vllm", RerankApiReranker)],
)
def test_reranker_providers(
    settings_factory: Callable[..., Settings], provider: str, expected: type
) -> None:
    settings = settings_factory(reranker_provider=provider)
    assert isinstance(build_reranker(settings), expected)


def test_collection_depends_on_the_embedding_model(
    settings_factory: Callable[..., Settings],
) -> None:
    settings = settings_factory(
        embedding_provider="vllm",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_dimension=1024,
        qdrant_collection_prefix="chunks",
    )
    store = build_vector_store(settings, build_embedder(settings))
    assert isinstance(store, QdrantVectorStore)
    assert store.collection == "chunks__qwen-qwen3-embedding-0-6b__1024"
