"""Live evaluation mode: references resolved against one user's indexed documents only."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from app.core.config import Settings
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import InMemoryVectorStore
from evaluation.dataset import DatasetError
from evaluation.systems import SYSTEMS, run_system
from tests.fixtures.corpus import service_agreement, travel_policy
from tests.integration.conftest import signup, upload


def test_live_context_evaluates_only_the_documents_of_the_user(
    client: TestClient,
    integration_settings: Settings,
    embedder: EmbeddingProvider,
    vector_store: InMemoryVectorStore,
) -> None:
    alice = signup(client, "alice@docintel.dev")
    bob = signup(client, "bob@docintel.dev")
    upload(client, alice, "service-agreement.txt", service_agreement())
    upload(client, bob, "travel-policy.txt", travel_policy())

    from evaluation.live import LiveContext

    context = LiveContext(
        integration_settings,
        "ALICE@docintel.dev",
        top_k=5,
        embedder=embedder,
        vector_store=vector_store,
    )
    try:
        alice_chunk = context.resolve("service-agreement.txt#0")
        assert context.resolve(alice_chunk) == alice_chunk
        # Bob's document does not exist for Alice.
        with pytest.raises(DatasetError, match="does not match"):
            context.resolve("travel-policy.txt#0")
        with pytest.raises(DatasetError, match="invalid chunk reference"):
            context.resolve("service-agreement.txt")

        for system in SYSTEMS:
            result = run_system(
                context.retrieval.pipeline,
                context.user_id,
                system,
                "termination notice, travel and hotel expenses",
                5,
            )
            assert result.degraded == []
            # The query matches both users; only Alice's passages can come back.
            assert result.chunk_ids
            for chunk_id in result.chunk_ids:
                assert context.resolve(chunk_id) == chunk_id
            owned = {context.resolve(f"service-agreement.txt#{i}") for i in range(3)}
            assert set(result.chunk_ids) <= owned
    finally:
        context.close()


def test_live_context_rejects_an_unknown_user(
    integration_settings: Settings,
    db_engine: Engine,  # skips the test (instead of failing) when PostgreSQL is down
) -> None:
    from evaluation.live import LiveContext

    with pytest.raises(ValueError, match="no user"):
        LiveContext(integration_settings, "nobody@docintel.dev", top_k=5)
