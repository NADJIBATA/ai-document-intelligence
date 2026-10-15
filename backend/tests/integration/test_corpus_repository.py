"""PostgreSQL side of the retrieval engine: corpus loading, signature, hydration."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.repositories.corpus import PostgresChunkStore, PostgresCorpusSource
from tests.fixtures.corpus import secret_merger_memo, service_agreement, travel_policy
from tests.integration.conftest import signup, upload


def _user_id(db_engine: Engine, email: str) -> uuid.UUID:
    with db_engine.connect() as connection:
        value: uuid.UUID = connection.execute(
            text("SELECT id FROM users WHERE email = :e"), {"e": email}
        ).scalar_one()
    return value


@pytest.fixture
def world(client: TestClient, db_engine: Engine) -> dict[str, Any]:
    alice = signup(client, "alice@docintel.dev")
    bob = signup(client, "bob@docintel.dev")
    agreement = upload(client, alice, "service-agreement.txt", service_agreement()).json()["id"]
    memo = upload(client, bob, "merger-memo.txt", secret_merger_memo()).json()["id"]
    return {
        "alice": alice,
        "alice_id": _user_id(db_engine, "alice@docintel.dev"),
        "bob_id": _user_id(db_engine, "bob@docintel.dev"),
        "agreement": uuid.UUID(agreement),
        "memo": uuid.UUID(memo),
    }


def test_corpus_contains_only_the_users_indexed_chunks(
    world: dict[str, Any], session_factory: sessionmaker[Session], db_engine: Engine
) -> None:
    corpus = PostgresCorpusSource(session_factory)
    alice_entries = corpus.load(world["alice_id"])
    assert len(alice_entries) == 3
    assert {entry.document_id for entry in alice_entries} == {world["agreement"]}
    assert all("Falcon" not in entry.text for entry in alice_entries)
    assert {entry.document_id for entry in corpus.load(world["bob_id"])} == {world["memo"]}
    assert corpus.load(uuid.uuid4()) == []

    # A document that is not (or no longer) "indexed" is invisible to the search.
    with db_engine.begin() as connection:
        connection.execute(
            text("UPDATE documents SET status = 'processing' WHERE id = :id"),
            {"id": world["agreement"]},
        )
    assert corpus.load(world["alice_id"]) == []


def test_signature_changes_with_the_corpus(
    client: TestClient, world: dict[str, Any], session_factory: sessionmaker[Session]
) -> None:
    corpus = PostgresCorpusSource(session_factory)
    initial = corpus.signature(world["alice_id"])
    assert corpus.signature(world["alice_id"]) == initial  # stable
    bob_initial = corpus.signature(world["bob_id"])

    policy = upload(client, world["alice"], "travel-policy.txt", travel_policy()).json()["id"]
    after_upload = corpus.signature(world["alice_id"])
    assert after_upload != initial
    assert corpus.signature(world["bob_id"]) == bob_initial  # other users are unaffected

    client.delete(f"/api/documents/{policy}", headers=world["alice"])
    after_delete = corpus.signature(world["alice_id"])
    assert after_delete != after_upload

    client.delete(f"/api/documents/{world['agreement']}", headers=world["alice"])
    assert corpus.signature(world["alice_id"]) == "0:0:-"
    assert corpus.signature(uuid.uuid4()) == "0:0:-"


def test_chunk_store_only_returns_chunks_of_the_user(
    world: dict[str, Any], session_factory: sessionmaker[Session], db_engine: Engine
) -> None:
    with db_engine.connect() as connection:
        rows = connection.execute(text("SELECT id, owner_id FROM document_chunks")).all()
    alice_ids = [row.id for row in rows if row.owner_id == world["alice_id"]]
    bob_ids = [row.id for row in rows if row.owner_id == world["bob_id"]]
    store = PostgresChunkStore(session_factory)

    found = store.get_many(world["alice_id"], alice_ids + bob_ids + [uuid.uuid4()])
    assert set(found) == set(alice_ids)
    chunk = found[alice_ids[0]]
    assert chunk.filename == "service-agreement.txt"
    assert chunk.document_id == world["agreement"]
    assert chunk.text
    assert chunk.rerank_score is None

    assert store.get_many(world["alice_id"], bob_ids) == {}
    assert store.get_many(world["alice_id"], []) == {}
