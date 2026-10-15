"""User A cannot access documents belonging to User B.

Every document route is exercised with a foreign document id: the answer must be a
404 (existence is not even revealed) and nothing may be modified.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from tests.fakes import InMemoryStorage
from tests.fixtures.builders import contract_pdf
from tests.integration.conftest import signup, upload


@pytest.fixture
def two_users(client: TestClient, db_engine: Engine) -> dict[str, Any]:
    alice = signup(client, "alice@docintel.dev")
    bob = signup(client, "bob@docintel.dev")
    document_id = upload(client, alice, "alice-contract.pdf", contract_pdf()).json()["id"]
    with db_engine.connect() as connection:
        chunk_id = connection.execute(
            text("SELECT id FROM document_chunks WHERE document_id = :d LIMIT 1"),
            {"d": document_id},
        ).scalar_one()
    return {"alice": alice, "bob": bob, "document_id": document_id, "chunk_id": str(chunk_id)}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/documents/{document_id}"),
        ("GET", "/api/documents/{document_id}/status"),
        ("GET", "/api/documents/{document_id}/file"),
        ("GET", "/api/documents/{document_id}/chunks/{chunk_id}"),
        ("POST", "/api/documents/{document_id}/reprocess"),
        ("DELETE", "/api/documents/{document_id}"),
    ],
)
def test_user_a_cannot_access_documents_of_user_b(
    client: TestClient, two_users: dict[str, Any], method: str, path: str
) -> None:
    url = path.format(**two_users)
    response = client.request(method, url, headers=two_users["bob"])
    assert response.status_code == 404
    assert "alice" not in response.text.lower()

    # Alice still has full access: nothing was modified by Bob's attempt.
    own = client.get(
        f"/api/documents/{two_users['document_id']}/status", headers=two_users["alice"]
    )
    assert own.status_code == 200
    assert own.json()["status"] == "indexed"


def test_listing_only_returns_own_documents(client: TestClient, two_users: dict[str, Any]) -> None:
    bob_list = client.get("/api/documents", headers=two_users["bob"]).json()
    assert bob_list == {"items": [], "total": 0, "limit": 20, "offset": 0}
    alice_list = client.get("/api/documents", headers=two_users["alice"]).json()
    assert alice_list["total"] == 1


def test_foreign_file_is_untouched_after_delete_attempt(
    client: TestClient, two_users: dict[str, Any], storage: InMemoryStorage
) -> None:
    client.delete(f"/api/documents/{two_users['document_id']}", headers=two_users["bob"])
    assert len(storage.objects) == 1
    status = client.get(
        f"/api/documents/{two_users['document_id']}/status", headers=two_users["alice"]
    )
    assert status.json()["status"] == "indexed"


def test_chunk_of_another_document_is_not_reachable_through_own_document(
    client: TestClient, two_users: dict[str, Any]
) -> None:
    """Bob uses his own document id with Alice's chunk id."""
    own = upload(client, two_users["bob"], "bob.txt", b"Bob's own notes about lunch.")
    bob_document_id = own.json()["id"]
    response = client.get(
        f"/api/documents/{bob_document_id}/chunks/{two_users['chunk_id']}",
        headers=two_users["bob"],
    )
    assert response.status_code == 404
