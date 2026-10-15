from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from tests.fixtures.corpus import contract_2025, contract_2026, travel_policy
from tests.integration.conftest import signup, upload


def stats(client: TestClient, headers: dict[str, str]) -> Any:
    response = client.get("/api/dashboard/stats", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_empty_dashboard(client: TestClient) -> None:
    headers = signup(client, "alice@docintel.dev")
    assert stats(client, headers) == {
        "documents": {"total": 0, "uploaded": 0, "processing": 0, "indexed": 0, "failed": 0},
        "conversations": 0,
        "comparisons": 0,
        "passages": 0,
        "storage_bytes": 0,
    }


def test_dashboard_counts_only_my_data(client: TestClient, db_engine: Engine) -> None:
    alice = signup(client, "alice@docintel.dev")
    bob = signup(client, "bob@docintel.dev")
    a = upload(client, alice, "contract_2025.txt", contract_2025()).json()["id"]
    b = upload(client, alice, "contract_2026.txt", contract_2026()).json()["id"]
    failed = upload(client, alice, "travel-policy.txt", travel_policy()).json()["id"]
    with db_engine.begin() as connection:
        connection.execute(
            text("UPDATE documents SET status = 'failed' WHERE id = :id"), {"id": failed}
        )
    client.post("/api/chat", headers=alice, json={"question": "What is the monthly fee?"})
    client.post(
        "/api/documents/compare", headers=alice, json={"document_a_id": a, "document_b_id": b}
    )
    upload(client, bob, "contract_2025.txt", contract_2025())

    assert stats(client, alice) == {
        "documents": {"total": 3, "uploaded": 0, "processing": 0, "indexed": 2, "failed": 1},
        "conversations": 1,
        "comparisons": 1,
        "passages": 7,  # 3 + 4: passages of the failed document are not counted
        "storage_bytes": len(contract_2025()) + len(contract_2026()) + len(travel_policy()),
    }

    bob_stats = stats(client, bob)
    assert bob_stats["documents"]["total"] == 1
    assert (bob_stats["conversations"], bob_stats["comparisons"]) == (0, 0)


def test_dashboard_requires_authentication(client: TestClient) -> None:
    assert client.get("/api/dashboard/stats").status_code == 401
