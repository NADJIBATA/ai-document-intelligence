"""Conversation history and its isolation between users."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app.llm.fake import FakeLLMProvider
from tests.fixtures.corpus import secret_merger_memo, service_agreement
from tests.integration.conftest import signup, upload


def ask(client: TestClient, headers: dict[str, str], question: str, **body: Any) -> Any:
    response = client.post("/api/chat", headers=headers, json={"question": question, **body})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def alice(client: TestClient) -> dict[str, Any]:
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "service-agreement.txt", service_agreement()).json()["id"]
    return {"headers": headers, "document_id": document_id}


# ------------------------------------------------------------------- history
def test_list_conversations_most_recently_active_first(
    client: TestClient, alice: dict[str, Any]
) -> None:
    headers = alice["headers"]
    first = ask(client, headers, "What is the notice period?")["conversation_id"]
    second = ask(client, headers, "When are invoices payable?")["conversation_id"]

    listing = client.get("/api/conversations", headers=headers).json()
    assert listing["total"] == 2
    assert [item["id"] for item in listing["items"]] == [second, first]
    assert listing["items"][0]["title"] == "When are invoices payable?"
    assert listing["items"][0]["message_count"] == 2

    # A new message brings the older conversation back to the top.
    ask(client, headers, "And the late payment penalties?", conversation_id=first)
    listing = client.get("/api/conversations", headers=headers).json()
    assert [item["id"] for item in listing["items"]] == [first, second]
    assert listing["items"][0]["message_count"] == 4
    assert listing["items"][0]["updated_at"] > listing["items"][0]["created_at"]


def test_pagination(client: TestClient, alice: dict[str, Any]) -> None:
    for number in range(3):
        ask(client, alice["headers"], f"Question number {number} about the notice period?")
    page = client.get("/api/conversations?limit=2&offset=0", headers=alice["headers"]).json()
    assert (page["total"], page["limit"], page["offset"], len(page["items"])) == (3, 2, 0, 2)
    rest = client.get("/api/conversations?limit=2&offset=2", headers=alice["headers"]).json()
    assert len(rest["items"]) == 1
    assert client.get("/api/conversations?limit=0", headers=alice["headers"]).status_code == 422


def test_long_first_question_gives_a_short_title(client: TestClient, alice: dict[str, Any]) -> None:
    question = "What is the notice period " + "and the payment terms " * 10 + "?"
    conversation_id = ask(client, alice["headers"], question)["conversation_id"]
    detail = client.get(f"/api/conversations/{conversation_id}", headers=alice["headers"]).json()
    assert len(detail["title"]) <= 80
    assert detail["title"].endswith("…")
    assert detail["messages"][0]["content"] == question


def test_sources_remain_readable_after_the_document_is_deleted(
    client: TestClient, alice: dict[str, Any]
) -> None:
    headers = alice["headers"]
    body = ask(client, headers, "What is the notice period to terminate?")
    assert body["sources"]
    assert client.delete(f"/api/documents/{alice['document_id']}", headers=headers).status_code == (
        204
    )
    detail = client.get(f"/api/conversations/{body['conversation_id']}", headers=headers).json()
    assert detail["messages"][1]["sources"] == body["sources"]  # snapshot kept
    # ...but the passage itself is gone, and so is the knowledge.
    source = body["sources"][0]
    gone = client.get(
        f"/api/documents/{source['document_id']}/chunks/{source['chunk_id']}", headers=headers
    )
    assert gone.status_code == 404
    again = ask(client, headers, "What is the notice period to terminate?")
    assert again["answered"] is False


def test_delete_conversation(client: TestClient, alice: dict[str, Any], db_engine: Engine) -> None:
    headers = alice["headers"]
    conversation_id = ask(client, headers, "What is the notice period?")["conversation_id"]
    assert client.delete(f"/api/conversations/{conversation_id}", headers=headers).status_code == (
        204
    )
    assert client.get(f"/api/conversations/{conversation_id}", headers=headers).status_code == 404
    assert client.delete(f"/api/conversations/{conversation_id}", headers=headers).status_code == (
        404
    )
    assert client.get("/api/conversations", headers=headers).json()["total"] == 0
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM messages")).scalar_one() == 0


def test_unknown_conversation(client: TestClient, alice: dict[str, Any]) -> None:
    response = client.get(f"/api/conversations/{uuid.uuid4()}", headers=alice["headers"])
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"
    assert client.get("/api/conversations/not-a-uuid", headers=alice["headers"]).status_code == 422


# ----------------------------------------------------------------- isolation
@pytest.fixture
def two_users(client: TestClient, alice: dict[str, Any]) -> dict[str, Any]:
    bob = signup(client, "bob@docintel.dev")
    bob_document = upload(client, bob, "merger-memo.txt", secret_merger_memo()).json()["id"]
    alice_conversation = ask(client, alice["headers"], "What is the notice period to terminate?")
    bob_conversation = ask(client, bob, "What did the board approve for Project Zebra?")
    return {
        "alice": alice["headers"],
        "bob": bob,
        "alice_document": alice["document_id"],
        "bob_document": bob_document,
        "alice_conversation": alice_conversation["conversation_id"],
        "bob_conversation": bob_conversation["conversation_id"],
    }


def test_a_user_only_lists_his_own_conversations(
    client: TestClient, two_users: dict[str, Any]
) -> None:
    alice_list = client.get("/api/conversations", headers=two_users["alice"]).json()
    bob_list = client.get("/api/conversations", headers=two_users["bob"]).json()
    assert [item["id"] for item in alice_list["items"]] == [two_users["alice_conversation"]]
    assert [item["id"] for item in bob_list["items"]] == [two_users["bob_conversation"]]


def test_a_user_cannot_read_continue_or_delete_another_users_conversation(
    client: TestClient, two_users: dict[str, Any], db_engine: Engine
) -> None:
    target = two_users["bob_conversation"]
    alice = two_users["alice"]

    read = client.get(f"/api/conversations/{target}", headers=alice)
    assert read.status_code == 404
    assert read.json()["error"]["code"] == "conversation_not_found"
    assert "Zebra" not in read.text

    follow_up = client.post(
        "/api/chat",
        headers=alice,
        json={"question": "And what was the price?", "conversation_id": target},
    )
    assert follow_up.status_code == 404

    assert client.delete(f"/api/conversations/{target}", headers=alice).status_code == 404

    # Nothing changed for Bob.
    detail = client.get(f"/api/conversations/{target}", headers=two_users["bob"])
    assert detail.status_code == 200
    assert len(detail.json()["messages"]) == 2
    with db_engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM conversations")).scalar_one() == 2


def test_chat_never_sends_another_users_passages_to_the_model(
    client: TestClient, two_users: dict[str, Any], llm: FakeLLMProvider
) -> None:
    llm.calls.clear()
    for question in (
        "What did the board approve for Project Zebra and Falcon Industries?",
        "What is the notice period to terminate the acquisition agreement?",
    ):
        response = client.post(
            "/api/chat",
            headers=two_users["alice"],
            json={"question": question, "document_ids": [two_users["bob_document"]]},
        )
        body = response.json()
        assert body["answered"] is False  # Bob's document is invisible, even when named
        assert body["sources"] == []

        body = client.post(
            "/api/chat", headers=two_users["alice"], json={"question": question}
        ).json()
        assert all(
            source["document_id"] == two_users["alice_document"] for source in body["sources"]
        )

    assert llm.calls, "the model was called for the unrestricted questions"
    for call in llm.calls:
        # Everything before <question> is the retrieved context (the question itself
        # legitimately contains the words Alice typed).
        context = call[-1].content.split("<question>")[0]
        assert "<source id=" in context
        assert "Falcon" not in context
        assert "ninety million" not in context
        assert "merger-memo.txt" not in context
