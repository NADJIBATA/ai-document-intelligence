"""Chat through the API: retrieval -> strict prompt -> answer with verified sources."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from app.llm.base import LLMError
from app.llm.fake import FakeLLMProvider
from app.rag.prompts import REFUSAL_MESSAGES, SYSTEM_PROMPT
from tests.fixtures.corpus import poisoned_policy, service_agreement, travel_policy
from tests.integration.conftest import signup, upload

AGREEMENT = "service-agreement.txt"
POLICY = "travel-policy.txt"


def ask(client: TestClient, headers: dict[str, str], question: str, **body: Any) -> Any:
    return client.post("/api/chat", headers=headers, json={"question": question, **body})


def indexed_user(client: TestClient, email: str = "alice@docintel.dev") -> dict[str, Any]:
    headers = signup(client, email)
    agreement = upload(client, headers, AGREEMENT, service_agreement()).json()["id"]
    policy = upload(client, headers, POLICY, travel_policy()).json()["id"]
    return {"headers": headers, "agreement": agreement, "policy": policy}


def _count(db_engine: Engine, table: str) -> int:
    with db_engine.connect() as connection:
        return int(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())  # noqa: S608


def test_answer_is_grounded_and_cites_its_source(client: TestClient, llm: FakeLLMProvider) -> None:
    user = indexed_user(client)
    response = ask(client, user["headers"], "What is the notice period to terminate?")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["answered"] is True
    assert "notice period of thirty days" in body["answer"]
    assert body["answer"].endswith("[S1]")
    assert len(body["sources"]) == 1
    source = body["sources"][0]
    assert source["ref"] == "S1"
    assert source["document"] == AGREEMENT
    assert source["document_id"] == user["agreement"]
    assert "notice period of thirty days" in source["excerpt"]
    assert source["score"] > 0
    assert body["metadata"]["model"] == "fake-llm"
    assert body["metadata"]["degraded"] == []
    assert body["metadata"]["total_ms"] >= body["metadata"]["retrieval_ms"]

    # The cited passage can be opened (click on a source in the UI).
    chunk = client.get(
        f"/api/documents/{source['document_id']}/chunks/{source['chunk_id']}",
        headers=user["headers"],
    )
    assert chunk.status_code == 200
    assert "notice period of thirty days" in chunk.json()["text"]

    # The model received the strict prompt and at most TOP_K passages.
    sent = llm.calls[0]
    assert sent[0].role == "system"
    assert sent[0].content == SYSTEM_PROMPT
    assert 1 <= sent[-1].content.count("<source id=") <= 5
    assert sent[-1].content.endswith(
        "<question>\nWhat is the notice period to terminate?\n</question>"
    )


def test_exchange_is_saved_in_a_new_conversation(client: TestClient, db_engine: Engine) -> None:
    user = indexed_user(client)
    body = ask(client, user["headers"], "What is the nightly hotel budget?").json()

    detail = client.get(f"/api/conversations/{body['conversation_id']}", headers=user["headers"])
    assert detail.status_code == 200
    conversation = detail.json()
    assert conversation["title"] == "What is the nightly hotel budget?"
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
    question, answer = conversation["messages"]
    assert question["content"] == "What is the nightly hotel budget?"
    assert question["sources"] == []
    assert question["answered"] is None
    assert answer["id"] == body["message_id"]
    assert answer["content"] == body["answer"]
    assert answer["sources"] == body["sources"]
    assert answer["answered"] is True
    assert question["created_at"] < answer["created_at"]

    with db_engine.connect() as connection:
        meta = connection.execute(
            text("SELECT metadata FROM messages WHERE id = :id"), {"id": body["message_id"]}
        ).scalar_one()
    assert meta["model"] == "fake-llm"
    assert meta["answered"] is True
    assert meta["citation_fallback"] is False


def test_follow_up_question_continues_the_conversation(
    client: TestClient, llm: FakeLLMProvider
) -> None:
    user = indexed_user(client)
    first = ask(client, user["headers"], "What is the notice period to terminate?").json()
    second = ask(
        client,
        user["headers"],
        "And when are invoices payable?",
        conversation_id=first["conversation_id"],
    ).json()

    assert second["conversation_id"] == first["conversation_id"]
    assert "sixty days" in second["answer"]

    # The previous exchange is sent as history, without its citation markers.
    sent = llm.calls[1]
    assert [message.role for message in sent] == ["system", "user", "assistant", "user"]
    assert sent[1].content == "What is the notice period to terminate?"
    assert "notice period of thirty days" in sent[2].content
    assert "[S1]" not in sent[2].content

    detail = client.get(
        f"/api/conversations/{first['conversation_id']}", headers=user["headers"]
    ).json()
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert detail["title"] == "What is the notice period to terminate?"


def test_history_is_limited_to_the_configured_turns(make_client: Any) -> None:
    llm = FakeLLMProvider()
    client: TestClient = make_client(llm_backend=llm, chat_history_turns=1)
    user = indexed_user(client)
    conversation_id = None
    for question in ("notice period?", "hotel budget?", "invoices payable?"):
        body = ask(client, user["headers"], question, conversation_id=conversation_id).json()
        conversation_id = body["conversation_id"]
    # System + one previous turn (2 messages) + the question.
    assert [message.role for message in llm.calls[2]] == ["system", "user", "assistant", "user"]
    assert llm.calls[2][1].content == "hotel budget?"


def test_question_outside_the_documents_is_refused(client: TestClient) -> None:
    user = indexed_user(client)
    body = ask(client, user["headers"], "Who won the football world cup?").json()
    assert body["answered"] is False
    assert body["answer"] == REFUSAL_MESSAGES["en"]
    assert body["sources"] == []

    french = ask(client, user["headers"], "Quel est le prix du pétrole brut ?").json()
    assert french["answered"] is False
    assert french["answer"] == REFUSAL_MESSAGES["fr"]


def test_without_any_document_the_model_is_not_called(
    client: TestClient, llm: FakeLLMProvider
) -> None:
    headers = signup(client, "empty@docintel.dev")
    body = ask(client, headers, "What is the notice period?").json()
    assert body["answered"] is False
    assert body["sources"] == []
    assert body["metadata"]["model"] is None
    assert body["metadata"]["llm_ms"] == 0
    assert llm.calls == []
    # The refused exchange is still part of the history.
    detail = client.get(f"/api/conversations/{body['conversation_id']}", headers=headers).json()
    assert detail["messages"][1]["answered"] is False


def test_answer_can_be_restricted_to_documents(client: TestClient, llm: FakeLLMProvider) -> None:
    user = indexed_user(client)
    body = ask(
        client,
        user["headers"],
        "What is the notice period to terminate?",
        document_ids=[user["policy"]],
    ).json()
    # The agreement holds the answer, but only the travel policy may be used.
    assert all(source["document_id"] == user["policy"] for source in body["sources"])
    assert "service-agreement.txt" not in llm.calls[0][-1].content


def test_sources_never_come_from_the_model_output(make_client: Any) -> None:
    llm = FakeLLMProvider(
        responses=["The notice is ninety days (secret.pdf, page 99) [S1][S9]. Also [S42]."]
    )
    client: TestClient = make_client(llm_backend=llm)
    user = indexed_user(client)
    body = ask(client, user["headers"], "What is the notice period to terminate?").json()

    assert [source["ref"] for source in body["sources"]] == ["S1"]
    assert body["sources"][0]["document"] == AGREEMENT  # not "secret.pdf"
    assert body["sources"][0]["page"] != 99
    assert "[S9]" not in body["answer"]
    assert "[S42]" not in body["answer"]


def test_model_outage_returns_503_and_saves_nothing(make_client: Any, db_engine: Engine) -> None:
    llm = FakeLLMProvider()
    client: TestClient = make_client(llm_backend=llm)
    user = indexed_user(client)
    llm.fail_with = LLMError("llm: HTTP 401", status_code=401)

    response = ask(client, user["headers"], "What is the notice period to terminate?")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"
    assert _count(db_engine, "conversations") == 0
    assert _count(db_engine, "messages") == 0

    llm.fail_with = None
    assert ask(client, user["headers"], "What is the notice period?").status_code == 200
    assert _count(db_engine, "messages") == 2


def test_conversation_deleted_while_the_answer_is_generated(
    make_client: Any, db_engine: Engine
) -> None:
    def delete_everything(messages: Any, schema: Any) -> str:
        with db_engine.begin() as connection:
            connection.execute(text("DELETE FROM conversations"))
        return "Thirty days [S1]."

    plain: TestClient = make_client()
    user = indexed_user(plain)
    conversation_id = ask(plain, user["headers"], "What is the notice period?").json()[
        "conversation_id"
    ]

    racing: TestClient = make_client(llm_backend=FakeLLMProvider(handler=delete_everything))
    response = ask(racing, user["headers"], "And the invoices?", conversation_id=conversation_id)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"
    assert _count(db_engine, "messages") == 0


def test_prompt_injection_in_a_document_is_neutralised(
    client: TestClient, llm: FakeLLMProvider, db_engine: Engine
) -> None:
    headers = signup(client, "alice@docintel.dev")
    document_id = upload(client, headers, "remote-policy.txt", poisoned_policy()).json()["id"]

    # Detection at ingestion: the hostile passage is flagged, the legitimate one is not.
    with db_engine.connect() as connection:
        flags = connection.execute(
            text(
                "SELECT flagged_injection FROM document_chunks "
                "WHERE document_id = :d ORDER BY chunk_index"
            ),
            {"d": document_id},
        ).scalars()
        assert list(flags) == [False, True]

    body = ask(client, headers, "How many days per week may employees work remotely?").json()
    assert body["answered"] is True
    assert "three days per week" in body["answer"]
    assert "PWNED" not in body["answer"]

    prompt = llm.calls[0][-1].content
    # The forged closing/opening tags are escaped: still one block per passage.
    assert prompt.count("<source id=") == prompt.count("</source>") == 2
    assert "&lt;/source&gt;&lt;source id=" in prompt
    assert prompt.count('flagged="true"') == 1
    # Instructions live in the system message only.
    assert llm.calls[0][0].content == SYSTEM_PROMPT


def test_unknown_conversation(client: TestClient) -> None:
    headers = signup(client, "alice@docintel.dev")
    response = ask(client, headers, "notice period?", conversation_id=str(uuid.uuid4()))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"


def test_chat_requires_authentication(client: TestClient) -> None:
    assert client.post("/api/chat", json={"question": "notice period?"}).status_code == 401
    assert client.get("/api/conversations").status_code == 401


def test_chat_validation(client: TestClient, make_client: Any) -> None:
    headers = signup(client, "alice@docintel.dev")
    for payload in (
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "ok", "conversation_id": "not-a-uuid"},
        {"question": "ok", "document_ids": ["not-a-uuid"]},
    ):
        response = client.post("/api/chat", headers=headers, json=payload)
        assert response.status_code == 422, payload

    strict: TestClient = make_client(max_question_chars=50)
    response = ask(strict, headers, "long question " * 10)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "question_too_long"


def test_chat_is_rate_limited_per_user(make_client: Any) -> None:
    limited: TestClient = make_client(rate_limit_chat="2/minute")
    alice = signup(limited, "alice@docintel.dev")
    bob = signup(limited, "bob@docintel.dev")
    assert ask(limited, alice, "notice?").status_code == 200
    assert ask(limited, alice, "notice?").status_code == 200
    assert ask(limited, alice, "notice?").status_code == 429
    assert ask(limited, bob, "notice?").status_code == 200


def test_chat_endpoints_are_documented(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"200", "400", "401", "404", "422", "429", "503"} <= set(
        paths["/api/chat"]["post"]["responses"]
    )
    assert "get" in paths["/api/conversations"]
    assert {"get", "delete"} <= set(paths["/api/conversations/{conversation_id}"])
