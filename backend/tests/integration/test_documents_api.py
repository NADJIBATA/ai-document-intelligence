from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from tests.fakes import InMemoryStorage, RecordingDispatcher
from tests.fixtures.builders import build_docx, contract_pdf, long_text, simple_pdf
from tests.integration.conftest import signup, upload


class TestUpload:
    def test_pdf_is_stored_processed_and_indexed(
        self, client: TestClient, storage: InMemoryStorage, db_engine: Engine
    ) -> None:
        headers = signup(client, "alice@docintel.dev")
        response = upload(client, headers, "contract.pdf", contract_pdf())
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["filename"] == "contract.pdf"
        assert body["content_type"] == "application/pdf"
        assert body["status"] == "uploaded"

        status = client.get(f"/api/documents/{body['id']}/status", headers=headers).json()
        assert status["status"] == "indexed"
        assert status["page_count"] == 3
        assert status["chunk_count"] >= 3
        assert status["error_message"] is None

        # The original file is in object storage under a server-generated key.
        (key,) = storage.objects
        assert key.endswith(f"/documents/{body['id']}/original.pdf")
        assert "contract" not in key

        with db_engine.connect() as connection:
            rows = connection.execute(
                text("SELECT page, section FROM document_chunks ORDER BY chunk_index")
            ).all()
        assert (3, "Termination") in [tuple(row) for row in rows]

    def test_docx_and_txt(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        docx = upload(
            client,
            headers,
            "policy.docx",
            build_docx([("Heading 1", "Scope"), ("Normal", "Text.")]),
        )
        txt = upload(client, headers, "notes.txt", long_text(5).encode())
        assert docx.status_code == txt.status_code == 202
        for response in (docx, txt):
            document = client.get(f"/api/documents/{response.json()['id']}", headers=headers)
            assert document.json()["status"] == "indexed"
            assert document.json()["page_count"] is None

    def test_file_name_is_sanitized(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        response = upload(client, headers, "../../secret/<evil>.pdf", simple_pdf("x"))
        assert response.status_code == 202
        assert response.json()["filename"] == "evil_.pdf"

    def test_duplicate_upload_is_rejected(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        assert upload(client, headers, "a.pdf", simple_pdf("same")).status_code == 202
        duplicate = upload(client, headers, "copy.pdf", simple_pdf("same"))
        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "duplicate_document"

    def test_same_file_for_two_users_is_allowed(self, client: TestClient) -> None:
        alice = signup(client, "alice@docintel.dev")
        bob = signup(client, "bob@docintel.dev")
        assert upload(client, alice, "a.pdf", simple_pdf("shared")).status_code == 202
        assert upload(client, bob, "a.pdf", simple_pdf("shared")).status_code == 202

    def test_too_large_file(self, make_client: Any) -> None:
        client: TestClient = make_client(max_upload_size_mb=1)
        headers = signup(client, "alice@docintel.dev")
        response = upload(client, headers, "big.txt", b"a" * (1024 * 1024 + 1))
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    def test_unsupported_extension(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        response = upload(client, headers, "image.png", b"\x89PNG\r\n\x1a\n")
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_extension"

    def test_spoofed_extension(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        response = upload(client, headers, "invoice.pdf", b"MZ\x90\x00" + b"\x00" * 200)
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "content_mismatch"

    def test_empty_file(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        response = upload(client, headers, "empty.txt", b"")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "empty_file"

    def test_missing_file_field(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        assert client.post("/api/documents", headers=headers).status_code == 422

    def test_upload_requires_authentication(self, client: TestClient) -> None:
        response = client.post("/api/documents", files={"file": ("a.pdf", simple_pdf("x"))})
        assert response.status_code == 401

    def test_unprocessable_document_ends_in_failed_status(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        response = upload(client, headers, "broken.pdf", b"%PDF-1.4\nnot a real pdf")
        assert response.status_code == 202
        status = client.get(f"/api/documents/{response.json()['id']}/status", headers=headers)
        assert status.json()["status"] == "failed"
        assert "corrupted" in status.json()["error_message"]

    def test_storage_unavailable(self, client: TestClient, storage: InMemoryStorage) -> None:
        headers = signup(client, "alice@docintel.dev")
        storage.available = False
        response = upload(client, headers, "a.pdf", simple_pdf("x"))
        assert response.status_code == 503
        assert client.get("/api/documents", headers=headers).json()["total"] == 0

    def test_queue_unavailable_marks_document_failed(self, make_client: Any) -> None:
        dispatcher = RecordingDispatcher()
        dispatcher.fail = True
        client: TestClient = make_client(task_dispatcher=dispatcher)
        headers = signup(client, "alice@docintel.dev")
        response = upload(client, headers, "a.pdf", simple_pdf("x"))
        assert response.status_code == 503
        (document,) = client.get("/api/documents", headers=headers).json()["items"]
        assert document["status"] == "failed"
        assert "could not be scheduled" in document["error_message"]

    def test_upload_is_rate_limited_per_user(self, make_client: Any) -> None:
        client: TestClient = make_client(rate_limit_upload="2/hour")
        headers = signup(client, "alice@docintel.dev")
        assert upload(client, headers, "1.txt", b"first text").status_code == 202
        assert upload(client, headers, "2.txt", b"second text").status_code == 202
        assert upload(client, headers, "3.txt", b"third text").status_code == 429
        # Another user is not affected (limit keyed by user, not by IP).
        other = signup(client, "bob@docintel.dev")
        assert upload(client, other, "1.txt", b"first text").status_code == 202


class TestReadAndList:
    def test_list_is_paginated_and_filterable(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        for i in range(3):
            upload(client, headers, f"doc{i}.txt", f"Document number {i}.".encode())
        upload(client, headers, "broken.pdf", b"%PDF-1.4 broken")

        page = client.get("/api/documents?limit=2&offset=0", headers=headers).json()
        assert page["total"] == 4
        assert len(page["items"]) == 2
        assert page["items"][0]["filename"] == "broken.pdf"  # most recent first

        failed = client.get("/api/documents?status=failed", headers=headers).json()
        assert [d["filename"] for d in failed["items"]] == ["broken.pdf"]

        assert client.get("/api/documents?limit=500", headers=headers).status_code == 422
        assert client.get("/api/documents?status=unknown", headers=headers).status_code == 422

    def test_get_unknown_document(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        response = client.get(
            "/api/documents/00000000-0000-0000-0000-000000000000", headers=headers
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "document_not_found"
        assert client.get("/api/documents/not-a-uuid", headers=headers).status_code == 422

    def test_download_original_file(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        content = contract_pdf()
        document_id = upload(client, headers, "Contrat été.pdf", content).json()["id"]
        response = client.get(f"/api/documents/{document_id}/file", headers=headers)
        assert response.status_code == 200
        assert response.content == content
        assert response.headers["content-type"] == "application/pdf"
        assert response.headers["content-disposition"].startswith("inline;")
        assert (
            "filename*=UTF-8''Contrat%20%C3%A9t%C3%A9.pdf"
            in response.headers["content-disposition"]
        )
        assert response.headers["content-security-policy"] == "sandbox"

    def test_download_when_file_vanished(
        self, client: TestClient, storage: InMemoryStorage
    ) -> None:
        headers = signup(client, "alice@docintel.dev")
        document_id = upload(client, headers, "a.pdf", simple_pdf("x")).json()["id"]
        storage.objects.clear()
        response = client.get(f"/api/documents/{document_id}/file", headers=headers)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "file_missing"

    def test_get_chunk(self, client: TestClient, db_engine: Engine) -> None:
        headers = signup(client, "alice@docintel.dev")
        document_id = upload(client, headers, "contract.pdf", contract_pdf()).json()["id"]
        with db_engine.connect() as connection:
            chunk_id = connection.execute(
                text("SELECT id FROM document_chunks WHERE page = 3")
            ).scalar_one()
        response = client.get(f"/api/documents/{document_id}/chunks/{chunk_id}", headers=headers)
        assert response.status_code == 200
        chunk = response.json()
        assert chunk["page"] == 3
        assert chunk["section"] == "Termination"
        assert "sixty (60) days" in chunk["text"]


class TestDeleteAndReprocess:
    def test_delete_removes_rows_and_file(
        self, client: TestClient, storage: InMemoryStorage, db_engine: Engine
    ) -> None:
        headers = signup(client, "alice@docintel.dev")
        document_id = upload(client, headers, "contract.pdf", contract_pdf()).json()["id"]
        assert storage.objects

        assert client.delete(f"/api/documents/{document_id}", headers=headers).status_code == 204
        assert client.get(f"/api/documents/{document_id}", headers=headers).status_code == 404
        assert client.delete(f"/api/documents/{document_id}", headers=headers).status_code == 404
        assert storage.objects == {}
        with db_engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM document_chunks")).scalar() == 0

    def test_delete_falls_back_to_inline_cleanup_when_queue_is_down(
        self, make_client: Any, storage: InMemoryStorage
    ) -> None:
        dispatcher = RecordingDispatcher()
        client: TestClient = make_client(task_dispatcher=dispatcher)
        headers = signup(client, "alice@docintel.dev")
        document_id = upload(client, headers, "a.pdf", simple_pdf("x")).json()["id"]
        dispatcher.fail = True
        assert client.delete(f"/api/documents/{document_id}", headers=headers).status_code == 204
        assert storage.objects == {}

    def test_reprocess_requires_failed_status(self, client: TestClient) -> None:
        headers = signup(client, "alice@docintel.dev")
        document_id = upload(client, headers, "a.pdf", contract_pdf()).json()["id"]
        conflict = client.post(f"/api/documents/{document_id}/reprocess", headers=headers)
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "invalid_document_status"

    def test_reprocess_after_failure(
        self, client: TestClient, storage: InMemoryStorage, db_engine: Engine
    ) -> None:
        headers = signup(client, "alice@docintel.dev")
        document_id = upload(client, headers, "a.pdf", contract_pdf()).json()["id"]
        with db_engine.begin() as connection:
            connection.execute(
                text("UPDATE documents SET status = 'failed', error_message = 'boom'")
            )
        response = client.post(f"/api/documents/{document_id}/reprocess", headers=headers)
        assert response.status_code == 202
        status = client.get(f"/api/documents/{document_id}/status", headers=headers).json()
        assert status["status"] == "indexed"
        assert status["error_message"] is None
