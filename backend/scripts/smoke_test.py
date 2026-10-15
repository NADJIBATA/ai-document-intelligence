"""End-to-end smoke test of a running deployment, through nginx like a browser.

Checks: liveness and readiness, registration and login, upload, asynchronous indexing,
hybrid search, isolation between two users, (optionally) a cited LLM answer, deletion.
Creates two throw-away accounts (smoke-<random>@example.com).

    docker compose exec backend python -m scripts.smoke_test --base-url http://nginx
    docker compose exec backend python -m scripts.smoke_test --base-url http://nginx --chat

Exit code 0 when every step passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from typing import Any

import httpx

DOCUMENT = (
    "Smoke test agreement.\n\n"
    "The supplier delivers the platform within forty-five days. Either party may "
    "terminate the agreement with a written notice period of ninety days. The annual "
    "subscription fee is twelve thousand euros, invoiced in January.\n"
)
QUESTION = "What is the notice period to terminate the agreement?"


class SmokeTestError(Exception):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestError(message)


def _step(name: str) -> None:
    sys.stdout.write(f"- {name}... ")
    sys.stdout.flush()


def _ok(detail: str = "") -> None:
    sys.stdout.write(f"ok{f' ({detail})' if detail else ''}\n")


def _json(response: httpx.Response, expected: int) -> Any:
    if response.status_code != expected:
        raise SmokeTestError(
            f"{response.request.method} {response.request.url.path}: HTTP "
            f"{response.status_code} (expected {expected}): {response.text[:300]}"
        )
    return response.json() if response.content else None


def _signup(client: httpx.Client) -> dict[str, str]:
    email = f"smoke-{secrets.token_hex(6)}@example.com"
    # Random password with at least one letter and one digit (password policy).
    password = f"Smoke{secrets.token_hex(12)}9"
    _json(client.post("/api/auth/register", json={"email": email, "password": password}), 201)
    tokens = _json(client.post("/api/auth/login", json={"email": email, "password": password}), 200)
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def run(client: httpx.Client, *, chat: bool, frontend: bool, timeout: float) -> None:
    _step("liveness /health")
    _json(client.get("/health"), 200)
    _ok()

    _step("readiness /ready")
    ready = client.get("/ready")
    body = ready.json()
    failing = {k: v for k, v in body.get("checks", {}).items() if v.get("status") != "ok"}
    _check(ready.status_code == 200, f"not ready: {failing}")
    _ok(", ".join(body["checks"]))

    if frontend:
        _step("frontend /")
        page = client.get("/")
        _check(page.status_code == 200 and '<div id="root">' in page.text, "no application")
        _ok()

    _step("registration and login (user A, user B)")
    alice = _signup(client)
    bob = _signup(client)
    _ok()

    _step("upload")
    document = _json(
        client.post(
            "/api/documents",
            headers=alice,
            files={"file": ("smoke-test.txt", DOCUMENT.encode(), "text/plain")},
        ),
        202,
    )
    document_id = document["id"]
    _ok(document_id)

    try:
        _step("asynchronous indexing (Celery worker, embeddings, Qdrant)")
        deadline = time.monotonic() + timeout
        status = document["status"]
        while status not in ("indexed", "failed") and time.monotonic() < deadline:
            time.sleep(2)
            state = _json(client.get(f"/api/documents/{document_id}/status", headers=alice), 200)
            status = state["status"]
        _check(status == "indexed", f"document status '{status}' after {timeout:.0f} s")
        _ok()

        _step("hybrid search")
        search = _json(client.post("/api/search", headers=alice, json={"query": QUESTION}), 200)
        _check(bool(search["results"]), "no result")
        _check(search["results"][0]["document_id"] == document_id, "unexpected first result")
        degraded = search["degraded"]
        _ok(f"degraded stages: {', '.join(degraded)}" if degraded else "all stages")

        _step("isolation: user B cannot see user A's document")
        _json(client.get(f"/api/documents/{document_id}", headers=bob), 404)
        other = _json(client.post("/api/search", headers=bob, json={"query": QUESTION}), 200)
        _check(other["results"] == [], "user B received a passage of user A")
        _ok()

        if chat:
            _step("cited answer (LLM)")
            answer = _json(
                client.post("/api/chat", headers=alice, json={"question": QUESTION}), 200
            )
            _check(answer["answered"], f"no answer: {answer['answer'][:200]}")
            _check(bool(answer["sources"]), "answer without sources")
            _ok(answer["metadata"]["model"] or "")
    finally:
        _step("cleanup")
        _json(client.delete(f"/api/documents/{document_id}", headers=alice), 204)
        _ok()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test of a running deployment.")
    parser.add_argument("--base-url", default="http://nginx", help="default: http://nginx")
    parser.add_argument("--chat", action="store_true", help="also ask the LLM (paid API call)")
    parser.add_argument("--timeout", type=float, default=300.0, help="indexing timeout (s)")
    parser.add_argument("--no-frontend", action="store_true", help="API only (no nginx)")
    args = parser.parse_args(argv)
    sys.stdout.write(f"Smoke test of {args.base_url}\n")
    try:
        with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
            run(client, chat=args.chat, frontend=not args.no_frontend, timeout=args.timeout)
    except (SmokeTestError, httpx.HTTPError) as exc:
        sys.stdout.write(f"FAILED\n  {exc}\n")
        return 1
    sys.stdout.write("All checks passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
