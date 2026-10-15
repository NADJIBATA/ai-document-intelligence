"""scripts/smoke_test.py, run against the application (fakes for the models)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scripts.smoke_test import main, run


def test_smoke_test_passes_against_the_application(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    # TestClient is an httpx.Client: the script runs unchanged (no nginx, no frontend).
    run(client, chat=True, frontend=False, timeout=10)
    output = capsys.readouterr().out
    assert "isolation: user B cannot see user A's document... ok" in output
    assert "cleanup... ok" in output


def test_unreachable_deployment_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--base-url", "http://127.0.0.1:9", "--timeout", "1"]) == 1
    assert "FAILED" in capsys.readouterr().out
