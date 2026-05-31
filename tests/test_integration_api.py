"""Real end-to-end API tests that make ACTUAL model + tool calls.

These are gated and skipped by default — they cost tokens. To run them:

    # PowerShell
    $env:DS_RUN_INTEGRATION="1"; .venv/Scripts/python.exe -m pytest tests/test_integration_api.py -v

    # bash
    DS_RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration_api.py -v

Requires OPENAI_API_KEY (loaded from .env) and network access (web_search tool).
They exercise the path the mock suite cannot: HTTP -> real LLM planner ->
validate -> compile -> execute (real tools + model) -> record -> serve back.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings

_ENABLED = os.environ.get("DS_RUN_INTEGRATION") == "1" and bool(
    os.environ.get("OPENAI_API_KEY")
)

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="set DS_RUN_INTEGRATION=1 (and OPENAI_API_KEY) to run real-call e2e",
)

MODEL = os.environ.get("DS_MODEL", "gpt-5.4-nano")


def _client(tmp_path: Path) -> TestClient:
    settings = ApiSettings.from_env(
        {"DS_RUNS_DIR": str(tmp_path), "DS_MODEL_ALLOWLIST": MODEL}
    )
    return TestClient(create_app(settings))


def _openai(prompt: str, **extra) -> dict:
    body = {"prompt": prompt, "planner": "openai", "model": MODEL}
    body.update(extra)
    return body


def _poll_until_terminal(client: TestClient, run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = client.get(f"/runs/{run_id}").json()
        if last.get("state") in {"ok", "failed", "paused"}:
            return last
        time.sleep(1.0)
    return last


def test_real_sync_run_uses_real_model(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/runs",
        json=_openai("Compare SQLite and DuckDB for local analytics; recommend one.", mode="sync"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["spec"]["nodes"]
    # The real model must have produced output — never the mock echo.
    import json as _json

    assert "<mock-llm>" not in _json.dumps(body["values"])


def test_real_async_run_completes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/runs",
        json=_openai("Summarize two vector-index approaches and recommend one.", mode="async", run_id="it-async"),
    )
    assert resp.status_code == 202
    final = _poll_until_terminal(client, "it-async")
    assert final["state"] == "ok"


def test_real_resume_finishes_with_real_model(tmp_path: Path) -> None:
    """Regression guard for the resume config-preservation bug.

    A run created with planner=openai that pauses MUST be finished by the real
    model, not the server-default mock runner.
    """
    client = _client(tmp_path)
    created = client.post(
        "/runs",
        json=_openai(
            "Draft a short product announcement. Then wait for human approval "
            "before finalizing. After approval, produce the final announcement.",
            mode="sync",
            run_id="it-resume",
        ),
    )
    assert created.status_code == 200
    if created.json()["status"] != "paused":
        pytest.skip("planner did not emit a wait_for_event graph this run")

    resumed = client.post(
        "/runs/it-resume/resume",
        json={"event": {"event_type": "human_approval", "approved": True}},
    )
    assert resumed.status_code == 200
    body = resumed.json()
    assert body["status"] == "ok"

    import json as _json

    # The whole point: resume must NOT fall back to the mock runner.
    assert "<mock-llm>" not in _json.dumps(body["values"])


def test_real_chain_runs(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/chains",
        json=_openai(
            "Investigate whether Postgres or MySQL fits a small SaaS better.",
            mode="sync",
            run_id="it-chain",
            max_iterations=2,
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "stopped", "max_iterations"}
    assert len(body["steps"]) >= 1
