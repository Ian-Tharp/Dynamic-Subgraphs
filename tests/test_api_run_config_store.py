"""Deterministic (token-free) guards for run-config persistence.

These prove the resume/replay config-preservation fix without any model calls:
the persisted api_config.json is written on create and *read* by resume/replay
(so a run created with one config is never finished with the server default).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.run_config_store import load_run_config, save_run_config
from app.api.settings import ApiSettings


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    save_run_config(
        tmp_path,
        planner="llm",
        provider="openai",
        model="gpt-5.4-nano",
    )
    assert load_run_config(tmp_path) == {
        "planner": "llm",
        "provider": "openai",
        "model": "gpt-5.4-nano",
    }


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_run_config(tmp_path) is None


def test_load_old_config_defaults_provider_to_openai(tmp_path: Path) -> None:
    (tmp_path / "api_config.json").write_text(
        json.dumps({"planner": "openai", "model": "gpt-5.4-nano"}),
        encoding="utf-8",
    )

    assert load_run_config(tmp_path) == {
        "planner": "openai",
        "provider": "openai",
        "model": "gpt-5.4-nano",
    }


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)})))


def test_create_run_persists_resolved_config(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "run_id": "cfg-1", "mode": "sync"})

    config = load_run_config(tmp_path / "cfg-1")
    assert config == {
        "planner": "mock",
        "provider": "openai",
        "model": "gpt-5.4-nano",
    }


def test_resume_reads_persisted_config_not_server_default(tmp_path: Path) -> None:
    # Seed a real recorded run (mock, free), then poison its persisted config
    # with a model that is NOT in the allowlist. If resume reads the persisted
    # config (fixed behavior), resolution fails with 400 before any execution.
    # If it used the server default (the bug), it would sail past resolution.
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "run_id": "cfg-2", "mode": "sync"})
    save_run_config(tmp_path / "cfg-2", planner="mock", model="not-allowlisted-model")

    resp = client.post("/runs/cfg-2/resume", json={"event": {"ok": True}})

    assert resp.status_code == 400
    assert "not-allowlisted-model" in json.dumps(resp.json())


def test_replay_reads_persisted_config_not_server_default(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "run_id": "cfg-3", "mode": "sync"})
    save_run_config(tmp_path / "cfg-3", planner="mock", model="not-allowlisted-model")

    resp = client.post("/runs/cfg-3/replay", json={})

    assert resp.status_code == 400
    assert "not-allowlisted-model" in json.dumps(resp.json())
