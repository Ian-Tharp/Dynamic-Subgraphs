# tests/test_api_health_registry.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings


def _client(tmp_path) -> TestClient:
    settings = ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)})
    return TestClient(create_app(settings))


def test_healthz_ok(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_registry_lists_kinds_and_allowlists(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.get("/registry")
    assert resp.status_code == 200
    body = resp.json()
    kinds = {k["kind"] for k in body["node_kinds"]}
    assert "llm_call" in kinds and "wait_for_event" in kinds
    assert "web_search" in body["tools"]
    assert "critic" in body["subagents"]
    assert "python_eval" in body["forbidden_kinds"]
    sample = next(k for k in body["node_kinds"] if k["kind"] == "llm_call")
    assert "param_schema" in sample
    assert sample["param_schema"]["type"] == "object"
