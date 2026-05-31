# tests/test_api_chains.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)})))


def test_create_chain_sync(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/chains",
        json={"prompt": "investigate", "run_id": "chain-1", "mode": "sync", "max_iterations": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chain_id"] == "chain-1"
    assert body["status"] in {"ok", "stopped", "max_iterations"}
    assert isinstance(body["steps"], list)


def test_get_chain(tmp_path) -> None:
    client = _client(tmp_path)
    client.post(
        "/chains",
        json={"prompt": "investigate", "run_id": "chain-2", "mode": "sync", "max_iterations": 1},
    )
    resp = client.get("/chains/chain-2")
    assert resp.status_code == 200
    assert resp.json()["chain_id"] == "chain-2"


def test_get_chain_404(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.get("/chains/ghost").status_code == 404


def test_create_chain_async_202(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/chains",
        json={"prompt": "x", "run_id": "chain-3", "mode": "async", "max_iterations": 1},
    )
    assert resp.status_code == 202
