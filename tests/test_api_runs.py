# tests/test_api_runs.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings


def _client(tmp_path, **env) -> TestClient:
    env = {"DS_RUNS_DIR": str(tmp_path), **env}
    return TestClient(create_app(ApiSettings.from_env(env)))


def test_create_run_sync_returns_full_result(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post("/runs", json={"prompt": "compare A and B", "mode": "sync"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["spec"]["graph_id"]
    assert body["values"]
    assert body["links"]["self"].startswith("/runs/")


def test_create_run_async_returns_202(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/runs", json={"prompt": "x", "mode": "async", "run_id": "async-1"}
    )
    assert resp.status_code == 202
    assert resp.json()["run_id"] == "async-1"


def test_async_run_becomes_visible(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "mode": "async", "run_id": "async-2"})
    # mock runs finish near-instantly; poll once.
    status = client.get("/runs/async-2")
    assert status.status_code == 200
    assert status.json()["run_id"] == "async-2"


def test_auto_run_fast_returns_200(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post("/runs", json={"prompt": "x", "run_id": "auto-1"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_duplicate_run_id_conflicts(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "run_id": "dup", "mode": "sync"})
    resp = client.post("/runs", json={"prompt": "y", "run_id": "dup", "mode": "sync"})
    assert resp.status_code == 409


def test_get_run_404(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.get("/runs/ghost").status_code == 404


def test_list_runs(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "run_id": "list-1", "mode": "sync"})
    resp = client.get("/runs")
    assert resp.status_code == 200
    ids = {r["run_id"] for r in resp.json()["runs"]}
    assert "list-1" in ids


def test_file_subresources(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "run_id": "files-1", "mode": "sync"})
    assert client.get("/runs/files-1/spec").status_code == 200
    assert client.get("/runs/files-1/output").status_code == 200
    assert client.get("/runs/files-1/trace").status_code == 200
    graph = client.get("/runs/files-1/graph")
    assert graph.status_code == 200
    assert "graph" in graph.text.lower() or "-->" in graph.text
    assert client.get("/runs/files-1/summary").status_code == 200


def test_model_outside_allowlist_rejected(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post("/runs", json={"prompt": "x", "model": "evil", "mode": "sync"})
    assert resp.status_code == 400


def test_auth_required_when_key_set(tmp_path) -> None:
    client = _client(tmp_path, DS_API_KEY="secret")
    resp = client.post("/runs", json={"prompt": "x", "mode": "sync"})
    assert resp.status_code == 401
    ok = client.post(
        "/runs",
        json={"prompt": "x", "mode": "sync", "run_id": "authed"},
        headers={"Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200
