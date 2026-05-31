# tests/test_api_stream.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings


def test_stream_emits_status_and_done(tmp_path) -> None:
    app = create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)}))
    client = TestClient(app)
    client.post("/runs", json={"prompt": "x", "run_id": "stream-1", "mode": "async"})

    with client.stream("GET", "/runs/stream-1/trace/stream") as resp:
        assert resp.status_code == 200
        text = "".join(chunk for chunk in resp.iter_text())

    assert "event: status" in text
    assert "event: done" in text
