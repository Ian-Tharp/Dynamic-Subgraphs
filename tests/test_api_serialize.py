# tests/test_api_serialize.py
from __future__ import annotations

from pathlib import Path

from app.api.serialize import run_result_payload, run_status_payload
from app.assembly import RunConfig, build_supervisor
from app.recording import FileRecorder


def _run(tmp_path: Path, run_id: str):
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    supervisor = build_supervisor(
        RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False),
        recorder=recorder,
    )
    return supervisor.run("compare", run_id=run_id)


def test_run_result_payload_shape(tmp_path: Path) -> None:
    result = _run(tmp_path, "ser-001")
    payload = run_result_payload(result)
    assert payload["run_id"] == "ser-001"
    assert payload["status"] == "ok"
    assert "values" in payload
    assert payload["spec"] is not None
    assert payload["spec"]["graph_id"]


def test_run_status_payload_shape(tmp_path: Path) -> None:
    result = _run(tmp_path, "ser-002")
    payload = run_status_payload(result)
    assert payload["run_id"] == "ser-002"
    assert payload["state"] in {"ok", "failed", "paused"}
