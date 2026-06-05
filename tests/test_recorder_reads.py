# tests/test_recorder_reads.py
from __future__ import annotations

from pathlib import Path

import pytest

from app.assembly import RunConfig, build_supervisor
from app.recording import FileRecorder


def _seed_run(tmp_path: Path, run_id: str) -> FileRecorder:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    supervisor = build_supervisor(
        RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False),
        recorder=recorder,
    )
    supervisor.run("seed", run_id=run_id)
    return recorder


def test_exists_true_after_run(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    assert recorder.exists("rec-001") is True


def test_exists_false_for_unknown(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path)
    assert recorder.exists("nope") is False


def test_list_runs_returns_seeded_ids(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    _seed_run(tmp_path, "rec-002")

    summaries = recorder.list_runs()

    ids = {s["run_id"] for s in summaries}
    assert {"rec-001", "rec-002"} <= ids
    sample = next(s for s in summaries if s["run_id"] == "rec-001")
    assert sample["status"] in {"ok", "failed", "paused"}
    assert isinstance(sample["nodes"], int)


def test_load_output_has_values(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    output = recorder.load_output("rec-001")
    assert "values" in output and "ok" in output


def test_load_output_missing_raises(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        recorder.load_output("ghost")


def test_artifact_path_rejects_traversal(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    with pytest.raises(ValueError):
        recorder.artifact_path("rec-001", "../escape.txt")


def test_run_dir_points_at_run(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    assert recorder.run_dir("rec-001") == tmp_path / "rec-001"


def test_run_dir_rejects_bare_dot_segments(tmp_path: Path) -> None:
    # The charset excludes separators, so the only traversal token is an all-dots
    # id ('..' would select the runs-root parent). It must be refused.
    recorder = FileRecorder(root_dir=tmp_path)
    for bad in ("..", ".", "..."):
        with pytest.raises(ValueError):
            recorder.run_dir(bad)
