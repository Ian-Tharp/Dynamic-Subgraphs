# tests/test_assembly.py
from __future__ import annotations

from pathlib import Path

from app.assembly import RunConfig, build_supervisor
from app.recording import FileRecorder
from app.supervisor import StaticPlanner, Supervisor


def test_build_supervisor_mock_returns_supervisor(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)

    supervisor = build_supervisor(config, recorder=recorder)

    assert isinstance(supervisor, Supervisor)


def test_build_supervisor_mock_runs_end_to_end(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)
    supervisor = build_supervisor(config, recorder=recorder)

    result = supervisor.run("compare two things", run_id="assembly-001")

    assert result.status == "ok"
    assert (tmp_path / "assembly-001" / "spec.json").exists()


def test_build_supervisor_mock_uses_static_planner(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)

    supervisor = build_supervisor(config, recorder=recorder)

    assert isinstance(supervisor._planner, StaticPlanner)
