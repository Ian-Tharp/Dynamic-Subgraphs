"""_EvalProducer + ArtifactContext.eval_result tests (Slice 7 PR3)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.models import GraphSpec
from app.recording.recorder import (
    DEFAULT_PRODUCERS,
    ArtifactContext,
    FileRecorder,
    _EvalProducer,
)
from app.registry import validate_graph_spec
from app.runtime import ExecutionResult


def _run_pipeline(spec: GraphSpec, *, run_id: str, runners=None) -> ExecutionResult:
    from app.runtime import LangGraphExecutor

    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(runners=runners)
    return executor.execute(executor.compile(validated), run_id=run_id)


@pytest.fixture
def successful_result(minimal_spec: GraphSpec) -> tuple[GraphSpec, ExecutionResult]:
    return minimal_spec, _run_pipeline(minimal_spec, run_id="eval-ok")


def _eval_payload() -> dict:
    return {
        "schema_version": 1,
        "run_id": "r1",
        "gate": "deterministic@v1",
        "quality": 0.8,
        "total_tokens": 1234,
    }


def test_eval_producer_skips_without_payload(tmp_path, minimal_spec: GraphSpec) -> None:
    spec, result = minimal_spec, _run_pipeline(minimal_spec, run_id="no-eval")
    ctx = ArtifactContext(
        run_id="r1", directory=tmp_path, spec=spec, result=result, prompt=None
    )
    assert _EvalProducer().applies(ctx) is False


def test_eval_producer_writes_json(tmp_path, minimal_spec: GraphSpec) -> None:
    spec, result = minimal_spec, _run_pipeline(minimal_spec, run_id="with-eval")
    payload = _eval_payload() | {"evaluated_at": datetime.now(UTC)}
    ctx = ArtifactContext(
        run_id="r1",
        directory=tmp_path,
        spec=spec,
        result=result,
        prompt=None,
        eval_result=payload,
    )
    producer = _EvalProducer()
    assert producer.applies(ctx) is True
    path = producer.write(ctx)
    assert path.name == "eval.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["quality"] == 0.8
    # default=str gracefully degrades non-JSON types (datetime) to strings.
    assert isinstance(loaded["evaluated_at"], str)


def test_eval_producer_registered() -> None:
    assert any(p.filename == "eval.json" for p in DEFAULT_PRODUCERS)


def test_existing_context_call_sites_still_compile(
    tmp_path: Path, minimal_spec: GraphSpec
) -> None:
    # eval_result is defaulted — record() without it must keep working.
    spec, result = minimal_spec, _run_pipeline(minimal_spec, run_id="compat")
    rec = FileRecorder(root_dir=tmp_path)
    record = rec.record(spec=spec, result=result)
    assert "eval" not in record.artifacts  # applies() gated it off


def test_record_eval_round_trip(tmp_path: Path, minimal_spec: GraphSpec) -> None:
    spec, result = minimal_spec, _run_pipeline(minimal_spec, run_id="round-trip")
    rec = FileRecorder(root_dir=tmp_path)
    rec.record(spec=spec, result=result)
    run_id = result.trace.run_id
    path = rec.record_eval(run_id, _eval_payload())
    assert path is not None and path.name == "eval.json"
    loaded = rec.load_eval(run_id)
    assert loaded is not None and loaded["quality"] == 0.8


def test_record_eval_respects_selection(
    tmp_path: Path, minimal_spec: GraphSpec
) -> None:
    # EVAL not in selection -> record_eval is a no-op returning None.
    spec, result = minimal_spec, _run_pipeline(minimal_spec, run_id="selection-test")
    rec = FileRecorder(root_dir=tmp_path, selection=frozenset({"spec.json"}))
    rec.record(spec=spec, result=result)
    run_id = result.trace.run_id
    assert rec.record_eval(run_id, _eval_payload()) is None
    assert not (tmp_path / run_id / "eval.json").exists()


def test_load_eval_none_for_old_dirs(tmp_path: Path, minimal_spec: GraphSpec) -> None:
    spec, result = minimal_spec, _run_pipeline(minimal_spec, run_id="old-dir")
    rec = FileRecorder(root_dir=tmp_path)
    rec.record(spec=spec, result=result)
    assert rec.load_eval(result.trace.run_id) is None


def test_list_runs_surfaces_eval_fields(
    tmp_path: Path, minimal_spec: GraphSpec
) -> None:
    spec, result = minimal_spec, _run_pipeline(minimal_spec, run_id="list-eval")
    rec = FileRecorder(root_dir=tmp_path)
    rec.record(spec=spec, result=result)
    run_id = result.trace.run_id
    rec.record_eval(
        run_id,
        {
            **_eval_payload(),
            "value_per_ktok": 0.65,
            "fingerprint": {"origin": "invented"},
            "deterministic": True,
            "quality_floor_met": True,
        },
    )
    (entry,) = [r for r in rec.list_runs() if r["run_id"] == run_id]
    assert entry["quality"] == 0.8
    assert entry["value_per_ktok"] == 0.65
    assert entry["origin"] == "invented"
    assert entry["deterministic"] is True
    assert entry["quality_floor_met"] is True


def test_list_runs_omits_eval_fields_when_absent(
    tmp_path: Path, minimal_spec: GraphSpec
) -> None:
    spec, result = minimal_spec, _run_pipeline(minimal_spec, run_id="no-eval-list")
    rec = FileRecorder(root_dir=tmp_path)
    rec.record(spec=spec, result=result)
    (entry,) = rec.list_runs()
    assert "quality" not in entry


def test_list_runs_skips_corrupt_eval_json(
    tmp_path: Path, minimal_spec: GraphSpec
) -> None:
    spec, result = minimal_spec, _run_pipeline(minimal_spec, run_id="corrupt-eval")
    rec = FileRecorder(root_dir=tmp_path)
    rec.record(spec=spec, result=result)
    run_id = result.trace.run_id
    (tmp_path / run_id / "eval.json").write_text("{bad json", encoding="utf-8")
    (entry,) = rec.list_runs()
    assert "quality" not in entry


def test_record_eval_unrecorded_run_raises(tmp_path: Path) -> None:
    rec = FileRecorder(root_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        rec.record_eval("never-recorded", _eval_payload())
