"""_EvalProducer + ArtifactContext.eval_result tests (Slice 7 PR3)."""

import json
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
    ctx = ArtifactContext(
        run_id="r1",
        directory=tmp_path,
        spec=spec,
        result=result,
        prompt=None,
        eval_result=_eval_payload(),
    )
    producer = _EvalProducer()
    assert producer.applies(ctx) is True
    path = producer.write(ctx)
    assert path.name == "eval.json"
    assert json.loads(path.read_text(encoding="utf-8"))["quality"] == 0.8


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
