"""FileRecorder coverage: artifacts on disk, round-trips, safety, failure capture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import GraphSpec, NodeKind, RunTrace, TraceEvent, TraceEventKind
from app.recording import FileRecorder, Recorder, render_mermaid
from app.registry import validate_graph_spec
from app.runtime import ExecutionResult, LangGraphExecutor


# ---------- fixtures local to recording ----------


def _run_pipeline(spec: GraphSpec, *, run_id: str, runners=None) -> ExecutionResult:
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(runners=runners)
    return executor.execute(executor.compile(validated), run_id=run_id)


@pytest.fixture
def successful_result(minimal_spec: GraphSpec) -> tuple[GraphSpec, ExecutionResult]:
    return minimal_spec, _run_pipeline(minimal_spec, run_id="rec-ok")


@pytest.fixture
def failed_result(minimal_spec: GraphSpec) -> tuple[GraphSpec, ExecutionResult]:
    def explode(state, params):
        raise RuntimeError("boom")

    return minimal_spec, _run_pipeline(
        minimal_spec, run_id="rec-fail", runners={NodeKind.LLM_CALL: explode}
    )


# ---------- recorder satisfies the Protocol ----------


def test_file_recorder_satisfies_recorder_protocol() -> None:
    recorder: Recorder = FileRecorder()

    assert hasattr(recorder, "record")


# ---------- artifacts on disk ----------


def test_recorder_creates_all_expected_files(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(spec=spec, result=result)

    assert record.directory == tmp_path / "rec-ok"
    for name in ("spec.json", "trace.jsonl", "output.json", "graph.mmd", "summary.md"):
        assert (record.directory / name).exists(), f"missing {name}"


def test_record_returns_paths_for_each_artifact(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(spec=spec, result=result)

    assert set(record.artifacts.keys()) >= {
        "spec",
        "trace",
        "output",
        "mermaid",
        "summary",
    }
    for path in record.artifacts.values():
        assert path.is_file()


# ---------- file contents ----------


def test_spec_json_round_trips_back_to_graph_spec(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(spec=spec, result=result)
    raw = json.loads(record.artifacts["spec"].read_text(encoding="utf-8"))
    restored = GraphSpec.model_validate(raw)

    assert restored.graph_id == spec.graph_id
    assert [n.id for n in restored.nodes] == [n.id for n in spec.nodes]


def test_trace_jsonl_is_line_delimited_and_each_event_parses(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(spec=spec, result=result)
    lines = record.artifacts["trace"].read_text(encoding="utf-8").splitlines()
    events = [TraceEvent.model_validate_json(line) for line in lines]

    assert len(events) == len(result.trace.events)
    assert [e.kind for e in events] == [e.kind for e in result.trace.events]


def test_output_json_excludes_trace_events_but_includes_status(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(spec=spec, result=result)
    output = json.loads(record.artifacts["output"].read_text(encoding="utf-8"))

    assert "events" not in output
    assert output["ok"] is True
    assert output["error"] is None
    assert "values" in output


def test_mermaid_contains_every_node_id_and_special_sentinels(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(spec=spec, result=result)
    mermaid = record.artifacts["mermaid"].read_text(encoding="utf-8")

    assert "graph TD" in mermaid
    assert "START" in mermaid
    assert "END" in mermaid
    for node in spec.nodes:
        assert node.id in mermaid
        assert node.kind.value in mermaid


def test_summary_md_reports_status_goal_and_counts(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(spec=spec, result=result)
    summary = record.artifacts["summary"].read_text(encoding="utf-8")

    assert spec.goal in summary
    assert spec.graph_id in summary
    assert "ok" in summary.lower()


# ---------- failed runs are first-class ----------


def test_failed_run_is_still_recorded(tmp_path: Path, failed_result) -> None:
    spec, result = failed_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(spec=spec, result=result)
    output = json.loads(record.artifacts["output"].read_text(encoding="utf-8"))
    summary = record.artifacts["summary"].read_text(encoding="utf-8")

    assert output["ok"] is False
    assert output["error"] == "boom"
    assert output["errors"][0]["type"] == "RuntimeError"
    assert "failed" in summary.lower()


# ---------- overwrite policy ----------


def test_recorder_refuses_to_overwrite_existing_run_by_default(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    recorder.record(spec=spec, result=result)
    with pytest.raises(FileExistsError):
        recorder.record(spec=spec, result=result)


def test_recorder_overwrites_when_overwrite_true(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)

    recorder.record(spec=spec, result=result)
    second = recorder.record(spec=spec, result=result)

    assert second.directory.exists()


# ---------- prompt is optional ----------


def test_prompt_md_is_absent_when_no_prompt_provided(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(spec=spec, result=result)

    assert "prompt" not in record.artifacts
    assert not (record.directory / "prompt.md").exists()


def test_prompt_md_is_written_when_prompt_provided(
    tmp_path: Path, successful_result
) -> None:
    spec, result = successful_result
    recorder = FileRecorder(root_dir=tmp_path)

    record = recorder.record(
        spec=spec, result=result, prompt="investigate library tradeoffs"
    )

    assert (record.directory / "prompt.md").read_text(encoding="utf-8") == (
        "investigate library tradeoffs"
    )


# ---------- run_id safety ----------


@pytest.mark.parametrize(
    "bad_id",
    ["", "../escape", "..\\escape", "a/b", "a\\b", "a:b", "a b", "weird?id"],
)
def test_recorder_rejects_unsafe_run_id(
    tmp_path: Path, minimal_spec: GraphSpec, bad_id: str
) -> None:
    trace = RunTrace(
        run_id=bad_id,
        graph_id=minimal_spec.graph_id,
        events=[
            TraceEvent(kind=TraceEventKind.NODE_START, node_id="step"),
            TraceEvent(kind=TraceEventKind.NODE_FINISH, node_id="step"),
        ],
    )
    result = ExecutionResult(state={"values": {}}, trace=trace, ok=True, error=None)
    recorder = FileRecorder(root_dir=tmp_path)

    with pytest.raises(ValueError):
        recorder.record(spec=minimal_spec, result=result)


# ---------- mermaid renderer is deterministic ----------


def test_render_mermaid_is_deterministic_for_same_spec(
    minimal_spec: GraphSpec,
) -> None:
    a = render_mermaid(minimal_spec)
    b = render_mermaid(minimal_spec)

    assert a == b


def test_render_mermaid_emits_one_arrow_per_edge(minimal_spec: GraphSpec) -> None:
    rendered = render_mermaid(minimal_spec)

    arrow_count = rendered.count("-->")

    assert arrow_count == len(minimal_spec.edges)
