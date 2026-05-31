"""End-to-end coverage for emit_artifact: sinks, runner, full pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.registry import validate_graph_spec
from app.runtime import (
    CollectingArtifactSink,
    FileArtifactSink,
    LangGraphExecutor,
    make_emit_artifact_runner,
    run_emit_artifact,
)
from app.runtime.state import make_initial_state


# ---------- echo default ----------


def test_default_run_emit_artifact_returns_fake_reference_without_io(
    tmp_path: Path,
) -> None:
    out = run_emit_artifact(
        make_initial_state(),
        {
            "artifact_type": "report",
            "name": "summary",
            "content_key": "draft",
        },
    )

    assert out == {
        "result": {
            "name": "summary",
            "type": "report",
            "path": "<unpersisted>",
        }
    }
    # No files written to disk by the echo.
    assert list(tmp_path.iterdir()) == []


# ---------- CollectingArtifactSink ----------


def test_collecting_sink_captures_emissions_with_content() -> None:
    sink = CollectingArtifactSink()

    sink.emit(
        name="report-1", artifact_type="report", content="hello", run_id="r1"
    )
    sink.emit(
        name="data", artifact_type="file", content={"k": 1}, run_id="r1"
    )

    assert len(sink.emitted) == 2
    assert sink.emitted[0]["name"] == "report-1"
    assert sink.emitted[0]["content"] == "hello"
    assert sink.emitted[1]["content"] == {"k": 1}


# ---------- FileArtifactSink ----------


def test_file_sink_writes_string_content_with_type_extension(
    tmp_path: Path,
) -> None:
    sink = FileArtifactSink(root_dir=tmp_path)

    ref = sink.emit(
        name="proposal",
        artifact_type="report",
        content="# Proposal\n\nText.",
        run_id="run-x",
    )

    assert ref["type"] == "report"
    assert ref["run_id"] == "run-x"
    path = Path(ref["path"])
    assert path.exists()
    assert path.name == "proposal.md"
    assert path.parent == tmp_path / "run-x" / "artifacts"
    assert path.read_text(encoding="utf-8") == "# Proposal\n\nText."


def test_file_sink_writes_structured_content_as_json(tmp_path: Path) -> None:
    sink = FileArtifactSink(root_dir=tmp_path)

    ref = sink.emit(
        name="snapshot",
        artifact_type="file",
        content={"alpha": 1, "items": ["a", "b"]},
        run_id="r",
    )

    path = Path(ref["path"])
    assert path.suffix == ".json"
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed == {"alpha": 1, "items": ["a", "b"]}


def test_file_sink_sanitizes_unsafe_artifact_names(tmp_path: Path) -> None:
    sink = FileArtifactSink(root_dir=tmp_path)

    ref = sink.emit(
        name="../escape attempt/with weird?chars",
        artifact_type="report",
        content="x",
        run_id="r",
    )

    path = Path(ref["path"])
    expected_dir = (tmp_path / "r" / "artifacts").resolve()

    # The dangerous bits — path separators and special chars — are stripped.
    assert "/" not in path.name
    assert "\\" not in path.name
    assert "?" not in path.name
    assert " " not in path.name
    # And the resulting file lands inside the expected directory (no
    # traversal escapes via `..` as a real path component).
    assert path.resolve().parent == expected_dir
    assert path.exists()


def test_file_sink_falls_back_to_unscoped_when_no_run_id(tmp_path: Path) -> None:
    sink = FileArtifactSink(root_dir=tmp_path)

    ref = sink.emit(
        name="standalone",
        artifact_type="file",
        content="solo",
        run_id=None,
    )

    path = Path(ref["path"])
    assert path.parent == tmp_path / "_unscoped" / "artifacts"
    assert "run_id" not in ref


# ---------- make_emit_artifact_runner ----------


def test_runner_reads_content_from_state_and_calls_sink() -> None:
    sink = CollectingArtifactSink()
    runner = make_emit_artifact_runner(sink)

    state = make_initial_state(metadata={"run_id": "rid-1"})
    state["values"]["draft"] = "the document body"

    out = runner(
        state,
        {
            "artifact_type": "report",
            "name": "final",
            "content_key": "draft",
        },
    )

    assert out["result"]["name"] == "final"
    assert out["result"]["type"] == "report"
    assert sink.emitted[0]["content"] == "the document body"
    assert sink.emitted[0]["run_id"] == "rid-1"


def test_runner_raises_clear_error_when_content_key_missing_from_state() -> None:
    sink = CollectingArtifactSink()
    runner = make_emit_artifact_runner(sink)

    with pytest.raises(RuntimeError, match="content_key 'ghost'"):
        runner(
            make_initial_state(),
            {
                "artifact_type": "file",
                "name": "out",
                "content_key": "ghost",
            },
        )


# ---------- end-to-end ----------


def test_end_to_end_emit_artifact_with_default_echo_runner() -> None:
    spec = GraphSpec(
        graph_id="emit-e2e-echo",
        goal="exercise emit_artifact end-to-end with the echo default",
        nodes=[
            NodeSpec(
                id="draft",
                kind=NodeKind.LLM_CALL,
                outputs=["report_body"],
                params={"instruction": "draft a report"},
            ),
            NodeSpec(
                id="publish",
                kind=NodeKind.EMIT_ARTIFACT,
                inputs=["report_body"],
                outputs=["report_ref"],
                params={
                    "artifact_type": "report",
                    "name": "final_report",
                    "content_key": "report_body",
                },
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "draft"}),
            EdgeSpec.model_validate({"from": "draft", "to": "publish"}),
            EdgeSpec.model_validate({"from": "publish", "to": "END"}),
        ],
    )
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor()

    result = executor.execute(executor.compile(validated), run_id="emit-echo")

    assert result.ok is True
    ref = result.state["values"]["report_ref"]
    assert ref["name"] == "final_report"
    assert ref["type"] == "report"
    assert ref["path"] == "<unpersisted>"


def test_end_to_end_emit_artifact_with_file_sink_writes_to_disk(
    tmp_path: Path,
) -> None:
    spec = GraphSpec(
        graph_id="emit-e2e-file",
        goal="exercise emit_artifact end-to-end with a real FileArtifactSink",
        nodes=[
            NodeSpec(
                id="draft",
                kind=NodeKind.LLM_CALL,
                outputs=["body"],
                params={"instruction": "the final write-up content"},
            ),
            NodeSpec(
                id="publish",
                kind=NodeKind.EMIT_ARTIFACT,
                inputs=["body"],
                outputs=["ref"],
                params={
                    "artifact_type": "report",
                    "name": "writeup",
                    "content_key": "body",
                },
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "draft"}),
            EdgeSpec.model_validate({"from": "draft", "to": "publish"}),
            EdgeSpec.model_validate({"from": "publish", "to": "END"}),
        ],
    )
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={
            NodeKind.EMIT_ARTIFACT: make_emit_artifact_runner(
                FileArtifactSink(root_dir=tmp_path)
            ),
        }
    )

    result = executor.execute(executor.compile(validated), run_id="emit-file")

    assert result.ok is True
    ref = result.state["values"]["ref"]
    file_path = Path(ref["path"])
    assert file_path.exists()
    assert file_path.parent == tmp_path / "emit-file" / "artifacts"
    assert file_path.name == "writeup.md"
    assert file_path.read_text(encoding="utf-8") == "the final write-up content"
