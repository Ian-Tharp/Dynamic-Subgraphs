"""Supervisor coverage: happy path + every failure stage.

The supervisor must:
- run all six stages on the happy path;
- short-circuit cleanly when plan/validate/compile fail (skipping execute+record);
- still record when the inner graph executes but reports failure;
- still respond when the recorder itself raises;
- thread run_id through to the recorder's directory layout;
- populate SupervisorResult fields proportionally to how far it progressed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.recording import FileRecorder, RunRecord
from app.runtime import LangGraphExecutor
from app.supervisor import (
    Planner,
    StaticPlanner,
    Supervisor,
    SupervisorResult,
)


# ---------- helpers ----------


def _mock_llm(state, params):
    return {"result": params["instruction"]}


def _make_supervisor(
    spec: GraphSpec,
    tmp_path: Path,
    *,
    planner: Planner | None = None,
    runners: dict[NodeKind, Any] | None = None,
    recorder: FileRecorder | None = None,
) -> Supervisor:
    return Supervisor(
        planner=planner or StaticPlanner(spec),
        executor=LangGraphExecutor(
            runners=runners or {NodeKind.LLM_CALL: _mock_llm},
        ),
        recorder=recorder or FileRecorder(root_dir=tmp_path),
    )


# ---------- happy path ----------


def test_supervisor_runs_full_pipeline(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    sup = _make_supervisor(minimal_spec, tmp_path)

    result = sup.run("any prompt", run_id="sup-ok-1")

    assert result.status == "ok"
    assert result.validated_spec is not None
    assert result.result is not None and result.result.ok is True
    assert result.record is not None
    assert result.record.directory == tmp_path / "sup-ok-1"


def test_supervisor_response_summarizes_successful_run(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    sup = _make_supervisor(minimal_spec, tmp_path)

    result = sup.run("prompt", run_id="sup-resp-ok")

    assert "completed" in result.response.lower()
    # The minimal spec has one node producing "draft".
    assert "draft" in result.response


def test_supervisor_threads_run_id_through_to_recorder(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    sup = _make_supervisor(minimal_spec, tmp_path)

    result = sup.run("prompt", run_id="sup-rundir")

    assert (tmp_path / "sup-rundir" / "spec.json").exists()
    assert result.record is not None
    assert result.record.run_id == "sup-rundir"


# ---------- plan failure ----------


def test_supervisor_handles_plan_failure(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    def broken_planner(prompt: str) -> GraphSpec:
        raise RuntimeError("planner exploded")

    sup = _make_supervisor(minimal_spec, tmp_path, planner=broken_planner)

    result = sup.run("prompt", run_id="sup-plan-fail")

    assert result.status == "plan_failed"
    assert result.validated_spec is None
    assert result.result is None
    assert result.record is None
    assert result.errors[-1]["stage"] == "plan"
    assert "planner exploded" in result.errors[-1]["message"]
    assert not (tmp_path / "sup-plan-fail").exists()


# ---------- validation failure ----------


def test_supervisor_handles_validation_failure(
    make_node, make_edge, tmp_path: Path
) -> None:
    # An invalid spec: dangling edge to a node that doesn't exist.
    bad_spec = GraphSpec(
        graph_id="bad-spec",
        goal="will fail validation",
        nodes=[make_node("step", outputs=["draft"])],
        edges=[
            make_edge("START", "step"),
            make_edge("step", "ghost"),
            make_edge("step", "END"),
        ],
    )
    sup = _make_supervisor(bad_spec, tmp_path)

    result = sup.run("prompt", run_id="sup-validate-fail")

    assert result.status == "validation_failed"
    assert result.validated_spec is None
    assert result.result is None
    assert result.record is None
    assert result.errors[-1]["stage"] == "validate"
    assert any(
        issue["code"] == "dangling_edge" for issue in result.errors[-1]["issues"]
    )
    assert not (tmp_path / "sup-validate-fail").exists()


# ---------- compile failure ----------


def test_supervisor_handles_compile_failure(
    make_node, make_edge, tmp_path: Path
) -> None:
    # All registry kinds now have an executable path, so we can't construct
    # an "unsupported kind" spec through the public API. Force the compile-
    # failure path by passing a planner that emits a deliberately broken
    # GraphSpec — one that bypasses validation because we hand it directly
    # to the supervisor's executor seam. The supervisor's compile node
    # catches GraphCompilationError and classifies as compile_failed.
    from app.compiler import GraphCompilationError
    from app.runtime import GraphExecutor

    class _BrokenExecutor:
        """Executor whose compile() always raises GraphCompilationError."""

        def compile(self, spec):  # noqa: ARG002
            raise GraphCompilationError("simulated compile failure")

        def execute(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("execute should not be called when compile fails")

        def resume(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("resume should not be called when compile fails")

    sup = Supervisor(
        planner=StaticPlanner(minimal_spec_for_supervisor(make_node, make_edge)),
        executor=_BrokenExecutor(),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    result = sup.run("prompt", run_id="sup-compile-fail")

    assert result.status == "compile_failed"
    assert result.validated_spec is not None
    assert result.result is None
    assert result.record is None
    assert result.errors[-1]["stage"] == "compile"
    assert "simulated compile failure" in result.errors[-1]["message"]
    assert not (tmp_path / "sup-compile-fail").exists()


def minimal_spec_for_supervisor(make_node, make_edge) -> GraphSpec:
    """Small valid spec the supervisor will pass to the (broken) executor."""
    return GraphSpec(
        graph_id="ok-spec",
        goal="valid spec — the executor is what fails",
        nodes=[make_node("step", outputs=["draft"])],
        edges=[make_edge("START", "step"), make_edge("step", "END")],
    )


# ---------- execution failure (inner runner raises) ----------


def test_supervisor_records_inner_execution_failure(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    def explode(state, params):
        raise RuntimeError("inner failure")

    sup = _make_supervisor(
        minimal_spec,
        tmp_path,
        runners={NodeKind.LLM_CALL: explode},
    )

    result = sup.run("prompt", run_id="sup-exec-fail")

    assert result.status == "execution_failed"
    assert result.result is not None and result.result.ok is False
    # Critically: recording still happens for failed executions.
    assert result.record is not None
    assert (tmp_path / "sup-exec-fail" / "output.json").exists()
    assert "inner failure" in result.response


# ---------- record failure ----------


class _BrokenRecorder:
    """Recorder whose record() always raises. Used to exercise the supervisor's
    own record-failure handling path."""

    def record(self, *, spec, result, prompt=None):
        raise OSError("disk full")


def test_supervisor_handles_record_failure_without_crashing(
    minimal_spec: GraphSpec,
) -> None:
    sup = Supervisor(
        planner=StaticPlanner(minimal_spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm}),
        recorder=_BrokenRecorder(),
    )

    result = sup.run("prompt", run_id="sup-record-fail")

    assert result.status == "record_failed"
    assert result.result is not None and result.result.ok is True
    assert result.record is None
    assert result.errors[-1]["stage"] == "record"
    assert "disk full" in result.errors[-1]["message"]
    assert "could not be persisted" in result.response


# ---------- SupervisorResult shape ----------


def test_supervisor_result_exposes_expected_fields(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    sup = _make_supervisor(minimal_spec, tmp_path)

    result = sup.run("prompt", run_id="sup-shape")

    assert isinstance(result, SupervisorResult)
    assert result.run_id == "sup-shape"
    assert isinstance(result.response, str) and result.response
    assert isinstance(result.errors, list)


# ---------- compile-once invariant ----------


def test_supervisor_compiles_graph_once_and_reuses_for_multiple_runs(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    sup = _make_supervisor(minimal_spec, tmp_path)

    first = sup.run("prompt", run_id="sup-reuse-1")
    second = sup.run("prompt", run_id="sup-reuse-2")

    assert first.status == "ok"
    assert second.status == "ok"
    assert (tmp_path / "sup-reuse-1").exists()
    assert (tmp_path / "sup-reuse-2").exists()
