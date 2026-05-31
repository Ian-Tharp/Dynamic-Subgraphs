"""End-to-end coverage for wait_for_event: pause, resume, multiple waits,
checkpointer enforcement, and supervisor integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.compiler import GraphCompilationError
from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.recording import FileRecorder
from app.registry import validate_graph_spec
from app.runtime import LangGraphExecutor
from app.supervisor import StaticPlanner, Supervisor


# ---------- runners ----------


def _identity_llm(state, params):
    return {"result": params["instruction"]}


def _echo_with_upstream(state, params):
    """Surface upstream values in the result so we can confirm event payloads propagate."""
    values = state.get("values", {}) or {}
    return {"result": {"instruction": params["instruction"], "upstream": dict(values)}}


# ---------- spec helpers ----------


def _wait_then_finalize_spec() -> GraphSpec:
    """seed -> wait_for_user -> finalize -> END."""
    nodes = [
        NodeSpec(
            id="seed",
            kind=NodeKind.LLM_CALL,
            outputs=["question"],
            params={"instruction": "what should we ask the human?"},
        ),
        NodeSpec(
            id="wait_for_user",
            kind=NodeKind.WAIT_FOR_EVENT,
            outputs=["user_answer"],
            params={"event_type": "user_response"},
        ),
        NodeSpec(
            id="finalize",
            kind=NodeKind.LLM_CALL,
            inputs=["user_answer"],
            outputs=["final"],
            params={"instruction": "wrap up with the user's answer"},
        ),
    ]
    edges = [
        EdgeSpec.model_validate({"from": "START", "to": "seed"}),
        EdgeSpec.model_validate({"from": "seed", "to": "wait_for_user"}),
        EdgeSpec.model_validate({"from": "wait_for_user", "to": "finalize"}),
        EdgeSpec.model_validate({"from": "finalize", "to": "END"}),
    ]
    return GraphSpec(
        graph_id="wait-test",
        goal="basic wait/resume",
        nodes=nodes,
        edges=edges,
    )


def _two_sequential_waits_spec() -> GraphSpec:
    """seed -> wait_a -> wait_b -> finalize."""
    nodes = [
        NodeSpec(
            id="seed",
            kind=NodeKind.LLM_CALL,
            outputs=["context"],
            params={"instruction": "set context"},
        ),
        NodeSpec(
            id="wait_a",
            kind=NodeKind.WAIT_FOR_EVENT,
            outputs=["event_a"],
            params={"event_type": "first_event"},
        ),
        NodeSpec(
            id="wait_b",
            kind=NodeKind.WAIT_FOR_EVENT,
            inputs=["event_a"],
            outputs=["event_b"],
            params={"event_type": "second_event"},
        ),
        NodeSpec(
            id="finalize",
            kind=NodeKind.LLM_CALL,
            inputs=["event_a", "event_b"],
            outputs=["final"],
            params={"instruction": "use both events"},
        ),
    ]
    edges = [
        EdgeSpec.model_validate({"from": "START", "to": "seed"}),
        EdgeSpec.model_validate({"from": "seed", "to": "wait_a"}),
        EdgeSpec.model_validate({"from": "wait_a", "to": "wait_b"}),
        EdgeSpec.model_validate({"from": "wait_b", "to": "finalize"}),
        EdgeSpec.model_validate({"from": "finalize", "to": "END"}),
    ]
    return GraphSpec(
        graph_id="two-waits",
        goal="sequential pauses",
        nodes=nodes,
        edges=edges,
    )


# ---------- checkpointer enforcement ----------


def test_compile_requires_checkpointer_when_wait_for_event_present() -> None:
    spec = _wait_then_finalize_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor()  # no checkpointer

    with pytest.raises(GraphCompilationError) as exc:
        executor.compile(validated)

    assert "checkpointer" in str(exc.value).lower()


def test_resume_without_checkpointer_raises() -> None:
    # Spec without wait_for_event so compile() succeeds; resume() still
    # must reject if no checkpointer is configured.
    spec = GraphSpec(
        graph_id="no-wait",
        goal="no wait",
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["draft"],
                params={"instruction": "x"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
    )
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(runners={NodeKind.LLM_CALL: _identity_llm})
    compiled = executor.compile(validated)

    with pytest.raises(RuntimeError, match="no checkpointer"):
        executor.resume(compiled, run_id="x", event=None)


# ---------- pause + resume ----------


def test_executor_pauses_at_wait_for_event_and_reports_paused() -> None:
    spec = _wait_then_finalize_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: _identity_llm},
        checkpointer=MemorySaver(),
    )
    compiled = executor.compile(validated)

    result = executor.execute(compiled, run_id="wait-1")

    assert result.paused is True
    assert result.ok is False
    assert result.error is None
    payloads = result.interrupt_payloads
    assert len(payloads) == 1
    assert payloads[0]["event_type"] == "user_response"
    assert payloads[0]["node_id"] == "wait_for_user"


def test_executor_resume_completes_run_and_routes_payload_into_output() -> None:
    spec = _wait_then_finalize_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: _echo_with_upstream},
        checkpointer=MemorySaver(),
    )
    compiled = executor.compile(validated)

    executor.execute(compiled, run_id="wait-resume")
    result = executor.resume(
        compiled, run_id="wait-resume", event="the user's reply"
    )

    assert result.paused is False
    assert result.ok is True
    assert result.state["values"]["user_answer"] == "the user's reply"
    # finalize ran after resume and saw the user_answer in upstream state.
    final = result.state["values"]["final"]
    assert isinstance(final, dict)
    assert final["upstream"]["user_answer"] == "the user's reply"


def test_executor_handles_two_sequential_waits() -> None:
    spec = _two_sequential_waits_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: _identity_llm},
        checkpointer=MemorySaver(),
    )
    compiled = executor.compile(validated)

    first = executor.execute(compiled, run_id="two-waits")
    assert first.paused is True
    assert first.interrupt_payloads[0]["event_type"] == "first_event"

    second = executor.resume(
        compiled, run_id="two-waits", event="payload-A"
    )
    assert second.paused is True
    assert second.interrupt_payloads[0]["event_type"] == "second_event"
    assert second.state["values"]["event_a"] == "payload-A"

    third = executor.resume(
        compiled, run_id="two-waits", event="payload-B"
    )
    assert third.paused is False
    assert third.ok is True
    assert third.state["values"]["event_a"] == "payload-A"
    assert third.state["values"]["event_b"] == "payload-B"


def test_wait_node_emits_start_and_finish_trace_on_resume() -> None:
    spec = _wait_then_finalize_spec()
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={NodeKind.LLM_CALL: _identity_llm},
        checkpointer=MemorySaver(),
    )
    compiled = executor.compile(validated)

    executor.execute(compiled, run_id="wait-trace")
    final = executor.resume(compiled, run_id="wait-trace", event="answer!")

    wait_events = [e for e in final.trace.events if e.node_id == "wait_for_user"]
    kinds = [e.kind.value for e in wait_events]
    assert kinds == ["node_start", "node_finish"]
    assert wait_events[1].data.get("event_type") == "user_response"


# ---------- supervisor integration ----------


def test_supervisor_reports_paused_status_and_persists_partial_run(
    tmp_path: Path,
) -> None:
    spec = _wait_then_finalize_spec()
    supervisor = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(
            runners={NodeKind.LLM_CALL: _identity_llm},
            checkpointer=MemorySaver(),
        ),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    result = supervisor.run("ask the human a question", run_id="sup-pause")

    assert result.status == "paused"
    assert result.result is not None and result.result.paused is True
    assert result.record is not None
    assert (tmp_path / "sup-pause" / "spec.json").exists()
    assert "user_response" in result.response


def test_supervisor_resume_completes_paused_run(tmp_path: Path) -> None:
    spec = _wait_then_finalize_spec()
    supervisor = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(
            runners={NodeKind.LLM_CALL: _echo_with_upstream},
            checkpointer=MemorySaver(),
        ),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    paused = supervisor.run("ask the human", run_id="sup-resume")
    assert paused.status == "paused"

    resumed = supervisor.resume("sup-resume", event="the user's answer")

    assert resumed.status == "ok"
    assert resumed.result is not None and resumed.result.ok is True
    assert (
        resumed.result.state["values"]["user_answer"] == "the user's answer"
    )


def test_supervisor_resume_with_unknown_run_id_reports_resume_failed(
    tmp_path: Path,
) -> None:
    spec = _wait_then_finalize_spec()
    supervisor = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(
            runners={NodeKind.LLM_CALL: _identity_llm},
            checkpointer=MemorySaver(),
        ),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    result = supervisor.resume("never-existed", event="payload")

    assert result.status == "resume_failed"
    assert result.errors[-1]["stage"] == "resume_load_spec"


# ---------- recorder load capability ----------


def test_recorder_load_validated_spec_round_trips(
    tmp_path: Path, minimal_spec: GraphSpec
) -> None:
    spec = minimal_spec
    supervisor = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _identity_llm}),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    supervisor.run("prompt", run_id="load-test")
    loaded = FileRecorder(root_dir=tmp_path).load_validated_spec("load-test")

    assert loaded.graph_id == spec.graph_id
    assert [n.id for n in loaded.nodes] == [n.id for n in spec.nodes]


def test_recorder_load_missing_run_raises_filenotfound(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        recorder.load_validated_spec("no-such-run")
