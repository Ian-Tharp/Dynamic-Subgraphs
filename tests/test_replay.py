"""Supervisor.replay: load spec, re-execute fresh, record under a new run id.

Replay is the "throw away the graph, keep the record" promise made concrete:
the original recording is the source of truth from which any future run can
be reconstructed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.recording import FileRecorder
from app.runtime import LangGraphExecutor
from app.supervisor import StaticPlanner, Supervisor


# ---------- runners ----------


def _identity_llm(state, params):
    return {"result": params["instruction"]}


def _counting_llm():
    """Runner that returns the invocation count, so two runs produce different output."""

    state: dict[str, int] = {"calls": 0}

    def runner(s, params):
        state["calls"] += 1
        return {"result": f"call#{state['calls']}: {params['instruction']}"}

    return runner


def _wait_then_finalize_spec() -> GraphSpec:
    return GraphSpec(
        graph_id="replay-wait",
        goal="replay against a spec containing wait_for_event",
        nodes=[
            NodeSpec(
                id="ask",
                kind=NodeKind.LLM_CALL,
                outputs=["question"],
                params={"instruction": "ask the human"},
            ),
            NodeSpec(
                id="wait",
                kind=NodeKind.WAIT_FOR_EVENT,
                outputs=["answer"],
                params={"event_type": "user_reply"},
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "ask"}),
            EdgeSpec.model_validate({"from": "ask", "to": "wait"}),
            EdgeSpec.model_validate({"from": "wait", "to": "END"}),
        ],
    )


# ---------- happy path ----------


def test_replay_runs_recorded_spec_and_writes_to_new_run_dir(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    supervisor = Supervisor(
        planner=StaticPlanner(minimal_spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _identity_llm}),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    original = supervisor.run("prompt", run_id="orig-1")
    replay = supervisor.replay("orig-1")

    assert original.status == "ok"
    assert replay.status == "ok"
    assert replay.run_id != "orig-1"
    assert replay.run_id.startswith("orig-1_replay_")
    assert (tmp_path / "orig-1" / "spec.json").exists()
    assert (tmp_path / replay.run_id / "spec.json").exists()


def test_replay_uses_recorded_spec_verbatim_without_calling_planner(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    """If the planner were called during replay, our broken planner would crash it."""
    call_log: list[str] = []

    def working_planner(prompt: str) -> GraphSpec:
        call_log.append(prompt)
        return minimal_spec

    def asserting_planner(prompt: str) -> GraphSpec:  # noqa: ARG001
        raise AssertionError("planner must not be called during replay")

    # First run with a working planner so we have a recorded spec to replay.
    sup_with_planner = Supervisor(
        planner=working_planner,
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _identity_llm}),
        recorder=FileRecorder(root_dir=tmp_path),
    )
    sup_with_planner.run("prompt", run_id="orig-no-replan")
    assert call_log == ["prompt"]  # original run did call the planner

    # Now construct a supervisor whose planner would explode if called.
    # Replay should still succeed because it doesn't re-plan.
    sup_no_planner = Supervisor(
        planner=asserting_planner,
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _identity_llm}),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    replay = sup_no_planner.replay("orig-no-replan")

    assert replay.status == "ok"
    # The asserting_planner was never invoked.
    assert call_log == ["prompt"]


def test_replay_executes_a_fresh_run_not_reusing_original_state(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    counting = _counting_llm()
    supervisor = Supervisor(
        planner=StaticPlanner(minimal_spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: counting}),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    original = supervisor.run("prompt", run_id="orig-fresh")
    replay = supervisor.replay("orig-fresh")

    # If replay reused original state, both runs would have the same call#.
    assert original.result is not None and replay.result is not None
    assert original.result.state["values"]["draft"] != replay.result.state["values"]["draft"]
    assert "call#1" in original.result.state["values"]["draft"]
    assert "call#2" in replay.result.state["values"]["draft"]


def test_replay_with_custom_new_run_id_is_honored(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    supervisor = Supervisor(
        planner=StaticPlanner(minimal_spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _identity_llm}),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    supervisor.run("prompt", run_id="orig-named")
    replay = supervisor.replay("orig-named", new_run_id="my-replay-1")

    assert replay.run_id == "my-replay-1"
    assert (tmp_path / "my-replay-1" / "spec.json").exists()


def test_replay_response_mentions_both_original_and_new_run_ids(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    supervisor = Supervisor(
        planner=StaticPlanner(minimal_spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _identity_llm}),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    supervisor.run("prompt", run_id="orig-msg")
    replay = supervisor.replay("orig-msg", new_run_id="replay-msg-1")

    assert "orig-msg" in replay.response
    assert "replay-msg-1" in replay.response


# ---------- failure paths ----------


def test_replay_of_unknown_run_id_returns_replay_failed(tmp_path: Path) -> None:
    supervisor = Supervisor(
        planner=StaticPlanner(
            GraphSpec(
                graph_id="x",
                goal="x",
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
        ),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _identity_llm}),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    result = supervisor.replay("never-existed")

    assert result.status == "replay_failed"
    assert result.errors[-1]["stage"] == "replay_load_spec"
    assert "never-existed" in result.response


# ---------- replay of paused spec ----------


def test_replay_of_spec_with_wait_for_event_pauses_fresh(tmp_path: Path) -> None:
    spec = _wait_then_finalize_spec()
    supervisor = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(
            runners={NodeKind.LLM_CALL: _identity_llm},
            checkpointer=MemorySaver(),
        ),
        recorder=FileRecorder(root_dir=tmp_path),
    )

    original = supervisor.run("p", run_id="orig-wait")
    assert original.status == "paused"

    replay = supervisor.replay("orig-wait")

    assert replay.status == "paused"
    assert replay.run_id != "orig-wait"
    # The replay paused on its own, fresh — confirms it didn't inherit the
    # original's already-paused checkpoint.
    assert (tmp_path / replay.run_id / "spec.json").exists()
