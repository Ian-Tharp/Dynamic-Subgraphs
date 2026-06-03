"""Supervisor meta-loop coverage."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from app.models import GraphSpec, NodeKind
from app.recording import FileRecorder
from app.runtime import LangGraphExecutor
from app.supervisor import (
    IterationContext,
    IterationDecision,
    LlmIterationDecider,
    StaticPlanner,
    Supervisor,
)
from app.supervisor.iteration import _IterationDecisionPayload


def _mock_llm(state, params):
    del state
    return {"result": params["instruction"]}


def _make_supervisor(
    spec: GraphSpec,
    tmp_path: Path,
    *,
    planner: Any | None = None,
) -> Supervisor:
    return Supervisor(
        planner=planner or StaticPlanner(spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm}),
        recorder=FileRecorder(root_dir=tmp_path),
    )


def _single_llm_spec(
    make_node, make_edge, *, graph_id: str, output_key: str
) -> GraphSpec:
    return GraphSpec(
        graph_id=graph_id,
        goal=f"produce {output_key}",
        nodes=[
            make_node(
                "step",
                outputs=[output_key],
                params={"instruction": f"value for {output_key}"},
            )
        ],
        edges=[make_edge("START", "step"), make_edge("step", "END")],
    )


def test_run_iteratively_stops_after_success_by_default(
    minimal_spec: GraphSpec,
    tmp_path: Path,
) -> None:
    sup = _make_supervisor(minimal_spec, tmp_path)

    result = sup.run_iteratively(
        "produce a draft",
        run_id="chain-default",
        max_iterations=3,
    )

    assert result.status == "ok"
    assert len(result.steps) == 1
    assert result.steps[0].run_id == "chain-default_iter_1"
    assert result.final_result is not None
    assert result.final_result.status == "ok"
    assert (tmp_path / "chain-default_iter_1" / "spec.json").exists()


def test_run_iteratively_replans_with_explicit_next_prompt(
    make_node,
    make_edge,
    tmp_path: Path,
) -> None:
    first_spec = _single_llm_spec(
        make_node, make_edge, graph_id="first-spec", output_key="first_output"
    )
    second_spec = _single_llm_spec(
        make_node, make_edge, graph_id="second-spec", output_key="final_output"
    )
    planned_prompts: list[str] = []

    def planner(prompt: str) -> GraphSpec:
        planned_prompts.append(prompt)
        return first_spec if len(planned_prompts) == 1 else second_spec

    def decider(context: IterationContext) -> IterationDecision:
        if context.iteration == 1:
            return IterationDecision(
                action="replan",
                reason="Need a second bounded attempt.",
                gaps=["missing final output"],
                next_prompt="second bounded attempt",
            )
        return IterationDecision(
            action="stop",
            reason="Final output exists.",
            success_criteria_met=True,
        )

    sup = _make_supervisor(first_spec, tmp_path, planner=planner)

    result = sup.run_iteratively(
        "original task",
        run_id="chain-replan",
        max_iterations=2,
        decider=decider,
    )

    assert result.status == "ok"
    assert planned_prompts == ["original task", "second bounded attempt"]
    assert [step.run_id for step in result.steps] == [
        "chain-replan_iter_1",
        "chain-replan_iter_2",
    ]
    assert result.final_result is not None
    assert result.final_result.validated_spec is not None
    assert result.final_result.validated_spec.graph_id == "second-spec"
    assert result.final_result.result is not None
    assert result.final_result.result.state["values"]["final_output"] == (
        "value for final_output"
    )


def test_run_iteratively_builds_compact_replan_prompt_when_missing(
    make_node,
    make_edge,
    tmp_path: Path,
) -> None:
    first_spec = _single_llm_spec(
        make_node, make_edge, graph_id="first-spec", output_key="first_output"
    )
    second_spec = _single_llm_spec(
        make_node, make_edge, graph_id="second-spec", output_key="final_output"
    )
    planned_prompts: list[str] = []

    def planner(prompt: str) -> GraphSpec:
        planned_prompts.append(prompt)
        return first_spec if len(planned_prompts) == 1 else second_spec

    def decider(context: IterationContext) -> IterationDecision:
        if context.iteration == 1:
            return IterationDecision(
                action="replan",
                reason="Need to address a gap.",
                gaps=["first_output was not enough"],
            )
        return IterationDecision(
            action="stop",
            reason="Done.",
            success_criteria_met=True,
        )

    sup = _make_supervisor(first_spec, tmp_path, planner=planner)

    result = sup.run_iteratively(
        "original task",
        run_id="chain-auto-prompt",
        max_iterations=2,
        decider=decider,
    )

    assert result.status == "ok"
    assert len(planned_prompts) == 2
    assert "Original user prompt:" in planned_prompts[1]
    assert "original task" in planned_prompts[1]
    assert "Previous iteration result:" in planned_prompts[1]
    assert "first_output" in planned_prompts[1]
    assert "first_output was not enough" in planned_prompts[1]


def test_run_iteratively_can_ask_user(
    minimal_spec: GraphSpec,
    tmp_path: Path,
) -> None:
    sup = _make_supervisor(minimal_spec, tmp_path)

    def decider(context: IterationContext) -> IterationDecision:
        del context
        return IterationDecision(
            action="ask_user",
            reason="The task is ambiguous.",
            question_to_user="Which source should I use?",
        )

    result = sup.run_iteratively(
        "ambiguous task",
        run_id="chain-ask",
        decider=decider,
    )

    assert result.status == "ask_user"
    assert result.response == "Which source should I use?"
    assert len(result.steps) == 1
    assert result.final_decision is not None
    assert result.final_decision.action == "ask_user"


# ---------- LlmIterationDecider ----------


class _FakeJudgeModel:
    """Mimics a `with_structured_output(_IterationDecisionPayload)` runnable."""

    def __init__(self, responses: Iterable[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list, /, **kwargs: Any) -> Any:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("_FakeJudgeModel ran out of scripted responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _ok_supervisor_context(make_node, make_edge, tmp_path):
    """Build a tiny supervisor + run one iteration, returning the context the decider sees."""
    from app.supervisor.iteration import IterationContext

    spec = _single_llm_spec(
        make_node, make_edge, graph_id="judge-spec", output_key="answer"
    )
    sup = _make_supervisor(spec, tmp_path)
    result = sup.run("explain dynamic subgraphs", run_id="judge-iter-1")
    return IterationContext(
        original_prompt="explain dynamic subgraphs",
        current_prompt="explain dynamic subgraphs",
        iteration=1,
        max_iterations=3,
        result=result,
        previous_steps=(),
    )


def test_llm_decider_converts_stop_payload_to_decision(
    make_node, make_edge, tmp_path: Path
) -> None:
    model = _FakeJudgeModel(
        [
            _IterationDecisionPayload(
                action="stop",
                reason="Output addresses the prompt.",
                success_criteria_met=True,
            )
        ]
    )
    decider = LlmIterationDecider(model)
    context = _ok_supervisor_context(make_node, make_edge, tmp_path)

    decision = decider(context)

    assert decision.action == "stop"
    assert decision.success_criteria_met is True
    assert "addresses" in decision.reason


def test_llm_decider_converts_replan_payload_with_gaps_and_prompt(
    make_node, make_edge, tmp_path: Path
) -> None:
    model = _FakeJudgeModel(
        [
            _IterationDecisionPayload(
                action="replan",
                reason="The explanation lacks concrete examples.",
                gaps=["missing examples", "no mention of state envelope"],
                next_prompt="Revise and include two concrete examples.",
            )
        ]
    )
    decider = LlmIterationDecider(model)
    context = _ok_supervisor_context(make_node, make_edge, tmp_path)

    decision = decider(context)

    assert decision.action == "replan"
    assert decision.gaps == ("missing examples", "no mention of state envelope")
    assert decision.next_prompt == "Revise and include two concrete examples."


def test_llm_decider_converts_ask_user_payload(
    make_node, make_edge, tmp_path: Path
) -> None:
    model = _FakeJudgeModel(
        [
            _IterationDecisionPayload(
                action="ask_user",
                reason="Ambiguous: dynamic subgraphs in math or in this project?",
                question_to_user="Did you mean dynamic subgraphs in graph theory, or in this project's runtime?",
            )
        ]
    )
    decider = LlmIterationDecider(model)
    context = _ok_supervisor_context(make_node, make_edge, tmp_path)

    decision = decider(context)

    assert decision.action == "ask_user"
    assert "graph theory" in decision.question_to_user


def test_llm_decider_includes_original_prompt_and_outputs_in_eval_message(
    make_node, make_edge, tmp_path: Path
) -> None:
    model = _FakeJudgeModel(
        [
            _IterationDecisionPayload(
                action="stop", reason="ok", success_criteria_met=True
            )
        ]
    )
    decider = LlmIterationDecider(model)
    context = _ok_supervisor_context(make_node, make_edge, tmp_path)

    decider(context)

    user_msg = model.calls[0][1].content
    assert "explain dynamic subgraphs" in user_msg
    assert "answer" in user_msg  # the output key was rendered
    assert "Iteration: 1 of 3" in user_msg


def test_llm_decider_includes_explicit_success_criteria_when_provided(
    make_node, make_edge, tmp_path: Path
) -> None:
    model = _FakeJudgeModel(
        [
            _IterationDecisionPayload(
                action="stop", reason="ok", success_criteria_met=True
            )
        ]
    )
    decider = LlmIterationDecider(
        model,
        success_criteria="The answer must mention 'topology' explicitly.",
    )
    context = _ok_supervisor_context(make_node, make_edge, tmp_path)

    decider(context)

    user_msg = model.calls[0][1].content
    assert "Explicit success criteria:" in user_msg
    assert "mention 'topology'" in user_msg


def test_llm_decider_fails_closed_when_model_raises(
    make_node, make_edge, tmp_path: Path
) -> None:
    model = _FakeJudgeModel([RuntimeError("API down")])
    decider = LlmIterationDecider(model)
    context = _ok_supervisor_context(make_node, make_edge, tmp_path)

    decision = decider(context)

    assert decision.action == "fail"
    assert "API down" in decision.reason


def test_llm_decider_fails_closed_when_model_returns_wrong_type(
    make_node, make_edge, tmp_path: Path
) -> None:
    model = _FakeJudgeModel(["not a payload"])
    decider = LlmIterationDecider(model)
    context = _ok_supervisor_context(make_node, make_edge, tmp_path)

    decision = decider(context)

    assert decision.action == "fail"
    assert "_IterationDecisionPayload" in decision.reason


# ---------- LlmIterationDecider defers obvious cases ----------


def test_llm_decider_defers_paused_runs_to_fallback_without_calling_model(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    """A paused run is unambiguous — the framework handles it. Don't waste tokens."""
    from langgraph.checkpoint.memory import MemorySaver

    from app.models import NodeSpec
    from app.models.graph_spec import EdgeSpec

    spec_with_wait = GraphSpec(
        graph_id="pause-spec",
        goal="pause",
        nodes=[
            NodeSpec(
                id="ask",
                kind=NodeKind.LLM_CALL,
                outputs=["question"],
                params={"instruction": "ask"},
            ),
            NodeSpec(
                id="wait",
                kind=NodeKind.WAIT_FOR_EVENT,
                outputs=["answer"],
                params={"event_type": "human_reply"},
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "ask"}),
            EdgeSpec.model_validate({"from": "ask", "to": "wait"}),
            EdgeSpec.model_validate({"from": "wait", "to": "END"}),
        ],
    )
    sup = Supervisor(
        planner=StaticPlanner(spec_with_wait),
        executor=LangGraphExecutor(
            runners={NodeKind.LLM_CALL: _mock_llm},
            checkpointer=MemorySaver(),
        ),
        recorder=FileRecorder(root_dir=tmp_path),
    )
    result = sup.run("p", run_id="paused-judge")
    assert result.status == "paused"

    model = _FakeJudgeModel([])  # would crash if invoked
    decider = LlmIterationDecider(model)
    context = IterationContext(
        original_prompt="p",
        current_prompt="p",
        iteration=1,
        max_iterations=3,
        result=result,
    )

    decision = decider(context)

    assert decision.action == "ask_user"
    assert model.calls == []  # LLM was not invoked


def test_llm_decider_defers_execution_failed_by_default(
    make_node, make_edge, tmp_path: Path
) -> None:
    def explode(state, params):
        raise RuntimeError("boom")

    spec = _single_llm_spec(
        make_node, make_edge, graph_id="boom-spec", output_key="answer"
    )
    sup = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: explode}),
        recorder=FileRecorder(root_dir=tmp_path),
    )
    result = sup.run("p", run_id="exec-fail")
    assert result.status == "execution_failed"

    model = _FakeJudgeModel([])
    decider = LlmIterationDecider(model)
    context = IterationContext(
        original_prompt="p",
        current_prompt="p",
        iteration=1,
        max_iterations=3,
        result=result,
    )

    decision = decider(context)

    assert decision.action == "fail"
    assert model.calls == []


def test_llm_decider_judges_execution_failed_when_opted_in(
    make_node, make_edge, tmp_path: Path
) -> None:
    def explode(state, params):
        raise RuntimeError("boom")

    spec = _single_llm_spec(
        make_node, make_edge, graph_id="boom-spec-2", output_key="answer"
    )
    sup = Supervisor(
        planner=StaticPlanner(spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: explode}),
        recorder=FileRecorder(root_dir=tmp_path),
    )
    result = sup.run("p", run_id="exec-fail-judged")

    model = _FakeJudgeModel(
        [
            _IterationDecisionPayload(
                action="replan",
                reason="The error looks transient, try again with simpler instruction.",
                gaps=["runner crashed"],
            )
        ]
    )
    decider = LlmIterationDecider(model, judge_failed_runs=True)
    context = IterationContext(
        original_prompt="p",
        current_prompt="p",
        iteration=1,
        max_iterations=3,
        result=result,
    )

    decision = decider(context)

    assert decision.action == "replan"
    assert model.calls != []


# ---------- integration: meta-loop with LlmIterationDecider ----------


def test_meta_loop_with_llm_decider_iterates_then_stops(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    """First decision: replan. Second decision: stop. Loop runs twice."""
    model = _FakeJudgeModel(
        [
            _IterationDecisionPayload(
                action="replan",
                reason="Output is too brief.",
                gaps=["lacks depth"],
            ),
            _IterationDecisionPayload(
                action="stop",
                reason="Output now sufficient.",
                success_criteria_met=True,
            ),
        ]
    )
    sup = _make_supervisor(minimal_spec, tmp_path)

    result = sup.run_iteratively(
        "explain it",
        run_id="meta-llm-chain",
        max_iterations=3,
        decider=LlmIterationDecider(model),
    )

    assert result.status == "ok"
    assert len(result.steps) == 2
    assert result.steps[0].decision.action == "replan"
    assert result.steps[1].decision.action == "stop"
    assert len(model.calls) == 2


# ---------- chain-level recording ----------


def test_run_iteratively_writes_chain_record_to_disk(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    import json

    sup = _make_supervisor(minimal_spec, tmp_path)

    sup.run_iteratively("the task", run_id="chain-record-1", max_iterations=2)

    chain_path = tmp_path / "chain-record-1" / "chain.json"
    summary_path = tmp_path / "chain-record-1" / "chain.md"
    assert chain_path.exists()
    assert summary_path.exists()

    payload = json.loads(chain_path.read_text(encoding="utf-8"))
    assert payload["chain_id"] == "chain-record-1"
    assert payload["original_prompt"] == "the task"
    assert payload["status"] == "ok"
    assert payload["step_count"] == 1
    assert len(payload["steps"]) == 1
    assert payload["steps"][0]["run_id"] == "chain-record-1_iter_1"


def test_chain_record_preserves_gap_progression_across_iterations(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    import json

    def decider(context: IterationContext) -> IterationDecision:
        if context.iteration == 1:
            return IterationDecision(
                action="replan",
                reason="iter 1 needs more",
                gaps=["missing X", "missing Y"],
            )
        return IterationDecision(
            action="stop",
            reason="iter 2 sufficient",
            success_criteria_met=True,
        )

    sup = _make_supervisor(minimal_spec, tmp_path)

    sup.run_iteratively(
        "the task",
        run_id="chain-gaps",
        max_iterations=3,
        decider=decider,
    )

    payload = json.loads(
        (tmp_path / "chain-gaps" / "chain.json").read_text(encoding="utf-8")
    )

    assert payload["step_count"] == 2
    assert payload["steps"][0]["decision"]["action"] == "replan"
    assert payload["steps"][0]["decision"]["gaps"] == ["missing X", "missing Y"]
    assert payload["steps"][1]["decision"]["action"] == "stop"
    assert payload["steps"][1]["decision"]["success_criteria_met"] is True
    assert payload["final_decision"]["action"] == "stop"


def test_chain_record_can_be_loaded_via_recorder(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    from app.recording import FileRecorder

    sup = _make_supervisor(minimal_spec, tmp_path)

    sup.run_iteratively("task", run_id="chain-load", max_iterations=1)

    loaded = FileRecorder(root_dir=tmp_path).load_chain("chain-load")

    assert loaded["chain_id"] == "chain-load"
    assert loaded["original_prompt"] == "task"
    assert isinstance(loaded["steps"], list)


def test_chain_record_load_missing_raises_filenotfound(tmp_path: Path) -> None:
    from app.recording import FileRecorder

    recorder = FileRecorder(root_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        recorder.load_chain("no-such-chain")


def test_chain_record_can_be_disabled(minimal_spec: GraphSpec, tmp_path: Path) -> None:
    sup = _make_supervisor(minimal_spec, tmp_path)

    sup.run_iteratively(
        "the task",
        run_id="chain-disabled",
        max_iterations=1,
        record_chain=False,
    )

    assert not (tmp_path / "chain-disabled" / "chain.json").exists()
    # Per-iteration directory still exists.
    assert (tmp_path / "chain-disabled_iter_1" / "spec.json").exists()


def test_chain_recording_failure_does_not_break_the_returned_result(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    """If the recorder.record_chain raises, the chain result should still come back."""

    class _BrokenChainRecorder:
        def record(self, **kwargs):
            from pathlib import Path

            from app.recording.recorder import RunRecord

            return RunRecord(
                run_id=kwargs["result"].trace.run_id,
                directory=Path("."),
                artifacts={},
            )

        def load_validated_spec(self, run_id):
            raise NotImplementedError

        def record_chain(self, *args, **kwargs):
            raise OSError("disk full")

    sup = Supervisor(
        planner=StaticPlanner(minimal_spec),
        executor=LangGraphExecutor(runners={NodeKind.LLM_CALL: _mock_llm}),
        recorder=_BrokenChainRecorder(),
    )

    result = sup.run_iteratively("task", run_id="chain-broken-record", max_iterations=1)

    assert result.status == "ok"
    assert len(result.steps) == 1


def test_chain_record_includes_status_for_max_iterations_path(
    minimal_spec: GraphSpec, tmp_path: Path
) -> None:
    import json

    def always_replan(context: IterationContext) -> IterationDecision:
        return IterationDecision(
            action="replan",
            reason="keep going",
            gaps=["never satisfied"],
        )

    sup = _make_supervisor(minimal_spec, tmp_path)

    sup.run_iteratively(
        "task", run_id="chain-max-iter", max_iterations=2, decider=always_replan
    )

    payload = json.loads(
        (tmp_path / "chain-max-iter" / "chain.json").read_text(encoding="utf-8")
    )

    assert payload["status"] == "max_iterations"
    assert payload["step_count"] == 2
    # Both iterations should be present.
    assert [s["iteration"] for s in payload["steps"]] == [1, 2]
