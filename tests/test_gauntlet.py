"""gauntlet: the builder/critic loop node.

Layers under test:
- `build_gauntlet_runner` — stop conditions, blind mapping, gap threading,
  budget composition, spend roll-up. Driven with a stub launcher + scripted
  critic (deterministic).
- registry/validator wiring — params, llm-call accounting, reference_key
  provenance.
- an end-to-end parent graph through `LangGraphExecutor` with a real child
  launcher over mock runners.
"""

from __future__ import annotations

import pytest

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.registry import Registry
from app.registry.errors import RegistryValidationError
from app.registry.validator import MAX_DEPTH_CEILING, validate_graph_spec
from app.runtime.gauntlet import (
    CriticVerdict,
    GauntletNoArtifact,
    build_gauntlet_runner,
    make_length_critic,
    make_llm_critic,
)
from app.runtime.subgraph import ChildResult, SubgraphDepthExceeded, make_child_launcher


def _state(*, values=None, graph_depth=0, run_id="root", llm_budget=None, consumed=0):
    metadata = {"run_id": run_id, "graph_depth": graph_depth}
    if llm_budget is not None:
        metadata["budget_max_llm_calls"] = llm_budget
    return {
        "values": {"bar": "REFERENCE", **dict(values or {})},
        "metadata": metadata,
        "counters": {"llm_calls_consumed": consumed},
        "events": [],
        "errors": [],
    }


_PARAMS = {"sub_goal": "write it", "name": "g", "reference_key": "bar"}


def _launcher(artifacts, calls):
    """Stub launcher: returns artifacts[round-1]; records every brief."""

    def launch(sub_goal, *, run_id, graph_depth, parent_run_id, inputs, **extra):
        calls.append(
            {"brief": sub_goal, "run_id": run_id, "depth": graph_depth, **extra}
        )
        text = artifacts[len(calls) - 1]
        return ChildResult(
            values={"artifact": text}, counters={"llm_calls_consumed": 1}, status="ok"
        )

    return launch


def _scripted_critic(verdicts, seen):
    """Critic scripted per round; `verdicts` entries are ("ours"|"bar", gap).

    It unblinds via the artifact text so the test can script "ours wins"
    without knowing the sealed coin flip.
    """

    def critic(a, b, *, goal):
        seen.append({"a": a, "b": b, "goal": goal})
        want, gap = verdicts[len(seen) - 1]
        ours_is_a = a != "REFERENCE"
        winner = "A" if (want == "ours") == ours_is_a else "B"
        return CriticVerdict(winner=winner, gap=gap)

    return critic


# ---------- stop conditions ----------


def test_wins_in_round_one_and_records_sealed_mapping() -> None:
    calls, seen = [], []
    runner = build_gauntlet_runner(
        _launcher(["v1"], calls), _scripted_critic([("ours", "none")], seen)
    )

    out = runner(_state(), _PARAMS)

    result = out["result"]
    assert result["stop_reason"] == "won"
    assert result["winner"] == "ours"
    assert result["artifact"] == "v1"
    assert len(result["rounds"]) == 1
    rnd = result["rounds"][0]
    assert rnd["child_run_id"] == "root__gl_g_r1"
    assert set(rnd["mapping"].values()) == {"ours", "bar"}  # sealed, both sides
    # the critic saw only the two artifacts + goal, blind
    assert {seen[0]["a"], seen[0]["b"]} == {"v1", "REFERENCE"}
    assert seen[0]["goal"] == "write it"
    # builder spend (1) + critic call (1) rolled up
    assert out["__spend__"]["llm_calls_consumed"] == 2


def test_loses_every_round_threads_gap_and_returns_best_so_far() -> None:
    calls, seen = [], []
    runner = build_gauntlet_runner(
        _launcher(["v1", "v2", "v3"], calls),
        _scripted_critic(
            [
                ("bar", "missing citations"),
                ("bar", "too short"),
                ("bar", "no examples"),
            ],
            seen,
        ),
    )

    out = runner(_state(), _PARAMS)

    result = out["result"]
    assert result["stop_reason"] == "max_rounds"
    assert result["winner"] == "bar"
    assert result["artifact"] == "v3"
    assert result["gap"] == "no examples"
    assert [r["winner"] for r in result["rounds"]] == ["bar", "bar", "bar"]
    # round 2's brief carries round 1's gap and artifact; round 1's carries none
    assert "Largest gap" not in calls[0]["brief"]
    assert "missing citations" in calls[1]["brief"]
    assert "v1" in calls[1]["brief"]
    assert "too short" in calls[2]["brief"]
    assert out["__spend__"]["llm_calls_consumed"] == 6


def test_same_gap_three_times_stops_early_and_asks_builder_to_measure() -> None:
    calls, seen = [], []
    gap = "the introduction is missing a thesis statement"
    runner = build_gauntlet_runner(
        _launcher(["v1", "v2", "v3", "v4", "v5"], calls),
        _scripted_critic([("bar", gap)] * 5, seen),
    )

    out = runner(_state(), {**_PARAMS, "max_rounds": 5})

    assert out["result"]["stop_reason"] == "repeated_gap"
    assert len(out["result"]["rounds"]) == 3
    # second occurrence: measure-don't-comply instruction in the brief
    assert "Measure it" not in calls[1]["brief"]
    assert "Measure it" in calls[2]["brief"]


def test_stops_on_budget_before_launching_an_unaffordable_round() -> None:
    calls, seen = [], []
    runner = build_gauntlet_runner(
        _launcher(["v1", "v2"], calls), _scripted_critic([("bar", "x")] * 2, seen)
    )
    # 3 calls granted: round 1 = builder(1) + critic(1); 1 left < 2 -> stop
    out = runner(_state(llm_budget=5, consumed=2), _PARAMS)

    assert out["result"]["stop_reason"] == "budget"
    assert len(calls) == 1
    assert calls[0]["max_llm_calls"] == 2  # remaining 3 minus one for the critic
    assert out["result"]["artifact"] == "v1"


def test_no_budget_at_all_raises_no_artifact() -> None:
    runner = build_gauntlet_runner(
        _launcher(["v1"], []), _scripted_critic([("ours", "x")], [])
    )
    with pytest.raises(GauntletNoArtifact):
        runner(_state(llm_budget=1), _PARAMS)


def test_builder_failure_keeps_best_so_far() -> None:
    calls, seen = [], []

    def launch(sub_goal, *, run_id, graph_depth, parent_run_id, inputs, **extra):
        calls.append(run_id)
        if len(calls) == 2:
            return ChildResult(values={}, counters={}, status="failed", response="boom")
        return ChildResult(values={"artifact": "v1"}, counters={}, status="ok")

    runner = build_gauntlet_runner(launch, _scripted_critic([("bar", "x")] * 2, seen))
    out = runner(_state(), _PARAMS)

    assert out["result"]["stop_reason"] == "builder_failed"
    assert out["result"]["artifact"] == "v1"
    assert out["result"]["rounds"][-1]["builder"] == "failed"


def test_builder_exception_keeps_latest_artifact() -> None:
    calls, seen = [], []

    def launch(sub_goal, *, run_id, graph_depth, parent_run_id, inputs, **extra):
        calls.append(run_id)
        if len(calls) == 2:
            raise RuntimeError("launch blew up")
        return ChildResult(values={"artifact": "v1"}, counters={}, status="ok")

    runner = build_gauntlet_runner(launch, _scripted_critic([("bar", "x")] * 2, seen))
    out = runner(_state(), _PARAMS)

    assert out["result"]["stop_reason"] == "builder_failed"
    assert out["result"]["artifact"] == "v1"


def test_critic_exception_keeps_latest_artifact() -> None:
    calls = []

    def critic(a, b, *, goal):
        raise RuntimeError("critic blew up")

    runner = build_gauntlet_runner(_launcher(["v1", "v2"], calls), critic)
    out = runner(_state(), _PARAMS)

    assert out["result"]["stop_reason"] == "critic_failed"
    assert out["result"]["artifact"] == "v1"
    assert out["__spend__"]["llm_calls_consumed"] == 2  # builder + billed critic


def test_builder_exception_reports_billed_planner_call() -> None:
    calls, seen = [], []

    def launch(sub_goal, *, run_id, graph_depth, parent_run_id, inputs, **extra):
        calls.append(run_id)
        if len(calls) == 2:
            exc = RuntimeError("child spec invalid")
            exc.llm_calls_spent = 1  # what make_child_launcher attaches
            raise exc
        return ChildResult(values={"artifact": "v1"}, counters={}, status="ok")

    runner = build_gauntlet_runner(launch, _scripted_critic([("bar", "x")] * 2, seen))
    out = runner(_state(), _PARAMS)

    assert out["result"]["stop_reason"] == "builder_failed"
    assert out["__spend__"]["llm_calls_consumed"] == 2  # critic round 1 + planner


def test_no_artifact_error_carries_billed_spend() -> None:
    def launch(*a, **k):
        return ChildResult(
            values={},
            counters={"llm_calls_consumed": 2, "nodes_executed": 3},
            status="failed",
            response="boom",
        )

    runner = build_gauntlet_runner(launch, _scripted_critic([], []))
    with pytest.raises(GauntletNoArtifact) as exc:
        runner(_state(), _PARAMS)

    assert exc.value.llm_calls_spent == 2
    assert exc.value.nodes_spent == 3


def test_first_round_exception_still_raises() -> None:
    def launch(*a, **k):
        raise RuntimeError("nothing to salvage")

    runner = build_gauntlet_runner(launch, _scripted_critic([], []))
    with pytest.raises(RuntimeError):
        runner(_state(), _PARAMS)


def test_deadline_stops_before_a_round() -> None:
    calls = []
    runner = build_gauntlet_runner(
        _launcher(["v1", "v2"], calls),
        _scripted_critic([("bar", "x")] * 2, []),
        clock=lambda: 100.0,
    )
    state = _state()
    state["metadata"]["wall_deadline_monotonic"] = 50.0
    with pytest.raises(GauntletNoArtifact):
        runner(state, _PARAMS)
    assert calls == []


def test_depth_ceiling_enforced_before_launch() -> None:
    def launcher(*a, **k):
        raise AssertionError("must not launch")

    runner = build_gauntlet_runner(launcher, make_length_critic())
    with pytest.raises(SubgraphDepthExceeded):
        runner(_state(graph_depth=MAX_DEPTH_CEILING), _PARAMS)


def test_replay_pins_child_run_ids_and_mapping() -> None:
    calls = []
    runner = build_gauntlet_runner(
        _launcher(["v1"], calls), _scripted_critic([("ours", "x")], [])
    )
    state = _state(run_id="Y")
    state["metadata"]["replay_of"] = "X"
    out = runner(state, _PARAMS)

    assert calls[0]["run_id"] == "Y__gl_g_r1"
    assert calls[0]["replay_of"] == "X__gl_g_r1"
    # same lineage seed -> same sealed mapping as the original run
    original = build_gauntlet_runner(
        _launcher(["v1"], []), _scripted_critic([("ours", "x")], [])
    )(_state(run_id="X"), _PARAMS)
    assert (
        out["result"]["rounds"][0]["mapping"]
        == original["result"]["rounds"][0]["mapping"]
    )


# ---------- critics ----------


def test_llm_critic_unblinds_correctly() -> None:
    from app.runtime.gauntlet import _CriticPayload

    class Model:
        def invoke(self, messages, **_):
            text = messages[1].content
            # pick whichever side holds "OURS"
            a_block = text.split("=== Artifact A ===")[1].split("=== Artifact B ===")[0]
            return _CriticPayload(winner="A" if "OURS" in a_block else "B", gap="g")

    calls = []

    def launch(sub_goal, *, run_id, graph_depth, parent_run_id, inputs, **extra):
        calls.append(run_id)
        return ChildResult(values={"artifact": "OURS"}, counters={}, status="ok")

    out = build_gauntlet_runner(launch, make_llm_critic(Model()))(_state(), _PARAMS)
    assert out["result"]["winner"] == "ours"


def test_length_critic_is_deterministic() -> None:
    critic = make_length_critic()
    assert critic("longer text", "short", goal="g").winner == "A"
    assert critic("short", "longer text", goal="g").winner == "B"


# ---------- registry / validator ----------


def _gauntlet_spec(*, inputs=("bar",)) -> GraphSpec:
    return GraphSpec(
        graph_id="g",
        goal="g",
        nodes=[
            NodeSpec(
                id="fetch",
                kind=NodeKind.LLM_CALL,
                outputs=["bar"],
                params={"instruction": "REFERENCE"},
            ),
            NodeSpec(
                id="loop",
                kind=NodeKind.GAUNTLET,
                inputs=list(inputs),
                outputs=["final"],
                params={
                    "sub_goal": "write a long one",
                    "name": "g",
                    "reference_key": "bar",
                },
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "fetch"}),
            EdgeSpec.model_validate({"from": "fetch", "to": "loop"}),
            EdgeSpec.model_validate({"from": "loop", "to": "END"}),
        ],
    )


def test_validator_requires_reference_key_in_inputs() -> None:
    with pytest.raises(RegistryValidationError) as exc:
        validate_graph_spec(_gauntlet_spec(inputs=()))
    assert any(i.code == "missing_reference_key_in_inputs" for i in exc.value.issues)
    validate_graph_spec(_gauntlet_spec())  # declared -> passes


def test_registry_counts_gauntlet_as_static_floor_not_runtime_call() -> None:
    reg = Registry()
    node = _gauntlet_spec().nodes[1]
    assert reg.count_llm_calls([node]) == 1  # planner floor
    assert reg.counts_as_llm_call_for_node(node) is False  # spend rolls up


# ---------- end to end ----------


def test_parent_graph_runs_gauntlet_with_real_child_launcher() -> None:
    from app.runtime.executor import LangGraphExecutor

    child = GraphSpec(
        graph_id="child",
        goal="child",
        nodes=[
            NodeSpec(
                id="write",
                kind=NodeKind.LLM_CALL,
                outputs=["artifact"],
                params={"instruction": "a much longer artifact than the reference"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "write"}),
            EdgeSpec.model_validate({"from": "write", "to": "END"}),
        ],
    )

    def echo(state, params):
        del state
        return {"result": params["instruction"]}

    runners = {NodeKind.LLM_CALL: echo}
    executor = LangGraphExecutor(runners=runners)
    launcher = make_child_launcher(planner=lambda goal: child, executor=executor)
    runners[NodeKind.GAUNTLET] = build_gauntlet_runner(
        launcher, make_length_critic(), ledger_registry=executor.ledgers
    )

    spec = validate_graph_spec(_gauntlet_spec())
    result = executor.execute(executor.compile(spec), run_id="parent")

    assert result.ok, result.error
    final = result.state["values"]["final"]
    assert final["stop_reason"] == "won"  # longer artifact beats "REFERENCE"
    assert final["artifact"] == "a much longer artifact than the reference"
    counters = result.state["counters"]
    # fetch(1) + child write(1) + critic(1); parent gauntlet node itself is not a call
    assert counters["llm_calls_consumed"] == 3
    assert counters["nodes_executed"] == 3  # fetch, loop, child write
