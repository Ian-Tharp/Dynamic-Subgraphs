"""Tests for dynamic_subgraphs/bench.py — 3-arm benchmark harness (PR8b)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from app.models import GraphSpec, NodeKind
from dynamic_subgraphs import (
    DeterministicEvalGate,
    DynamicSubgraphs,
    EngineConfig,
    EvalReference,
    Model,
    Recording,
)
from dynamic_subgraphs.bench import (
    ARMS,
    BenchIntegrityError,
    BenchTask,
    RouterLibrary,
    fill_spec,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Paths to the real authored specs
# ---------------------------------------------------------------------------

_LIBRARY_DIR = Path("docs/evals/router-library-v1")
_SINGLE_FIXED = Path("docs/evals/fixed-host-graph-v1.json")


# ---------------------------------------------------------------------------
# Helper: a minimal EvalReference with grounding_required=False
# ---------------------------------------------------------------------------


def _ref(checklist: list[str] | None = None) -> EvalReference:
    return EvalReference(
        checklist=checklist or [],
        grounding_required=False,
    )


# ---------------------------------------------------------------------------
# test_fill_spec_replaces_everywhere
# ---------------------------------------------------------------------------


def test_fill_spec_replaces_everywhere() -> None:
    """fill_spec must substitute in BOTH instruction AND tool_call args."""
    research_path = _LIBRARY_DIR / "research.json"
    spec = GraphSpec.model_validate(
        json.loads(research_path.read_text(encoding="utf-8"))
    )
    # A prompt that would break str.format (braces, quotes, backslash).
    prompt = 'find {x} "quoted" C:\\path stuff'

    filled = fill_spec(spec, prompt)

    # No literal placeholder should remain in any node's params (instructions
    # and tool args).  The spec's rationale field intentionally contains the
    # literal string "{prompt}" as documentation text, so we check params only.
    for node in filled.nodes:
        params_json = json.dumps(node.params)
        assert "{prompt}" not in params_json, (
            f"Node {node.id!r} still has a {{prompt}} placeholder in params"
        )

    # The prompt must land verbatim in at least one tool_call args string.
    found_in_args = False
    for node in filled.nodes:
        if node.kind == NodeKind.TOOL_CALL:
            args = node.params.get("args", {})
            if isinstance(args, dict):
                for v in args.values():
                    if isinstance(v, str) and prompt in v:
                        found_in_args = True
    assert found_in_args, "Prompt not found verbatim in any tool_call args"

    # The original spec's params must be UNCHANGED.
    for node in spec.nodes:
        params_json = json.dumps(node.params)
        if node.kind in (NodeKind.TOOL_CALL, NodeKind.LLM_CALL):
            # At least one node must still have the original placeholder.
            break
    original_node_params_json = json.dumps([n.params for n in spec.nodes])
    assert "{prompt}" in original_node_params_json, (
        "Original spec params were mutated (deep copy failed)"
    )


# ---------------------------------------------------------------------------
# test_fill_spec_literal_not_format
# ---------------------------------------------------------------------------


def test_fill_spec_literal_not_format() -> None:
    """Prompts containing braces must not raise and must land verbatim."""
    compare_path = _LIBRARY_DIR / "compare.json"
    spec = GraphSpec.model_validate(
        json.loads(compare_path.read_text(encoding="utf-8"))
    )
    prompt = "what is {undefined} and {also_undefined}?"

    # Must NOT raise (str.format would raise KeyError).
    filled = fill_spec(spec, prompt)

    # No {prompt} placeholder should remain in any node params.
    for node in filled.nodes:
        params_json = json.dumps(node.params)
        assert "{prompt}" not in params_json

    # The original braces from the user prompt must survive unaltered in params.
    all_params_json = json.dumps([n.params for n in filled.nodes])
    assert "{undefined}" in all_params_json


# ---------------------------------------------------------------------------
# test_router_routes_by_task_type_with_fallback
# ---------------------------------------------------------------------------


def test_router_routes_by_task_type_with_fallback() -> None:
    """RouterLibrary.from_dir: known type -> correct spec; unknown -> summarize."""
    lib = RouterLibrary.from_dir(_LIBRARY_DIR, _SINGLE_FIXED)

    compare_task = BenchTask(
        task_id="t-compare-1",
        task_type="compare",
        prompt="Compare A and B.",
        reference=_ref(),
    )
    unknown_task = BenchTask(
        task_id="t-unknown-1",
        task_type="totally_unknown_type",
        prompt="Do something weird.",
        reference=_ref(),
    )

    routed_compare = lib.route(compare_task)
    assert routed_compare.graph_id == "router-compare-v1"

    routed_unknown = lib.route(unknown_task)
    # Fallback must be the summarize spec.
    assert routed_unknown.graph_id == "router-summarize-v1"


# ---------------------------------------------------------------------------
# test_run_benchmark_offline
# ---------------------------------------------------------------------------


def test_run_benchmark_offline(tmp_path: Path) -> None:
    """2 tasks x 3 arms x 2 repeats = 12 rows; files written correctly."""
    lib = RouterLibrary.from_dir(_LIBRARY_DIR, _SINGLE_FIXED)
    tasks = [
        BenchTask(
            task_id="t-compare",
            task_type="compare",
            prompt="Compare apples and oranges.",
            reference=_ref(["apples", "oranges"]),
        ),
        BenchTask(
            task_id="t-summarize",
            task_type="summarize",
            prompt="Summarize this text: the quick brown fox.",
            reference=_ref(["fox"]),
        ),
    ]

    report = run_benchmark(
        tasks=tasks,
        repeats=2,
        runs_dir=tmp_path,
        router_library=lib,
        bench_id="test-offline",
        planner="mock",
    )

    # Row count: 2 tasks x 3 arms x 2 repeats = 12.
    assert len(report.rows) == 12

    # Every arm must be represented.
    assert {row.arm for row in report.rows} == set(ARMS)

    # Every row must have quality and value_per_ktok populated.
    for row in report.rows:
        assert isinstance(row.quality, float)
        # mock runs produce 0 tokens -> value_per_ktok is None (0/0 = None),
        # which is acceptable; what we cannot have is a missing quality field.
        assert row.quality >= 0.0

    # bench.jsonl must exist and have 12 lines.
    bench_jsonl = tmp_path / "bench" / "test-offline" / "bench.jsonl"
    assert bench_jsonl.exists()
    lines = [ln for ln in bench_jsonl.read_text("utf-8").splitlines() if ln.strip()]
    assert len(lines) == 12

    # report.json must have bench_id and pilot.per_task_cv keys.
    report_json = tmp_path / "bench" / "test-offline" / "report.json"
    payload = json.loads(report_json.read_text("utf-8"))
    assert payload["bench_id"] == "test-offline"
    assert "per_task_cv" in payload["pilot"]

    # Thinness stats: every per-task entry exposes samples + dropped_none so
    # a thin CV (e.g. mock runs with 0 tokens -> all None vpktok) is visible.
    per_task_cv = payload["pilot"]["per_task_cv"]
    assert set(per_task_cv) == {"t-compare", "t-summarize"}
    for entry in per_task_cv.values():
        assert "samples" in entry
        assert "dropped_none" in entry
        assert entry["samples"] + entry["dropped_none"] == 2  # repeats=2


# ---------------------------------------------------------------------------
# test_arms_share_applicable_dimensions  (A1 invariant)
# ---------------------------------------------------------------------------


def test_arms_share_applicable_dimensions(tmp_path: Path) -> None:
    """1 task x 3 arms x 1 repeat: applicable_dimensions must be identical."""
    lib = RouterLibrary.from_dir(_LIBRARY_DIR, _SINGLE_FIXED)
    tasks = [
        BenchTask(
            task_id="t-a1",
            task_type="summarize",
            prompt="Summarize the solar system.",
            reference=_ref(),
        )
    ]

    report = run_benchmark(
        tasks=tasks,
        repeats=1,
        runs_dir=tmp_path,
        router_library=lib,
        bench_id="test-a1",
        planner="mock",
    )

    assert len(report.rows) == 3

    # Collect sorted applicable_dimensions tuples across arms.
    dim_sets = {tuple(sorted(row.applicable_dimensions)) for row in report.rows}

    # The reference_only gate mode must give all arms the same applicable set.
    assert len(dim_sets) == 1, f"applicable_dimensions differ across arms: {dim_sets}"


# ---------------------------------------------------------------------------
# test_plan_failure_synthesizes_zero_row
# ---------------------------------------------------------------------------


def test_plan_failure_synthesizes_zero_row(tmp_path: Path) -> None:
    """A plan/validation failure must produce quality=0.0 / floor=False row.

    Choice: monkeypatch DynamicSubgraphs.run to return a RunResult with
    status='plan_failed' and eval=None.  This is the cheaper honest path —
    it tests the bench harness's zero-synthesis logic directly without having
    to stand up a fake provider whose planner raises (which would test the
    engine's exception path instead).  The test_sdk_eval fake-provider pattern
    tests the engine; here we test what the *harness* does when it receives a
    failure result.
    """
    from dynamic_subgraphs.engine import RunResult
    from dynamic_subgraphs.usage import TokenUsage

    lib = RouterLibrary.from_dir(_LIBRARY_DIR, _SINGLE_FIXED)
    task = BenchTask(
        task_id="t-fail",
        task_type="compare",
        prompt="Compare nothing with nothing.",
        reference=_ref(),
    )

    failed_result = RunResult(
        run_id="fake-run-plan-fail",
        status="plan_failed",
        response=None,
        plan=None,
        eval=None,
        usage=TokenUsage(),
    )

    with patch.object(DynamicSubgraphs, "run", return_value=failed_result):
        report = run_benchmark(
            tasks=[task],
            repeats=1,
            runs_dir=tmp_path,
            router_library=lib,
            bench_id="test-planfail",
            planner="mock",
        )

    assert len(report.rows) == 3
    for row in report.rows:
        assert row.quality == 0.0
        assert row.quality_floor_met is False


# ---------------------------------------------------------------------------
# test_ok_run_without_eval_raises_integrity_error
# ---------------------------------------------------------------------------


def test_ok_run_without_eval_raises_integrity_error(tmp_path: Path) -> None:
    """status=='ok' with eval=None is an infrastructure failure — abort-fast.

    The gate threw on a successful (money-spending) run and the engine
    swallowed+logged it. Booking that as quality-0 would silently bias the
    verdict toward KILL, so the harness must raise BenchIntegrityError
    instead of synthesizing a zero row.
    """
    from dynamic_subgraphs.engine import RunResult
    from dynamic_subgraphs.usage import TokenUsage

    lib = RouterLibrary.from_dir(_LIBRARY_DIR, _SINGLE_FIXED)
    task = BenchTask(
        task_id="t-gate-broke",
        task_type="compare",
        prompt="Compare X and Y.",
        reference=_ref(),
    )

    ok_but_unscored = RunResult(
        run_id="fake-run-ok-no-eval",
        status="ok",
        response="a fine answer",
        plan=None,
        eval=None,
        usage=TokenUsage(),
    )

    with (
        patch.object(DynamicSubgraphs, "run", return_value=ok_but_unscored),
        pytest.raises(BenchIntegrityError, match="no EvalResult"),
    ):
        run_benchmark(
            tasks=[task],
            repeats=1,
            runs_dir=tmp_path,
            router_library=lib,
            bench_id="test-gate-broke",
            planner="mock",
        )


# ---------------------------------------------------------------------------
# test_bench_id_reuse_refused
# ---------------------------------------------------------------------------


def test_bench_id_reuse_refused(tmp_path: Path) -> None:
    """A pre-existing non-empty bench dir must be refused BEFORE any run."""
    lib = RouterLibrary.from_dir(_LIBRARY_DIR, _SINGLE_FIXED)
    task = BenchTask(
        task_id="t-reuse",
        task_type="summarize",
        prompt="Summarize the moon.",
        reference=_ref(),
    )

    # Pre-create the bench dir with a stale file from a "prior attempt".
    stale_dir = tmp_path / "bench" / "test-reused-id"
    stale_dir.mkdir(parents=True)
    (stale_dir / "bench.jsonl").write_text("{}\n", encoding="utf-8")

    # The guard must fire before any engine.run call — patch run to explode
    # so the test fails loudly if the ordering ever regresses.
    def _must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("engine.run was called despite the reuse guard")

    with (
        patch.object(DynamicSubgraphs, "run", side_effect=_must_not_run),
        pytest.raises(BenchIntegrityError, match="unique per run"),
    ):
        run_benchmark(
            tasks=[task],
            repeats=1,
            runs_dir=tmp_path,
            router_library=lib,
            bench_id="test-reused-id",
            planner="mock",
        )


# ---------------------------------------------------------------------------
# test_prompt_visibility_on_real_runner_path
# ---------------------------------------------------------------------------


def test_prompt_visibility_on_real_runner_path(tmp_path: Path) -> None:
    """THE prompt-visibility test: the task sentence must reach LLM workers.

    Uses the fake-provider seam from test_sdk_eval: a provider whose
    build_chat returns a capturing fake chat that records every messages list
    it's invoked with and returns a canned response.  fill_spec is applied to
    the summarize spec (which has a {prompt} placeholder in the llm_call
    instruction).  Since fixed_spec short-circuits planning (StaticPlanner),
    the engine uses planner="llm" config but bypasses the LLM planner call —
    the capturing chat only sees worker invocations.  We assert the task
    sentence appears in at least one captured HumanMessage content.
    """
    from app.runtime import ProviderRegistry

    task_sentence = "Summarize the water cycle in three sentences."

    # Load and fill the summarize spec so {prompt} is replaced.
    summarize_spec = GraphSpec.model_validate(
        json.loads((_LIBRARY_DIR / "summarize.json").read_text(encoding="utf-8"))
    )
    filled_spec = fill_spec(summarize_spec, task_sentence)

    # A capturing fake chat: records all messages lists, returns canned reply.
    captured_messages: list[list[Any]] = []

    class _FakeAIMessage:
        content: str = "mock summary answer"

    class _CapturingChat:
        def invoke(self, messages: list[Any], /, **kwargs: Any) -> _FakeAIMessage:
            captured_messages.append(list(messages))
            return _FakeAIMessage()

    # A fake planner (build_structured_output) that returns the pre-filled spec
    # — this path is not exercised because fixed_spec short-circuits planning.
    class _FakePlanner:
        def invoke(self, messages: list[Any], /, **kwargs: Any) -> GraphSpec:
            return filled_spec

    class _FakeProvider:
        name = "fake-vis"

        def required_env_vars(self) -> tuple[str, ...]:
            return ()

        def build_chat(self, ref: Any) -> _CapturingChat:
            return _CapturingChat()

        def build_structured_output(self, ref: Any, schema: Any) -> _FakePlanner:
            return _FakePlanner()

    registry = ProviderRegistry()
    registry.register(_FakeProvider())

    engine = DynamicSubgraphs(
        EngineConfig(
            model=Model("fake-vis", "fake-model"),
            planner="llm",
            providers=registry,
            runs_dir=tmp_path,
            eval_gate=DeterministicEvalGate(),
        )
    )

    result = engine.run(
        task_sentence,
        fixed_spec=filled_spec,
        record=Recording.none(),
    )

    # The run must complete (the fake chat returns a valid response).
    assert result.ok, f"Run failed: {result.status} {result.errors}"

    # The task sentence must appear in a HumanMessage specifically — system
    # prompts don't count; the *user-facing* message must carry the task.
    found = False
    for msg_list in captured_messages:
        for msg in msg_list:
            if not isinstance(msg, HumanMessage):
                continue
            content = msg.content
            if isinstance(content, str) and task_sentence in content:
                found = True
                break
        if found:
            break

    assert found, (
        f"Task sentence not found in any worker HumanMessage.\n"
        f"Captured message counts: {[len(m) for m in captured_messages]}\n"
        f"First message contents: "
        f"{[getattr(m, 'content', repr(m))[:120] for m in (captured_messages[0] if captured_messages else [])]}"
    )
