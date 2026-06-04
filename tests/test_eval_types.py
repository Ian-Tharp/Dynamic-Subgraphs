"""Types & vocab for the eval/value layer (Slice 7, PR1).

No gate exists yet — these lock the persisted shape, the value-axis math, and
the recording vocabulary, plus the `TokenUsage` relocation that lets the engine
and the eval layer share it without an import cycle.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from dynamic_subgraphs import (
    EvalGate,
    EvalReference,
    EvalResult,
    EvalTags,
    RunFingerprint,
    ScoreComponent,
    TokenUsage,
)
from dynamic_subgraphs.eval import (
    EVAL_SCHEMA_VERSION,
    QUALITY_FLOOR_DEFAULT,
    value_per_ktok,
    value_per_usd,
)

# ---------- value axes ----------


def test_value_per_ktok_is_quality_over_thousand_tokens() -> None:
    assert value_per_ktok(0.8, 4000) == pytest.approx(0.2)


def test_value_per_ktok_none_without_tokens() -> None:
    # Tokens are always known for a real run, but guard the zero case.
    assert value_per_ktok(0.9, 0) is None


def test_value_per_usd_none_when_cost_unknown_or_zero() -> None:
    # The whole point: cost can be None (no price book) or ~0; never divide.
    assert value_per_usd(0.9, None) is None
    assert value_per_usd(0.9, 0.0) is None
    assert value_per_usd(0.9, 0.0009) == pytest.approx(1000.0)


# ---------- EvalResult shape & persistence ----------


def _fingerprint() -> RunFingerprint:
    return RunFingerprint(
        graph_id="g1",
        goal="compare A and B",
        prompt_sha256="abc",
        node_count=3,
        edge_count=2,
        max_depth=1,
        node_kind_histogram={"llm_call": 2, "reduce": 1},
        topology_signature="sig",
        instruction_sha256="ins",
    )


def _result() -> EvalResult:
    return EvalResult(
        run_id="r1",
        scored_at=datetime(2026, 6, 3, 12, 0, 0),
        gate="deterministic@v1",
        weights={"goal_completion": 0.4},
        applicable_dimensions=["plan_validity", "cost_efficiency"],
        quality=0.72,
        components=[
            ScoreComponent(
                dimension="plan_validity",
                score=0.9,
                verdict="pass",
                method="deterministic",
                signals={"node_count": 3.0},
            )
        ],
        overall_verdict="pass",
        cost_usd=0.0011,
        total_tokens=4000,
        value_per_ktok=value_per_ktok(0.72, 4000),
        value_per_usd=value_per_usd(0.72, 0.0011),
        quality_floor_met=True,
        fingerprint=_fingerprint(),
    )


def test_eval_result_json_round_trips() -> None:
    payload = _result().model_dump(mode="json")
    # JSON-safe with no custom encoder (datetime -> iso string).
    text = json.dumps(payload)
    restored = EvalResult.model_validate(json.loads(text))

    assert restored.run_id == "r1"
    assert restored.schema_version == EVAL_SCHEMA_VERSION
    assert restored.fingerprint is not None
    assert restored.fingerprint.node_kind_histogram == {"llm_call": 2, "reduce": 1}
    assert restored.components[0].signals == {"node_count": 3.0}


def test_signals_are_numeric_and_stable_across_round_trip() -> None:
    # signals is dict[str, float] on purpose: a mixed-type union would not
    # round-trip type-stably. ints coerce to float and stay float.
    comp = ScoreComponent(
        dimension="grounding",
        score=0.5,
        verdict="neutral",
        method="deterministic",
        signals={"claims": 2},  # int in...
    )
    restored = ScoreComponent.model_validate(
        json.loads(json.dumps(comp.model_dump(mode="json")))
    )
    assert restored.signals["claims"] == 2.0
    assert isinstance(restored.signals["claims"], float)


def test_quality_is_bounded() -> None:
    with pytest.raises(ValueError):
        ScoreComponent(
            dimension="grounding", score=1.5, verdict="pass", method="deterministic"
        )


def test_eval_reference_and_tags_construct() -> None:
    ref = EvalReference(checklist=["mentions cost"], grounding_required=True)
    tags = EvalTags(task_id="t1", origin="routed", reference=ref)
    assert tags.origin == "routed"
    assert tags.reference is not None and tags.reference.grounding_required


def test_quality_floor_default_is_documented_value() -> None:
    assert QUALITY_FLOOR_DEFAULT == 0.6


# ---------- TokenUsage relocation back-compat ----------


def test_token_usage_importable_from_both_paths() -> None:
    from dynamic_subgraphs.engine import TokenUsage as EngineTokenUsage
    from dynamic_subgraphs.usage import TokenUsage as UsageTokenUsage

    # Same class, wherever you import it — the relocation is transparent.
    assert TokenUsage is EngineTokenUsage is UsageTokenUsage


# ---------- EvalGate protocol ----------


def test_eval_gate_is_runtime_checkable() -> None:
    class _Gate:
        name = "fake"

        def evaluate(self, context: object) -> object:  # pragma: no cover - shape only
            return context

    assert isinstance(_Gate(), EvalGate)
    assert not isinstance(object(), EvalGate)
