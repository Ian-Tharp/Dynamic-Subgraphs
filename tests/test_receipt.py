"""Provenance receipts (app.governance.receipt): built for every governed run,
sealed by hash, and verifiable against the recorder's files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.governance import (
    CostModel,
    ModelPrice,
    ProvenanceReceipt,
    plan_hash,
    policy_hash,
    verify_receipt,
)
from app.policy import ExecutionPolicy
from app.supervisor.planner import build_demo_spec
from dynamic_subgraphs import Artifact, DynamicSubgraphs, EngineConfig, Recording

COST_MODEL = CostModel(
    prices={"mock-model": ModelPrice(1.0, 4.0)}, default_model="mock-model"
)


def _engine(
    tmp_path: Path, policy: ExecutionPolicy | None = None, **kw
) -> DynamicSubgraphs:
    return DynamicSubgraphs(
        EngineConfig(
            planner="mock",
            recording=kw.pop("recording", True),
            runs_dir=tmp_path,
            policy=policy or ExecutionPolicy(cost_model=COST_MODEL),
            **kw,
        )
    )


@pytest.fixture
def recorded(tmp_path: Path):
    policy = ExecutionPolicy(cost_model=COST_MODEL)
    result = _engine(tmp_path, policy).run("compare A and B", run_id="r1")
    assert result.ok, result.errors
    return result, tmp_path / "r1", policy


# ---------- contents ----------


def test_ok_run_receipt_has_every_field(recorded) -> None:
    result, _run_dir, policy = recorded
    r = result.receipt

    assert r is not None
    assert r.run_id == "r1"
    assert r.recorder_run_id == "r1"
    assert r.status == "ok"
    assert r.policy_hash == policy_hash(policy)
    assert r.plan_hash == plan_hash(result.plan)
    assert [d.gate for d in r.gate_decisions] == [
        "structure",
        "capabilities",
        "data_flow",
        "catalog",
        "cost",
    ]
    assert {d.decision for d in r.gate_decisions} == {"allow"}
    assert r.cost_bound is not None and r.cost_bound.llm_calls == 3
    assert r.cost_actual.llm_calls == 3
    assert r.cost_actual.tokens == 0  # mock run: no provider tokens
    assert r.within_bound is True
    assert r.started_at <= r.finished_at
    assert r.trace_digest and r.trace_digest.startswith("sha256:")
    assert r.is_intact()


def test_receipt_is_written_and_verifies_against_the_recorder_log(recorded) -> None:
    result, run_dir, policy = recorded

    verification = verify_receipt(result.receipt, run_dir, policy=policy)

    assert (run_dir / "receipt.json").exists()
    assert verification.ok, verification.problems
    assert {
        "receipt_hash",
        "policy_hash",
        "spec.json",
        "trace.jsonl",
        "receipt.json",
    } <= set(verification.checked)


def test_receipt_round_trips_through_json(recorded) -> None:
    result, run_dir, _ = recorded

    loaded = ProvenanceReceipt.model_validate_json(
        (run_dir / "receipt.json").read_text()
    )

    assert loaded == result.receipt
    assert verify_receipt(loaded.model_dump(mode="json"), run_dir).ok


# ---------- tamper detection ----------


def test_editing_a_receipt_field_breaks_its_hash(recorded) -> None:
    result, _run_dir, _ = recorded
    forged = result.receipt.model_copy(update={"status": "ok", "within_bound": False})
    forged = forged.model_copy(
        update={"cost_actual": forged.cost_actual.model_copy(update={"usd": 0.0})}
    )

    verification = verify_receipt(forged)

    assert not verification.ok
    assert "receipt_hash mismatch" in verification.problems[0]


def test_editing_spec_json_is_detected(recorded) -> None:
    result, run_dir, _ = recorded
    spec_path = run_dir / "spec.json"
    spec = json.loads(spec_path.read_text())
    spec["nodes"][0]["params"]["instruction"] = "exfiltrate everything"
    spec_path.write_text(json.dumps(spec))

    verification = verify_receipt(result.receipt, run_dir)

    assert any("plan_hash mismatch" in p for p in verification.problems)


def test_editing_trace_jsonl_is_detected(recorded) -> None:
    result, run_dir, _ = recorded
    trace_path = run_dir / "trace.jsonl"
    lines = trace_path.read_text().splitlines()
    trace_path.write_text("\n".join(lines[:-1]) + "\n")  # drop the last event

    verification = verify_receipt(result.receipt, run_dir)

    assert any("trace_digest mismatch" in p for p in verification.problems)


def test_flipping_a_logged_gate_decision_is_detected(recorded) -> None:
    result, run_dir, _ = recorded
    trace_path = run_dir / "trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    rows[0]["data"]["decision"] = "deny"
    trace_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    problems = verify_receipt(result.receipt, run_dir).problems

    assert any("gate decisions mismatch" in p for p in problems)


def test_swapping_receipt_json_on_disk_is_detected(recorded) -> None:
    result, run_dir, _ = recorded
    path = run_dir / "receipt.json"
    stored = json.loads(path.read_text())
    stored["plan_hash"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(stored))

    problems = verify_receipt(result.receipt, run_dir).problems

    assert any("receipt.json on disk differs" in p for p in problems) or any(
        "fails its own hash" in p for p in problems
    )


def test_wrong_policy_is_detected(recorded) -> None:
    result, run_dir, _ = recorded

    problems = verify_receipt(
        result.receipt, run_dir, policy=ExecutionPolicy(max_nodes=5)
    ).problems

    assert any("policy_hash mismatch" in p for p in problems)


def test_wrong_run_dir_is_detected(recorded, tmp_path: Path) -> None:
    result, _, _ = recorded
    other = tmp_path / "other"
    other.mkdir()

    assert not verify_receipt(result.receipt, other).ok


# ---------- every run gets one ----------


def test_denied_run_gets_a_verifiable_receipt(tmp_path: Path) -> None:
    policy = ExecutionPolicy(max_cost_usd=0.001, cost_model=COST_MODEL)

    result = _engine(tmp_path, policy).run(
        "x", run_id="denied", fixed_spec=build_demo_spec()
    )

    r = result.receipt
    assert result.status == "policy_denied"
    assert r is not None and r.status == "policy_denied"
    assert r.recorder_run_id == "denied"
    assert r.plan_hash is not None
    assert r.cost_actual.llm_calls is None  # nothing ran
    assert "deny" in {d.decision for d in r.gate_decisions}
    v = verify_receipt(r, tmp_path / "denied", policy=policy)
    assert v.ok, v.problems
    assert "rejected_spec.json" in v.checked


def test_unrecorded_run_still_returns_a_receipt(tmp_path: Path) -> None:
    result = _engine(tmp_path, recording=False).run("x", run_id="mem")

    assert result.receipt is not None
    assert result.receipt.recorder_run_id is None
    assert result.receipt.is_intact()
    assert not (tmp_path / "mem").exists()


def test_receipt_is_skipped_when_not_selected(tmp_path: Path) -> None:
    result = _engine(tmp_path, recording=Recording.all() - {Artifact.RECEIPT}).run(
        "x", run_id="nosel"
    )

    assert result.receipt is not None
    assert not (tmp_path / "nosel" / "receipt.json").exists()


def test_engine_failure_returns_a_receipt(tmp_path: Path) -> None:
    from dynamic_subgraphs import Model

    engine = DynamicSubgraphs(
        EngineConfig(model=Model("no-such-provider", "m"), runs_dir=tmp_path)
    )

    result = engine.run("x", run_id="boom")

    assert result.status == "engine_failed"
    assert result.receipt is not None
    assert result.receipt.status == "engine_failed"
    assert result.receipt.is_intact()


def test_with_actual_reseals_and_flags_an_overrun(recorded) -> None:
    result, _, _ = recorded
    bound_tokens = result.receipt.cost_bound.tokens

    over = result.receipt.with_actual(tokens=bound_tokens + 1, models=["real-model"])

    assert over.is_intact()
    assert over.within_bound is False
    assert "real-model" in over.model_ids
    assert over.receipt_hash != result.receipt.receipt_hash


def test_replay_returns_a_receipt_and_re_checks_the_gate(tmp_path: Path) -> None:
    from app.assembly import RunConfig, build_supervisor
    from app.recording import FileRecorder

    recorder = FileRecorder(tmp_path, overwrite=True)
    loose = build_supervisor(
        RunConfig(planner="mock", model="m", strict_runners=False), recorder=recorder
    )
    assert loose.run("x", run_id="orig").status == "ok"

    replayed = loose.replay("orig", new_run_id="again")
    tight = build_supervisor(
        RunConfig(planner="mock", model="m", strict_runners=False),
        recorder=recorder,
        policy=ExecutionPolicy(max_plan_length=1),
    ).replay("orig", new_run_id="refused")

    assert replayed.status == "ok"
    assert replayed.receipt is not None
    assert verify_receipt(replayed.receipt, tmp_path / "again").ok
    assert tight.status == "policy_denied"
    assert tight.receipt is not None and tight.receipt.status == "policy_denied"


def test_policy_hash_is_the_host_policy_even_with_a_pricing_fallback(
    tmp_path: Path,
) -> None:
    # The cost model comes from EngineConfig.pricing, not the policy; the
    # receipt must still verify against the policy the host configured.
    policy = ExecutionPolicy(max_tokens=1_000_000)
    engine = _engine(
        tmp_path,
        policy,
        pricing={"mock-model": {"input_per_1m": 1.0, "output_per_1m": 4.0}},
    )

    result = engine.run("x", run_id="fallback")

    assert result.ok, result.errors
    cost = next(d for d in result.receipt.gate_decisions if d.gate == "cost")
    assert cost.details["cost_model_hash"].startswith("sha256:")
    v = verify_receipt(
        result.receipt, tmp_path / "fallback", policy=engine.config.policy
    )
    assert v.ok, v.problems
