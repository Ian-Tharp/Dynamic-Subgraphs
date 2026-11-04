"""Receipts across human-approval resume: `Supervisor.resume` re-checks the plan
gate, returns a fresh receipt, and that receipt verifies against the run
directory the resume rewrote."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from app.governance import ProvenanceReceipt, verify_receipt
from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.policy import ExecutionPolicy
from app.recording import FileRecorder
from app.runtime import LangGraphExecutor
from app.supervisor import StaticPlanner, Supervisor


def _llm(state, params):
    return {"result": params["instruction"]}


def _wait_spec() -> GraphSpec:
    nodes = [
        NodeSpec(
            id="seed",
            kind=NodeKind.LLM_CALL,
            outputs=["question"],
            params={"instruction": "ask"},
        ),
        NodeSpec(
            id="approve",
            kind=NodeKind.WAIT_FOR_EVENT,
            outputs=["answer"],
            params={"event_type": "approval"},
        ),
        NodeSpec(
            id="finalize",
            kind=NodeKind.LLM_CALL,
            inputs=["answer"],
            outputs=["final"],
            params={"instruction": "wrap up"},
        ),
    ]
    ids = ["START", "seed", "approve", "finalize", "END"]
    edges = [
        EdgeSpec.model_validate({"from": a, "to": b})
        for a, b in itertools.pairwise(ids)
    ]
    return GraphSpec(graph_id="wait", goal="approval", nodes=nodes, edges=edges)


def _supervisor(
    tmp_path: Path, saver: MemorySaver, policy: ExecutionPolicy | None = None
) -> Supervisor:
    return Supervisor(
        planner=StaticPlanner(_wait_spec()),
        executor=LangGraphExecutor(
            runners={NodeKind.LLM_CALL: _llm}, checkpointer=saver
        ),
        recorder=FileRecorder(root_dir=tmp_path),
        policy=policy,
    )


def test_resume_returns_a_receipt_that_verifies_the_rewritten_run(
    tmp_path: Path,
) -> None:
    saver = MemorySaver()
    sup = _supervisor(tmp_path, saver)
    paused = sup.run("x", run_id="r")
    assert paused.status == "paused" and paused.receipt is not None
    run_dir = tmp_path / "r"
    assert verify_receipt(paused.receipt, run_dir).ok

    resumed = sup.resume("r", event="yes")

    assert resumed.status == "ok"
    r = resumed.receipt
    assert r is not None and r.status == "ok" and r.recorder_run_id == "r"
    assert r.plan_hash == paused.receipt.plan_hash
    assert r.started_at == paused.receipt.started_at
    assert [d.gate for d in r.gate_decisions] == [
        d.gate for d in paused.receipt.gate_decisions
    ]
    assert r.cost_actual.llm_calls == 2  # pre- and post-pause calls
    v = verify_receipt(r, run_dir, policy=sup.policy)
    assert v.ok, v.problems
    # receipt.json on disk is the resumed receipt, not the stale paused one.
    on_disk = ProvenanceReceipt.model_validate(
        json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    )
    assert on_disk.receipt_hash == r.receipt_hash
    # The trace keeps one copy of each gate decision.
    rows = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gate_rows = [row for row in rows if row["kind"] == "policy_decision"]
    assert len(gate_rows) == len(r.gate_decisions)


def test_resume_is_refused_when_the_policy_tightened_while_paused(
    tmp_path: Path,
) -> None:
    saver = MemorySaver()
    paused = _supervisor(tmp_path, saver).run("x", run_id="r")
    assert paused.status == "paused" and paused.receipt is not None

    tight = _supervisor(tmp_path, saver, ExecutionPolicy(max_plan_length=1))
    refused = tight.resume("r", event="yes")

    assert refused.status == "policy_denied"
    assert refused.result is None
    assert refused.errors[0]["stage"] == "resume_policy_gate"
    assert refused.receipt is not None and refused.receipt.status == "policy_denied"
    assert refused.receipt.recorder_run_id is None  # nothing new was recorded
    # The paused recording and its receipt are untouched and still verify.
    assert verify_receipt(paused.receipt, tmp_path / "r").ok
    # ...and the run is still resumable under a policy that allows it.
    assert _supervisor(tmp_path, saver).resume("r", event="yes").status == "ok"


def test_failed_resume_still_returns_a_receipt(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path, MemorySaver())

    result = sup.resume("never-ran", event="yes")

    assert result.status == "resume_failed"
    assert result.receipt is not None and result.receipt.status == "resume_failed"
