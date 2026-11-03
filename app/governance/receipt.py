"""Provenance receipt: a tamper-evident summary of one governed run.

A receipt ties together what the host allowed (``policy_hash``), what ran
(``plan_hash`` of the validated spec), why it was allowed or refused
(``gate_decisions``), what it could have cost (``cost_bound``) next to what it
did cost (``cost_actual``), which models were in play, when it ran, and where
the full log lives (``recorder_run_id`` + ``trace_digest``).

``receipt_hash`` is the sha256 of every other field, so editing any field
without recomputing it is detected. ``verify_receipt`` goes further and checks
the receipt against the recorder's files under ``runs/<run_id>/``:

- ``spec.json`` hashes to ``plan_hash``;
- ``trace.jsonl`` digests to ``trace_digest``, and its ``policy_decision``
  events match ``gate_decisions``;
- ``output.json`` agrees with ``status`` (when it records one);
- ``receipt.json``, if written, is this receipt;
- optionally, a given ``ExecutionPolicy`` hashes to ``policy_hash``.

OpenTelemetry: the recorder has no span concept today (it writes a flat
``TraceEvent`` list), so the receipt does not emit OTel GenAI attributes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.governance.hashing import plan_hash_from_json, sha256_hex, trace_digest
from app.models.trace import TraceEvent, TraceEventKind
from app.policy import ExecutionPolicy

RECEIPT_VERSION: Literal[1] = 1
RECEIPT_FILENAME = "receipt.json"


class CostFigures(BaseModel):
    """Calls, tokens and dollars. ``None`` means unknown, not zero."""

    llm_calls: int | None = None
    tokens: int | None = None
    usd: float | None = None


class GateDecisionRecord(BaseModel):
    gate: str
    decision: Literal["allow", "deny"]
    violations: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ProvenanceReceipt(BaseModel):
    """See the module docstring. Build with ``build_receipt``."""

    receipt_version: Literal[1] = RECEIPT_VERSION
    run_id: str
    recorder_run_id: str | None = None
    status: str
    policy_hash: str
    plan_hash: str | None = None
    catalog_entry_id: str | None = None
    catalog_entry_hash: str | None = None
    gate_decisions: list[GateDecisionRecord] = Field(default_factory=list)
    cost_bound: CostFigures | None = None
    cost_actual: CostFigures = Field(default_factory=CostFigures)
    within_bound: bool | None = None
    model_ids: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    trace_digest: str | None = None
    receipt_hash: str = ""

    def payload(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("receipt_hash", None)
        return data

    def compute_hash(self) -> str:
        return sha256_hex(self.payload())

    def sealed(self) -> ProvenanceReceipt:
        """Copy with ``within_bound`` and ``receipt_hash`` recomputed."""
        fresh = self.model_copy(update={"within_bound": _within(self)})
        return fresh.model_copy(update={"receipt_hash": fresh.compute_hash()})

    def with_actual(
        self,
        *,
        tokens: int | None = None,
        usd: float | None = None,
        models: Iterable[str] = (),
    ) -> ProvenanceReceipt:
        """Fill in observed spend (e.g. from the SDK's usage callback) and reseal."""
        actual = self.cost_actual.model_copy(
            update={
                k: v for k, v in {"tokens": tokens, "usd": usd}.items() if v is not None
            }
        )
        model_ids = sorted(set(self.model_ids) | set(models))
        return self.model_copy(
            update={"cost_actual": actual, "model_ids": model_ids}
        ).sealed()

    def is_intact(self) -> bool:
        return bool(self.receipt_hash) and self.receipt_hash == self.compute_hash()


def _within(r: ProvenanceReceipt) -> bool | None:
    b, a = r.cost_bound, r.cost_actual
    if b is None:
        return None
    checks: list[bool] = []
    for name in ("llm_calls", "tokens", "usd"):
        bv, av = getattr(b, name), getattr(a, name)
        if bv is not None and av is not None:
            checks.append(av <= bv + 1e-12)
    return all(checks) if checks else None


def gate_decisions_from_report(report: Any) -> list[GateDecisionRecord]:
    """Convert a ``PlanGateReport`` (or None) to receipt records."""
    if report is None:
        return []
    return [GateDecisionRecord(**d.as_dict()) for d in report.decisions]


def policy_decision_events(report: Any) -> list[TraceEvent]:
    """One ``policy_decision`` trace event per gate, for the recorder log."""
    if report is None:
        return []
    return [
        TraceEvent(
            kind=TraceEventKind.POLICY_DECISION,
            message=f"plan gate '{d.gate}': {'allow' if d.allowed else 'deny'}",
            data={
                "plan_hash": report.plan_hash,
                "policy_hash": report.policy_hash,
                **d.as_dict(),
            },
        )
        for d in report.decisions
    ]


def build_receipt(
    *,
    run_id: str,
    status: str,
    policy_hash: str,
    started_at: datetime,
    finished_at: datetime,
    plan_hash: str | None = None,
    gate_report: Any = None,
    llm_calls: int | None = None,
    tokens: int | None = None,
    usd: float | None = None,
    model_ids: Iterable[str] = (),
    recorder_run_id: str | None = None,
    trace_events: Iterable[TraceEvent] | None = None,
) -> ProvenanceReceipt:
    """Assemble and seal a receipt. ``gate_report`` is a ``PlanGateReport``."""
    bound = getattr(gate_report, "cost_bound", None)
    receipt = ProvenanceReceipt(
        run_id=run_id,
        recorder_run_id=recorder_run_id,
        status=str(status),
        policy_hash=policy_hash,
        plan_hash=plan_hash,
        catalog_entry_id=getattr(gate_report, "catalog_entry_id", None),
        catalog_entry_hash=getattr(gate_report, "catalog_entry_hash", None),
        gate_decisions=gate_decisions_from_report(gate_report),
        cost_bound=(
            CostFigures(llm_calls=bound.llm_calls, tokens=bound.tokens, usd=bound.usd)
            if bound is not None
            else None
        ),
        cost_actual=CostFigures(llm_calls=llm_calls, tokens=tokens, usd=usd),
        model_ids=sorted(set(model_ids)),
        started_at=started_at,
        finished_at=finished_at,
        trace_digest=trace_digest(trace_events) if trace_events is not None else None,
    )
    return receipt.sealed()


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiptVerification:
    """Result of ``verify_receipt``. ``ok`` iff no problem was found."""

    problems: tuple[str, ...] = ()
    checked: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.problems


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verify_receipt(
    receipt: ProvenanceReceipt | Mapping[str, Any],
    run_dir: Path | str | None = None,
    *,
    policy: ExecutionPolicy | None = None,
) -> ReceiptVerification:
    """Check a receipt's own hash and, given ``run_dir``, the recorder log.

    ``run_dir`` is ``<runs_dir>/<recorder_run_id>``. Files that were not
    recorded (e.g. the recording policy skipped ``trace.jsonl``) are reported
    in ``checked`` as skipped rather than as problems; a receipt that claims a
    ``trace_digest`` or ``plan_hash`` the log can't back is still flagged when
    the file exists and disagrees.
    """
    if not isinstance(receipt, ProvenanceReceipt):
        receipt = ProvenanceReceipt.model_validate(dict(receipt))
    problems: list[str] = []
    checked: list[str] = []

    checked.append("receipt_hash")
    if not receipt.is_intact():
        problems.append(
            "receipt_hash mismatch: the receipt was modified after it was sealed"
        )

    if policy is not None:
        from app.governance.plan_gate import policy_hash

        checked.append("policy_hash")
        if policy_hash(policy) != receipt.policy_hash:
            problems.append(
                "policy_hash mismatch: receipt was not issued under this policy"
            )

    if run_dir is None:
        return ReceiptVerification(tuple(problems), tuple(checked))
    run_dir = Path(run_dir)
    if receipt.recorder_run_id is None:
        problems.append("receipt has no recorder_run_id but a run_dir was given")
        return ReceiptVerification(tuple(problems), tuple(checked))
    if run_dir.name != receipt.recorder_run_id:
        problems.append(
            f"run_dir {run_dir.name!r} is not the recorded run "
            f"{receipt.recorder_run_id!r}"
        )

    spec_path = run_dir / "spec.json"
    if not spec_path.exists() and receipt.status == "policy_denied":
        # A refused plan is recorded as rejected_spec.json (never spec.json).
        spec_path = run_dir / "rejected_spec.json"
    if spec_path.exists():
        checked.append(spec_path.name)
        on_disk = plan_hash_from_json(json.loads(spec_path.read_text(encoding="utf-8")))
        if on_disk != receipt.plan_hash:
            problems.append(
                f"plan_hash mismatch: {spec_path.name} differs from the receipt"
            )
    elif receipt.status == "ok" and receipt.plan_hash is not None:
        checked.append("spec.json (not recorded)")

    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists():
        checked.append("trace.jsonl")
        rows = _read_trace(trace_path)
        if trace_digest(rows) != receipt.trace_digest:
            problems.append(
                "trace_digest mismatch: trace.jsonl differs from the receipt"
            )
        logged = [
            f"{r['data'].get('decision')}:{r['data'].get('gate')}"
            for r in rows
            if r.get("kind") == TraceEventKind.POLICY_DECISION.value
            and "gate" in r.get("data", {})
        ]
        claimed = [f"{d.decision}:{d.gate}" for d in receipt.gate_decisions]
        if logged != claimed:
            problems.append(
                "gate decisions mismatch: trace.jsonl logged "
                f"{logged} but the receipt claims {claimed}"
            )

    output_path = run_dir / "output.json"
    if output_path.exists():
        checked.append("output.json")
        output = json.loads(output_path.read_text(encoding="utf-8"))
        if "status" in output and output["status"] != receipt.status:
            problems.append(
                f"status mismatch: output.json says {output['status']!r}, the "
                f"receipt says {receipt.status!r}"
            )
        if receipt.status == "ok" and output.get("ok") is False:
            problems.append("status mismatch: output.json reports ok=false")

    receipt_path = run_dir / RECEIPT_FILENAME
    if receipt_path.exists():
        checked.append(RECEIPT_FILENAME)
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
        if stored.get("receipt_hash") != receipt.receipt_hash:
            problems.append("receipt.json on disk differs from this receipt")
        else:
            stored_receipt = ProvenanceReceipt.model_validate(stored)
            if not stored_receipt.is_intact():
                problems.append("receipt.json on disk fails its own hash check")

    return ReceiptVerification(tuple(problems), tuple(checked))
