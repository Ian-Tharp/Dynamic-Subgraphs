"""Receipt ledger (app.governance.ledger): append-only, hash-chained anchors for
persisted receipts, with optional signatures; plus `audit_run`."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.governance import (
    HmacSigner,
    LedgerEntry,
    ReceiptLedger,
    audit_run,
    build_receipt,
    default_ledger_path,
    verify_ledger,
)
from app.policy import ExecutionPolicy
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig

KEY = "k" * 32


def _receipt(run_id: str = "r1", status: str = "ok"):
    now = datetime.now(UTC)
    return build_receipt(
        run_id=run_id,
        recorder_run_id=run_id,
        status=status,
        policy_hash="sha256:p",
        started_at=now,
        finished_at=now,
    )


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def ledger(tmp_path: Path) -> ReceiptLedger:
    return ReceiptLedger(tmp_path / "ledger.jsonl", signer=HmacSigner(KEY))


# ---------- append ----------


def test_entries_chain_and_link_superseded_receipts(ledger: ReceiptLedger) -> None:
    paused = _receipt("r1", "paused")
    other = _receipt("r2")
    resumed = _receipt("r1", "ok")
    e0, e1, e2 = (ledger.append(r) for r in (paused, other, resumed))

    assert [e.seq for e in (e0, e1, e2)] == [0, 1, 2]
    assert e0.prev_entry_hash is None
    assert e1.prev_entry_hash == e0.entry_hash
    assert e2.prev_entry_hash == e1.entry_hash
    assert e1.supersedes is None
    assert e2.supersedes == paused.receipt_hash
    assert [e.status for e in ledger.history("r1")] == ["paused", "ok"]
    assert ledger.latest_for("r1") == e2
    assert ledger.head() == e2
    assert all(e.signature is not None for e in (e0, e1, e2))
    check = ledger.verify()
    assert check.ok, check.problems
    assert (check.entries, check.signed, check.head) == (3, 3, e2.entry_hash)


def test_append_is_idempotent(ledger: ReceiptLedger) -> None:
    r = _receipt()
    assert ledger.append(r) == ledger.append(r)
    assert len(ledger.entries()) == 1


def test_append_refuses_a_tampered_receipt(ledger: ReceiptLedger) -> None:
    forged = _receipt().model_copy(update={"status": "failed"})
    with pytest.raises(ValueError, match="receipt_hash"):
        ledger.append(forged)
    assert ledger.entries() == []


def test_concurrent_appends_stay_chained(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    receipts = [_receipt(f"r{i}") for i in range(24)]

    def worker(chunk) -> None:
        # Separate instances, same file: the per-path lock must still serialize.
        own = ReceiptLedger(path)
        for r in chunk:
            own.append(r)

    threads = [
        threading.Thread(target=worker, args=(receipts[i::4],)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check = verify_ledger(path)
    assert check.ok, check.problems
    assert check.entries == 24


# ---------- verify: tampering ----------


def test_edited_entry_is_caught(ledger: ReceiptLedger) -> None:
    for i in range(3):
        ledger.append(_receipt(f"r{i}"))
    rows = _lines(ledger.path)
    rows[1]["status"] = "failed"
    _write(ledger.path, rows)
    problems = ledger.verify().problems
    assert any("entry 1: entry_hash mismatch" in p for p in problems)


def test_rehashed_edit_breaks_the_signature_and_the_chain(
    ledger: ReceiptLedger,
) -> None:
    for i in range(3):
        ledger.append(_receipt(f"r{i}"))
    rows = _lines(ledger.path)
    rows[1]["status"] = "failed"
    rows[1]["entry_hash"] = LedgerEntry.model_validate(rows[1]).compute_hash()
    _write(ledger.path, rows)
    problems = ledger.verify().problems
    assert any("entry 1: signature does not verify" in p for p in problems)
    assert any("entry 2: prev_entry_hash" in p for p in problems)
    # Without the key the forged entry is self-consistent, but entry 2 still
    # points at the original.
    unsigned_view = verify_ledger(ledger.path)
    assert any("entry 2: prev_entry_hash" in p for p in unsigned_view.problems)


def test_deleted_middle_entry_is_caught(ledger: ReceiptLedger) -> None:
    for i in range(3):
        ledger.append(_receipt(f"r{i}"))
    rows = _lines(ledger.path)
    _write(ledger.path, [rows[0], rows[2]])
    problems = ledger.verify().problems
    assert any("expected seq 1" in p for p in problems)
    assert any("prev_entry_hash" in p for p in problems)


def test_truncation_is_caught_with_expect_head(ledger: ReceiptLedger) -> None:
    for i in range(3):
        ledger.append(_receipt(f"r{i}"))
    head = ledger.head().entry_hash
    rows = _lines(ledger.path)
    _write(ledger.path, rows[:2])
    assert ledger.verify().ok  # a shorter chain is still a valid chain...
    problems = ledger.verify(expect_head=head).problems
    assert any("expected" in p and "head" in p for p in problems)  # ...but not ours


def test_dropped_supersedes_link_is_caught(ledger: ReceiptLedger) -> None:
    ledger.append(_receipt("r1", "paused"))
    ledger.append(_receipt("r1", "ok"))
    rows = _lines(ledger.path)
    rows[1]["supersedes"] = None
    rows[1]["entry_hash"] = LedgerEntry.model_validate(rows[1]).compute_hash()
    _write(ledger.path, rows)
    assert any("supersedes" in p for p in verify_ledger(ledger.path).problems)


def test_garbage_line_is_reported(ledger: ReceiptLedger) -> None:
    ledger.append(_receipt())
    with ledger.path.open("a") as fh:
        fh.write("{not json\n")
    assert any("not a ledger entry" in p for p in ledger.verify().problems)


def test_require_signatures(tmp_path: Path) -> None:
    unsigned = ReceiptLedger(tmp_path / "l.jsonl")
    unsigned.append(_receipt())
    signer = HmacSigner(KEY)
    assert not unsigned.verify(verifier=signer, require_signatures=True).ok
    assert not unsigned.verify(require_signatures=True).ok  # no key to check with
    assert unsigned.verify().ok


def test_wrong_key_fails(ledger: ReceiptLedger) -> None:
    ledger.append(_receipt())
    other = HmacSigner("z" * 32)
    assert not verify_ledger(ledger.path, verifier=other).ok
    assert verify_ledger(ledger.path, verifier=HmacSigner(KEY)).ok


# ---------- anchoring + audit_run with real runs ----------


def _engine(tmp_path: Path, ledger: ReceiptLedger | None) -> DynamicSubgraphs:
    return DynamicSubgraphs(
        EngineConfig(
            planner="mock",
            recording=True,
            runs_dir=tmp_path,
            policy=ExecutionPolicy(),
            receipt_ledger=ledger,
        )
    )


def test_sdk_anchors_the_final_receipt_once(tmp_path: Path) -> None:
    ledger = ReceiptLedger.for_runs_dir(tmp_path, signer=HmacSigner(KEY))
    result = _engine(tmp_path, ledger).run("compare A and B", run_id="r1")
    assert result.ok
    entries = ledger.entries()
    # One entry per run, for the receipt with token usage filled in.
    assert len(entries) == 1
    assert entries[0].receipt_hash == result.receipt.receipt_hash
    assert ledger.path == default_ledger_path(tmp_path)
    audit = audit_run(tmp_path / "r1", ledger=ledger)
    assert audit.ok, audit.problems
    assert audit.signature == "valid"


def test_sdk_without_ledger_writes_none(tmp_path: Path) -> None:
    _engine(tmp_path, None).run("compare A and B", run_id="r1")
    assert not default_ledger_path(tmp_path).exists()


def test_audit_catches_a_resealed_receipt(tmp_path: Path) -> None:
    """Rewriting receipt.json *and* recomputing its hash passes verify_receipt's
    self-check, but the ledger never anchored the forged hash."""
    ledger = ReceiptLedger.for_runs_dir(tmp_path)
    result = _engine(tmp_path, ledger).run("compare A and B", run_id="r1")
    forged = result.receipt.model_copy(
        update={"model_ids": ["innocent-model"]}
    ).sealed()
    (tmp_path / "r1" / "receipt.json").write_text(forged.model_dump_json())

    no_ledger = audit_run(tmp_path / "r1")
    assert no_ledger.ok  # self-consistent forgery
    audit = audit_run(tmp_path / "r1", ledger=ledger)
    assert not audit.ok
    assert any("not anchored" in p for p in audit.problems)


def test_audit_catches_a_rolled_back_receipt(tmp_path: Path) -> None:
    """Restoring an older (superseded) receipt.json is flagged."""
    ledger = ReceiptLedger.for_runs_dir(tmp_path)
    result = _engine(tmp_path, ledger).run("compare A and B", run_id="r1")
    newer = result.receipt.with_actual(tokens=999)
    ledger.append(newer)
    audit = audit_run(tmp_path / "r1", ledger=ledger)
    assert any("superseded" in p for p in audit.problems)


def test_audit_reports_missing_receipt_and_policy_hash(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert "no receipt.json" in audit_run(tmp_path / "empty").problems[0]

    _engine(tmp_path, None).run("compare A and B", run_id="r1")
    audit = audit_run(tmp_path / "r1", expected_policy_hash="sha256:other")
    assert any("policy_hash" in p for p in audit.problems)
