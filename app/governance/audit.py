"""Audit one recorded run: its ``receipt.json``, its files, and its ledger anchor.

``audit_run`` is what ``ds verify`` and the MCP ``verify_receipt`` tool call. It
reads ``receipt.json`` from the run directory and runs ``verify_receipt``
against the directory, then (with a ledger) checks the receipt is anchored and
is the run's latest, and that the anchoring entry's signature verifies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.governance.ledger import LedgerEntry, ReceiptLedger
from app.governance.receipt import RECEIPT_FILENAME, ProvenanceReceipt, verify_receipt
from app.governance.signing import (
    KeyRing,
    SignatureVerifier,
    ledger_entry_message,
    verifier_for,
)
from app.policy import ExecutionPolicy


@dataclass(frozen=True)
class RunAudit:
    """Result of ``audit_run``. ``ok`` iff no problem was found."""

    run_id: str
    problems: tuple[str, ...]
    checked: tuple[str, ...]
    receipt: ProvenanceReceipt | None = None
    ledger_entry: LedgerEntry | None = None
    signature: str = "n/a"  # n/a | unsigned | valid | invalid | unchecked

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ok": self.ok,
            "problems": list(self.problems),
            "checked": list(self.checked),
            "status": self.receipt.status if self.receipt else None,
            "receipt_hash": self.receipt.receipt_hash if self.receipt else None,
            "ledger_entry": (
                self.ledger_entry.model_dump(mode="json") if self.ledger_entry else None
            ),
            "signature": self.signature,
        }


def load_stored_receipt(run_dir: Path | str) -> ProvenanceReceipt:
    """Parse ``<run_dir>/receipt.json``. Raises ``FileNotFoundError`` / ``ValueError``."""
    path = Path(run_dir) / RECEIPT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {RECEIPT_FILENAME} in {Path(run_dir)}")
    try:
        return ProvenanceReceipt.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"{path} is not a valid receipt: {exc}") from exc


def audit_run(
    run_dir: Path | str,
    *,
    ledger: ReceiptLedger | None = None,
    policy: ExecutionPolicy | None = None,
    expected_policy_hash: str | None = None,
    verifier: SignatureVerifier | KeyRing | None = None,
) -> RunAudit:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    try:
        receipt = load_stored_receipt(run_dir)
    except (FileNotFoundError, ValueError) as exc:
        return RunAudit(run_id, (str(exc),), (RECEIPT_FILENAME,))

    check = verify_receipt(receipt, run_dir, policy=policy)
    problems = list(check.problems)
    checked = list(check.checked)
    if expected_policy_hash is not None:
        checked.append("expected_policy_hash")
        if receipt.policy_hash != expected_policy_hash:
            problems.append(
                f"policy_hash is {receipt.policy_hash}, expected {expected_policy_hash}"
            )

    entry: LedgerEntry | None = None
    signature = "n/a"
    if ledger is not None:
        checked.append("ledger")
        problems.extend(ledger.check_anchor(receipt))
        entry = ledger.find(receipt.receipt_hash)
        if verifier is None and ledger.signer is not None:
            verifier = verifier_for(ledger.signer)
        if entry is not None:
            if entry.signature is None:
                signature = "unsigned"
            elif verifier is None:
                signature = "unchecked"
            elif verifier.verify(
                ledger_entry_message(entry.entry_hash), entry.signature
            ):
                signature = "valid"
            else:
                signature = "invalid"
                problems.append(
                    f"ledger entry {entry.seq}: signature does not verify "
                    f"(key_id {entry.signature.key_id!r})"
                )
            if entry.entry_hash != entry.compute_hash():
                problems.append(f"ledger entry {entry.seq}: entry_hash mismatch")
    return RunAudit(run_id, tuple(problems), tuple(checked), receipt, entry, signature)
