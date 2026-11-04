"""Append-only, hash-chained receipt ledger.

``receipt.json`` lives next to the files it vouches for, so whoever can rewrite
the run directory can rewrite the receipt too. The ledger is the "store
``receipt_hash`` somewhere append-only" step from ``docs/governance.md``, done
for you: one JSON line per persisted receipt, each line hashing the line
before it.

Each ``LedgerEntry`` records:

- ``seq`` (0-based, contiguous) and ``prev_entry_hash`` (the previous line's
  ``entry_hash``; ``None`` for the first line);
- the receipt it anchors: ``run_id`` (the recorded run directory),
  ``receipt_hash``, ``status``;
- ``supersedes``: the ``receipt_hash`` this receipt replaced for the same run,
  e.g. the paused receipt a resume replaced, or the SDK reseal that adds token
  usage. Following ``supersedes`` back gives the run's receipt history;
- ``entry_hash``: sha256 over every field above;
- ``signature`` (optional): a ``Signature`` over ``entry_hash`` from the
  ledger's ``ReceiptSigner``. Since every entry hashes its predecessor,
  verifying the newest signature vouches for the whole chain before it.

Editing, deleting, or reordering a line breaks the chain. Truncating the tail
does not break what's left, so anchor the head (``ReceiptLedger.head()``'s
``entry_hash``) somewhere you control, and pass it back as ``expect_head``;
``check_anchor`` also flags a ``receipt.json`` the ledger has never seen.

The ledger does file I/O (it is the one stateful piece of ``app.governance``).
Appends hold a process-wide lock per path plus an advisory ``flock`` where the
OS has one, so concurrent writers in one process or several stay ordered.
Reads scan the file, which is fine for the thousands of runs a host keeps.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from app.governance.hashing import sha256_hex
from app.governance.receipt import ProvenanceReceipt
from app.governance.signing import (
    KeyRing,
    ReceiptSigner,
    Signature,
    SignatureVerifier,
    ledger_entry_message,
    make_signature,
)

LEDGER_VERSION: Literal[1] = 1
# Default location under a runs directory. A dot-directory, so `list_runs`
# (which needs spec.json) never lists it, and a run id can't silently reuse it.
LEDGER_RELATIVE_PATH = Path(".ledger") / "receipts.jsonl"

try:  # POSIX advisory locks; Windows falls back to the in-process lock.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - platform dependent
    _fcntl = None  # type: ignore[assignment]

_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def default_ledger_path(runs_dir: Path | str) -> Path:
    return Path(runs_dir) / LEDGER_RELATIVE_PATH


class LedgerEntry(BaseModel):
    """One anchored receipt. See the module docstring."""

    ledger_version: Literal[1] = LEDGER_VERSION
    seq: int
    run_id: str
    receipt_run_id: str
    status: str
    receipt_hash: str
    supersedes: str | None = None
    appended_at: datetime
    prev_entry_hash: str | None = None
    entry_hash: str = ""
    signature: Signature | None = None

    def payload(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("entry_hash", None)
        data.pop("signature", None)
        return data

    def compute_hash(self) -> str:
        return sha256_hex(self.payload())


@dataclass(frozen=True)
class LedgerVerification:
    """Result of ``ReceiptLedger.verify``. ``ok`` iff no problem was found."""

    problems: tuple[str, ...] = ()
    entries: int = 0
    signed: int = 0
    head: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "entries": self.entries,
            "signed": self.signed,
            "head": self.head,
        }


@dataclass(frozen=True)
class _Scan:
    entries: list[LedgerEntry] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _path_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.Lock())


class ReceiptLedger:
    """An append-only JSONL file of ``LedgerEntry`` lines.

    ``signer`` (optional) signs each new entry. ``verifier`` defaults to the
    signer's own verifier when it has one (HMAC, Ed25519).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        signer: ReceiptSigner | None = None,
    ) -> None:
        self._path = Path(path)
        self._signer = signer

    @classmethod
    def for_runs_dir(
        cls, runs_dir: Path | str, *, signer: ReceiptSigner | None = None
    ) -> ReceiptLedger:
        return cls(default_ledger_path(runs_dir), signer=signer)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def signer(self) -> ReceiptSigner | None:
        return self._signer

    def __repr__(self) -> str:
        return f"ReceiptLedger({str(self._path)!r}, signer={self._signer!r})"

    # ----------------------------------------------------------------- write

    def append(self, receipt: ProvenanceReceipt) -> LedgerEntry:
        """Anchor a sealed receipt. Idempotent for a receipt already anchored.

        Raises ``ValueError`` for an unsealed or tampered receipt: the ledger
        only vouches for receipts whose hash checks out.
        """
        if not receipt.is_intact():
            raise ValueError("refusing to anchor a receipt whose receipt_hash is wrong")
        run_key = receipt.recorder_run_id or receipt.run_id
        with self._locked():
            scan = self._scan()
            for existing in scan.entries:
                if existing.receipt_hash == receipt.receipt_hash:
                    return existing
            previous = scan.entries[-1] if scan.entries else None
            prior_for_run = _latest_for(scan.entries, run_key)
            entry = LedgerEntry(
                seq=(previous.seq + 1) if previous is not None else 0,
                run_id=run_key,
                receipt_run_id=receipt.run_id,
                status=receipt.status,
                receipt_hash=receipt.receipt_hash,
                supersedes=prior_for_run.receipt_hash if prior_for_run else None,
                appended_at=datetime.now(UTC),
                prev_entry_hash=previous.entry_hash if previous is not None else None,
            )
            entry = entry.model_copy(update={"entry_hash": entry.compute_hash()})
            if self._signer is not None:
                entry = entry.model_copy(
                    update={
                        "signature": make_signature(
                            self._signer, ledger_entry_message(entry.entry_hash)
                        )
                    }
                )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry.model_dump(mode="json"), sort_keys=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return entry

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        with _path_lock(self._path):
            if _fcntl is None:  # pragma: no cover - platform dependent
                yield
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._path.with_name(self._path.name + ".lock")
            with lock_path.open("a") as lock_fh:
                _fcntl.flock(lock_fh.fileno(), _fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    _fcntl.flock(lock_fh.fileno(), _fcntl.LOCK_UN)

    # ------------------------------------------------------------------ read

    def _scan(self) -> _Scan:
        scan = _Scan()
        if not self._path.exists():
            return scan
        text = self._path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                scan.entries.append(LedgerEntry.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                scan.problems.append(
                    f"line {lineno}: not a ledger entry ({exc.__class__.__name__})"
                )
        return scan

    def entries(self) -> list[LedgerEntry]:
        """Every parseable entry, in file order."""
        return self._scan().entries

    def head(self) -> LedgerEntry | None:
        entries = self.entries()
        return entries[-1] if entries else None

    def find(self, receipt_hash: str) -> LedgerEntry | None:
        for entry in self.entries():
            if entry.receipt_hash == receipt_hash:
                return entry
        return None

    def history(self, run_id: str) -> list[LedgerEntry]:
        """Entries for one recorded run, oldest first."""
        return [e for e in self.entries() if e.run_id == run_id]

    def latest_for(self, run_id: str) -> LedgerEntry | None:
        return _latest_for(self.entries(), run_id)

    # ---------------------------------------------------------------- verify

    def verify(
        self,
        *,
        verifier: SignatureVerifier | KeyRing | None = None,
        require_signatures: bool = False,
        expect_head: str | None = None,
    ) -> LedgerVerification:
        """Check the whole chain.

        Flags: unparseable lines, a ``seq`` gap, a broken ``prev_entry_hash``
        link, an ``entry_hash`` that doesn't match its fields, a ``supersedes``
        that isn't the run's previous receipt, a duplicate ``receipt_hash``, a
        bad signature (when a verifier is available), an unsigned entry (when
        ``require_signatures``), and a head other than ``expect_head``.
        """
        if verifier is None and self._signer is not None:
            from app.governance.signing import verifier_for

            verifier = verifier_for(self._signer)
        if require_signatures and verifier is None:
            return LedgerVerification(
                ("require_signatures is set but no verifier key was given",)
            )

        scan = self._scan()
        problems = list(scan.problems)
        signed = 0
        prev_hash: str | None = None
        latest_by_run: dict[str, str] = {}
        seen: set[str] = set()
        for index, entry in enumerate(scan.entries):
            where = f"entry {entry.seq}"
            if entry.seq != index:
                problems.append(
                    f"{where}: expected seq {index} (an entry is missing or moved)"
                )
            if entry.prev_entry_hash != prev_hash:
                problems.append(
                    f"{where}: prev_entry_hash does not link to the entry before it"
                )
            if entry.entry_hash != entry.compute_hash():
                problems.append(f"{where}: entry_hash mismatch (the entry was edited)")
            expected_prior = latest_by_run.get(entry.run_id)
            if entry.supersedes != expected_prior:
                problems.append(
                    f"{where}: supersedes {entry.supersedes!r} but the run's previous "
                    f"receipt is {expected_prior!r}"
                )
            if entry.receipt_hash in seen:
                problems.append(
                    f"{where}: receipt {entry.receipt_hash} is anchored twice"
                )
            seen.add(entry.receipt_hash)
            if entry.signature is not None:
                signed += 1
                if verifier is not None and not verifier.verify(
                    ledger_entry_message(entry.entry_hash), entry.signature
                ):
                    problems.append(
                        f"{where}: signature does not verify (key_id "
                        f"{entry.signature.key_id!r}, alg {entry.signature.alg!r})"
                    )
            elif require_signatures:
                problems.append(f"{where}: entry is unsigned")
            latest_by_run[entry.run_id] = entry.receipt_hash
            prev_hash = entry.entry_hash

        head = scan.entries[-1].entry_hash if scan.entries else None
        if expect_head is not None and head != expect_head:
            problems.append(
                f"head is {head!r}, expected {expect_head!r} (entries were "
                "removed or added after the head was anchored)"
            )
        return LedgerVerification(tuple(problems), len(scan.entries), signed, head)

    def check_anchor(self, receipt: ProvenanceReceipt) -> list[str]:
        """Problems with how ``receipt`` sits in the ledger (empty when fine).

        The receipt must be anchored, and be the latest one for its run: a
        ``receipt.json`` that the ledger has superseded, or never saw, is
        flagged.
        """
        run_key = receipt.recorder_run_id or receipt.run_id
        entries = self.entries()
        entry = next(
            (e for e in entries if e.receipt_hash == receipt.receipt_hash), None
        )
        if entry is None:
            return [f"receipt {receipt.receipt_hash} is not anchored in the ledger"]
        problems: list[str] = []
        if entry.run_id != run_key:
            problems.append(
                f"the ledger anchors this receipt for run {entry.run_id!r}, not {run_key!r}"
            )
        latest = _latest_for(entries, run_key)
        if latest is not None and latest.receipt_hash != receipt.receipt_hash:
            problems.append(
                f"receipt is superseded: the ledger's latest receipt for run "
                f"{run_key!r} is {latest.receipt_hash} (seq {latest.seq})"
            )
        return problems


def _latest_for(entries: list[LedgerEntry], run_id: str) -> LedgerEntry | None:
    for entry in reversed(entries):
        if entry.run_id == run_id:
            return entry
    return None


def verify_ledger(
    path: Path | str,
    *,
    verifier: SignatureVerifier | KeyRing | None = None,
    require_signatures: bool = False,
    expect_head: str | None = None,
) -> LedgerVerification:
    """Convenience wrapper: ``ReceiptLedger(path).verify(...)``."""
    return ReceiptLedger(path).verify(
        verifier=verifier,
        require_signatures=require_signatures,
        expect_head=expect_head,
    )
