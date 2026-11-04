"""`ds` — command-line audit tools for recorded runs.

    ds verify                       # every run under ./runs, plus the receipt ledger
    ds verify run-1 run-2           # just these runs (ids or run directories)
    ds verify --hmac-key-env DS_RECEIPT_HMAC_KEY --require-signatures
    ds verify --public-key receipts.pub.pem --expect-head sha256:...
    ds ledger [--run RUN_ID]        # list ledger entries
    ds keygen --out receipts.pem    # new Ed25519 key pair for signing

`ds verify` exits 0 when everything checks out, 1 when it finds a problem,
and 2 on a usage error (bad flag, missing key, unreadable ledger path).

Also runnable as `python -m app.cli`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from app.governance import (
    RECEIPT_FILENAME,
    Ed25519Signer,
    Ed25519Verifier,
    HmacSigner,
    ReceiptLedger,
    RunAudit,
    SignatureVerifier,
    audit_run,
    default_ledger_path,
)

EXIT_OK = 0
EXIT_PROBLEMS = 1
EXIT_USAGE = 2


class UsageError(Exception):
    pass


def _short(value: str | None, n: int = 19) -> str:
    if not value:
        return "-"
    return value if len(value) <= n else value[:n] + "…"


def _runs_dir(args: argparse.Namespace) -> Path:
    return Path(args.runs_dir or os.environ.get("DS_RUNS_DIR") or "runs")


def _ledger(args: argparse.Namespace, runs_dir: Path) -> ReceiptLedger | None:
    if getattr(args, "no_ledger", False):
        return None
    if args.ledger:
        path = Path(args.ledger)
        if not path.is_file():
            raise UsageError(f"ledger file not found: {path}")
        return ReceiptLedger(path)
    path = default_ledger_path(runs_dir)
    return ReceiptLedger(path) if path.is_file() else None


def _verifier(args: argparse.Namespace) -> SignatureVerifier | None:
    if args.hmac_key_env and args.public_key:
        raise UsageError("pass only one of --hmac-key-env and --public-key")
    if args.hmac_key_env:
        secret = os.environ.get(args.hmac_key_env)
        if not secret:
            raise UsageError(f"environment variable {args.hmac_key_env} is not set")
        try:
            return HmacSigner(secret, key_id=args.key_id)
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
    if args.public_key:
        try:
            return Ed25519Verifier.from_pem_file(args.public_key, key_id=args.key_id)
        except (OSError, ValueError, TypeError, ImportError) as exc:
            raise UsageError(
                f"cannot load public key {args.public_key}: {exc}"
            ) from exc
    return None


def _resolve_runs(names: Sequence[str], runs_dir: Path) -> list[Path]:
    if not names:
        if not runs_dir.is_dir():
            raise UsageError(f"runs directory not found: {runs_dir}")
        return [
            child
            for child in sorted(runs_dir.iterdir())
            if child.is_dir()
            and not child.name.startswith(".")
            and (child / RECEIPT_FILENAME).is_file()
        ]
    resolved: list[Path] = []
    for name in names:
        under_root = runs_dir / name
        as_path = Path(name)
        if under_root.is_dir():
            resolved.append(under_root)
        elif as_path.is_dir():
            resolved.append(as_path)
        else:
            raise UsageError(f"no run directory {name!r} (looked in {runs_dir})")
    return resolved


def _print_audit(audit: RunAudit, out: TextIO) -> None:
    status = audit.receipt.status if audit.receipt else "?"
    head = "OK  " if audit.ok else "FAIL"
    extra = ""
    if audit.ledger_entry is not None:
        extra = f"  ledger seq {audit.ledger_entry.seq}, signature {audit.signature}"
    receipt_hash = audit.receipt.receipt_hash if audit.receipt else None
    print(f"{head} {audit.run_id}  {status}  {_short(receipt_hash)}{extra}", file=out)
    for problem in audit.problems:
        print(f"     - {problem}", file=out)


def cmd_verify(args: argparse.Namespace, out: TextIO) -> int:
    runs_dir = _runs_dir(args)
    verifier = _verifier(args)
    if args.require_signatures and verifier is None:
        raise UsageError("--require-signatures needs --hmac-key-env or --public-key")
    ledger = _ledger(args, runs_dir)
    run_dirs = _resolve_runs(args.runs, runs_dir)

    audits = [
        audit_run(
            run_dir,
            ledger=ledger,
            expected_policy_hash=args.policy_hash,
            verifier=verifier,
        )
        for run_dir in run_dirs
    ]
    # An unsigned entry is a problem only when signatures are required.
    if args.require_signatures:
        audits = [
            RunAudit(
                a.run_id,
                (*a.problems, "ledger entry is unsigned"),
                a.checked,
                a.receipt,
                a.ledger_entry,
                a.signature,
            )
            if a.signature == "unsigned"
            else a
            for a in audits
        ]

    ledger_result: dict[str, Any] | None = None
    warnings: list[str] = []
    if ledger is not None:
        check = ledger.verify(
            verifier=verifier,
            require_signatures=args.require_signatures,
            expect_head=args.expect_head,
        )
        ledger_result = {**check.as_dict(), "path": str(ledger.path)}
        if not args.runs:
            # Full sweep: a run the ledger anchors but whose receipt.json is
            # gone may have been deleted. Not every recording keeps
            # receipt.json, so this is a warning, not a failure.
            on_disk = {d.name for d in run_dirs}
            anchored = {e.run_id for e in ledger.entries()}
            for run_id in sorted(anchored - on_disk):
                warnings.append(
                    f"ledger anchors run {run_id!r} but {runs_dir / run_id} has "
                    f"no {RECEIPT_FILENAME}"
                )
    elif args.expect_head:
        raise UsageError("--expect-head needs a ledger (none found)")

    ok = all(a.ok for a in audits) and (ledger_result is None or ledger_result["ok"])
    if args.json:
        json.dump(
            {
                "ok": ok,
                "runs": [a.as_dict() for a in audits],
                "ledger": ledger_result,
                "warnings": warnings,
            },
            out,
            indent=2,
        )
        out.write("\n")
        return EXIT_OK if ok else EXIT_PROBLEMS

    if not audits:
        print(f"no runs with a {RECEIPT_FILENAME} under {runs_dir}", file=out)
    for audit in audits:
        _print_audit(audit, out)
    if ledger_result is None:
        print("ledger: none (receipts are not anchored)", file=out)
    else:
        state = "OK" if ledger_result["ok"] else "FAIL"
        print(
            f"ledger: {state}  {ledger_result['entries']} entries "
            f"({ledger_result['signed']} signed)  head {ledger_result['head'] or '-'}",
            file=out,
        )
        for problem in ledger_result["problems"]:
            print(f"     - {problem}", file=out)
    for warning in warnings:
        print(f"warning: {warning}", file=out)
    failed = sum(not a.ok for a in audits)
    print(
        f"{len(audits) - failed}/{len(audits)} runs verified"
        + ("" if ok else " — problems found"),
        file=out,
    )
    return EXIT_OK if ok else EXIT_PROBLEMS


def cmd_ledger(args: argparse.Namespace, out: TextIO) -> int:
    runs_dir = _runs_dir(args)
    ledger = _ledger(args, runs_dir)
    if ledger is None:
        raise UsageError(f"no receipt ledger at {default_ledger_path(runs_dir)}")
    entries = ledger.history(args.run) if args.run else ledger.entries()
    if args.json:
        json.dump([e.model_dump(mode="json") for e in entries], out, indent=2)
        out.write("\n")
        return EXIT_OK
    print(
        f"{'seq':>5}  {'run':<28} {'status':<18} {'receipt':<21} {'supersedes':<21} signed",
        file=out,
    )
    for e in entries:
        signed = f"{e.signature.alg}:{e.signature.key_id}" if e.signature else "-"
        print(
            f"{e.seq:>5}  {e.run_id:<28} {e.status:<18} {_short(e.receipt_hash):<21} "
            f"{_short(e.supersedes):<21} {signed}",
            file=out,
        )
    return EXIT_OK


def cmd_keygen(args: argparse.Namespace, out: TextIO) -> int:
    private_path = Path(args.out)
    public_path = (
        Path(args.public_out)
        if args.public_out
        else private_path.with_name(private_path.stem + ".pub.pem")
    )
    for path in (private_path, public_path):
        if path.exists() and not args.force:
            raise UsageError(f"{path} exists (use --force to overwrite)")
    try:
        signer = Ed25519Signer.generate(key_id=args.key_id)
    except ImportError as exc:
        raise UsageError(str(exc)) from exc
    private_path.write_bytes(signer.private_pem())
    os.chmod(private_path, 0o600)
    public_path.write_bytes(signer.verifier().public_pem())
    print(f"private key: {private_path}  (set DS_RECEIPT_ED25519_KEY_FILE)", file=out)
    print(f"public key:  {public_path}  (ds verify --public-key ...)", file=out)
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ds",
        description="Audit Dynamic Subgraphs runs: receipts, the receipt ledger, keys.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--runs-dir", help="runs directory (default: $DS_RUNS_DIR or ./runs)"
        )
        p.add_argument(
            "--ledger", help="ledger file (default: <runs-dir>/.ledger/receipts.jsonl)"
        )
        p.add_argument("--json", action="store_true", help="machine-readable output")

    verify = sub.add_parser(
        "verify", help="verify run receipts against their files and the ledger"
    )
    verify.add_argument(
        "runs", nargs="*", help="run ids or run directories (default: all)"
    )
    common(verify)
    verify.add_argument(
        "--no-ledger", action="store_true", help="skip the ledger checks"
    )
    verify.add_argument(
        "--hmac-key-env", metavar="VAR", help="env var holding the HMAC key"
    )
    verify.add_argument("--public-key", metavar="PEM", help="Ed25519 public key file")
    verify.add_argument(
        "--key-id", default="default", help="signing key id (default: default)"
    )
    verify.add_argument(
        "--require-signatures",
        action="store_true",
        help="fail on unsigned ledger entries",
    )
    verify.add_argument(
        "--policy-hash", help="every receipt must carry this policy_hash"
    )
    verify.add_argument(
        "--expect-head", help="the ledger head entry_hash you anchored earlier"
    )
    verify.set_defaults(func=cmd_verify)

    ledger = sub.add_parser("ledger", help="list receipt-ledger entries")
    common(ledger)
    ledger.add_argument("--run", help="only this run's entries")
    ledger.set_defaults(func=cmd_ledger)

    keygen = sub.add_parser("keygen", help="write a new Ed25519 key pair (PEM)")
    keygen.add_argument("--out", required=True, help="private key path")
    keygen.add_argument("--public-out", help="public key path (default: <out>.pub.pem)")
    keygen.add_argument("--key-id", default="default")
    keygen.add_argument("--force", action="store_true")
    keygen.set_defaults(func=cmd_keygen)
    return parser


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = _parser().parse_args(argv)
    try:
        code: int = args.func(args, out)
    except UsageError as exc:
        print(f"ds: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return code


if __name__ == "__main__":
    raise SystemExit(main())
