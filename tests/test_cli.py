"""`ds` CLI (app.cli): verify receipts and the ledger, list entries, keygen."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from app.cli import EXIT_OK, EXIT_PROBLEMS, EXIT_USAGE, main
from app.governance import HmacSigner, ReceiptLedger
from app.governance.signing import make_signature
from app.policy import ExecutionPolicy
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig

KEY = "k" * 32


def _ds(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(list(argv), out=buf)
    return code, buf.getvalue()


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    ledger = ReceiptLedger.for_runs_dir(tmp_path, signer=HmacSigner(KEY))
    engine = DynamicSubgraphs(
        EngineConfig(
            planner="mock",
            recording=True,
            runs_dir=tmp_path,
            policy=ExecutionPolicy(),
            receipt_ledger=ledger,
        )
    )
    for run_id in ("a1", "b2"):
        assert engine.run("compare A and B", run_id=run_id).ok
    return tmp_path


def test_verify_all_ok(runs: Path, monkeypatch) -> None:
    monkeypatch.setenv("RECEIPT_KEY", KEY)
    code, out = _ds(
        "verify",
        "--runs-dir",
        str(runs),
        "--hmac-key-env",
        "RECEIPT_KEY",
        "--require-signatures",
    )
    assert code == EXIT_OK, out
    assert "2/2 runs verified" in out
    assert "signature valid" in out
    assert "ledger: OK  2 entries (2 signed)" in out


def test_verify_json_and_single_run_by_path(runs: Path) -> None:
    code, out = _ds("verify", str(runs / "a1"), "--runs-dir", str(runs), "--json")
    assert code == EXIT_OK
    body = json.loads(out)
    assert body["ok"] is True
    assert [r["run_id"] for r in body["runs"]] == ["a1"]
    assert body["ledger"]["entries"] == 2


def test_verify_detects_tampered_spec(runs: Path) -> None:
    spec = runs / "b2" / "spec.json"
    data = json.loads(spec.read_text())
    data["goal"] = "something else"
    spec.write_text(json.dumps(data))
    code, out = _ds("verify", "--runs-dir", str(runs))
    assert code == EXIT_PROBLEMS
    assert "FAIL b2" in out and "plan_hash mismatch" in out
    assert "OK   a1" in out


def test_verify_detects_wrong_key_and_head(runs: Path, monkeypatch) -> None:
    monkeypatch.setenv("WRONG", "z" * 32)
    code, out = _ds("verify", "--runs-dir", str(runs), "--hmac-key-env", "WRONG")
    assert code == EXIT_PROBLEMS and "signature does not verify" in out

    code, out = _ds("verify", "--runs-dir", str(runs), "--expect-head", "sha256:nope")
    assert code == EXIT_PROBLEMS and "expected 'sha256:nope'" in out


def test_verify_warns_about_runs_missing_from_disk(runs: Path) -> None:
    (runs / "a1" / "receipt.json").unlink()
    code, out = _ds("verify", "--runs-dir", str(runs))
    assert code == EXIT_OK
    assert "warning: ledger anchors run 'a1'" in out


def test_verify_without_ledger(tmp_path: Path) -> None:
    engine = DynamicSubgraphs(
        EngineConfig(planner="mock", recording=True, runs_dir=tmp_path)
    )
    engine.run("compare A and B", run_id="solo")
    code, out = _ds("verify", "--runs-dir", str(tmp_path))
    assert code == EXIT_OK
    assert "ledger: none" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["verify", "ghost"],
        ["verify", "--require-signatures"],
        ["verify", "--hmac-key-env", "DS_TEST_UNSET_VAR"],
        ["verify", "--ledger", "/nonexistent/ledger.jsonl"],
    ],
)
def test_usage_errors(runs: Path, argv: list[str]) -> None:
    code, _ = _ds(*argv, "--runs-dir", str(runs))
    assert code == EXIT_USAGE


def test_ledger_listing(runs: Path) -> None:
    code, out = _ds("ledger", "--runs-dir", str(runs))
    assert code == EXIT_OK
    assert "a1" in out and "b2" in out and "hmac-sha256:default" in out
    code, out = _ds("ledger", "--runs-dir", str(runs), "--run", "b2", "--json")
    assert [e["run_id"] for e in json.loads(out)] == ["b2"]


def test_keygen_writes_a_usable_pair(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from app.governance import Ed25519Signer, Ed25519Verifier

    key = tmp_path / "receipts.pem"
    code, _ = _ds("keygen", "--out", str(key), "--key-id", "2026")
    assert code == EXIT_OK
    signer = Ed25519Signer.from_pem_file(key, key_id="2026")
    verifier = Ed25519Verifier.from_pem_file(
        tmp_path / "receipts.pub.pem", key_id="2026"
    )
    assert verifier.verify(b"m", make_signature(signer, b"m"))
    if sys.platform != "win32":  # Windows has no POSIX mode bits
        assert oct(key.stat().st_mode & 0o777) == "0o600"
    assert _ds("keygen", "--out", str(key))[0] == EXIT_USAGE  # no silent overwrite


def test_verify_with_ed25519_public_key(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from app.governance import Ed25519Signer

    signer = Ed25519Signer.generate()
    pub = tmp_path / "pub.pem"
    pub.write_bytes(signer.verifier().public_pem())
    runs = tmp_path / "runs"
    engine = DynamicSubgraphs(
        EngineConfig(
            planner="mock",
            recording=True,
            runs_dir=runs,
            receipt_ledger=ReceiptLedger.for_runs_dir(runs, signer=signer),
        )
    )
    engine.run("compare A and B", run_id="e1")
    code, out = _ds(
        "verify",
        "--runs-dir",
        str(runs),
        "--public-key",
        str(pub),
        "--require-signatures",
    )
    assert code == EXIT_OK, out
