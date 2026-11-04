"""Receipt signing (app.governance.signing): HMAC and Ed25519 signers, key ring."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.governance import (
    Ed25519Signer,
    Ed25519Verifier,
    HmacSigner,
    KeyRing,
    Signature,
    build_receipt,
    sign_receipt,
    verify_receipt_signature,
)
from app.governance.signing import (
    ledger_entry_message,
    receipt_message,
    verifier_for,
)

KEY = "k" * 32


def _receipt(status: str = "ok"):
    now = datetime.now(UTC)
    return build_receipt(
        run_id="r1",
        status=status,
        policy_hash="sha256:p",
        started_at=now,
        finished_at=now,
    )


def test_hmac_round_trip_and_wrong_key() -> None:
    signer = HmacSigner(KEY, key_id="k1")
    receipt = _receipt()
    sig = sign_receipt(receipt, signer)
    assert (sig.alg, sig.key_id) == ("hmac-sha256", "k1")
    assert verify_receipt_signature(receipt, sig, signer)
    assert not verify_receipt_signature(receipt, sig, HmacSigner("x" * 32, key_id="k1"))
    assert not verify_receipt_signature(_receipt("failed"), sig, signer)


def test_hmac_rejects_short_key_and_hides_secret() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        HmacSigner("short")
    assert KEY not in repr(HmacSigner(KEY))


def test_signature_is_domain_separated() -> None:
    signer = HmacSigner(KEY)
    sig = Signature(
        alg=signer.algorithm,
        key_id=signer.key_id,
        value=signer.sign(receipt_message("h")),
    )
    assert signer.verify(receipt_message("h"), sig)
    assert not signer.verify(ledger_entry_message("h"), sig)


def test_garbage_signature_values_do_not_raise() -> None:
    signer = HmacSigner(KEY)
    bad = Signature(alg="hmac-sha256", key_id="default", value="!!not base64!!")
    assert signer.verify(b"m", bad) is False


def test_unsealed_receipt_cannot_be_signed() -> None:
    receipt = _receipt().model_copy(update={"receipt_hash": ""})
    with pytest.raises(ValueError, match="unsealed"):
        sign_receipt(receipt, HmacSigner(KEY))


def test_ed25519_round_trip_via_pem() -> None:
    pytest.importorskip("cryptography")
    signer = Ed25519Signer.generate(key_id="ed")
    receipt = _receipt()
    sig = sign_receipt(receipt, signer)
    verifier = Ed25519Verifier.from_pem(signer.verifier().public_pem(), key_id="ed")
    assert verify_receipt_signature(receipt, sig, verifier)
    # A different key, or the right key under another key id, fails.
    other = Ed25519Signer.generate(key_id="ed").verifier()
    assert not verify_receipt_signature(receipt, sig, other)
    renamed = Ed25519Verifier.from_pem(signer.verifier().public_pem(), key_id="x")
    assert not verify_receipt_signature(receipt, sig, renamed)
    # Private key PEM round-trips.
    again = Ed25519Signer.from_pem(signer.private_pem(), key_id="ed")
    assert verify_receipt_signature(receipt, sign_receipt(receipt, again), verifier)


def test_key_ring_picks_verifier_by_key_id() -> None:
    old, new = HmacSigner(KEY, key_id="2025"), HmacSigner("n" * 32, key_id="2026")
    ring = KeyRing({"2025": old, "2026": new})
    receipt = _receipt()
    assert verify_receipt_signature(receipt, sign_receipt(receipt, old), ring)
    assert verify_receipt_signature(receipt, sign_receipt(receipt, new), ring)
    stranger = HmacSigner("s" * 32, key_id="2024")
    assert not verify_receipt_signature(receipt, sign_receipt(receipt, stranger), ring)
    assert ring.knows("2025") and not ring.knows("2024")


def test_verifier_for() -> None:
    hmac_signer = HmacSigner(KEY)
    assert verifier_for(hmac_signer) is hmac_signer
    pytest.importorskip("cryptography")
    assert isinstance(verifier_for(Ed25519Signer.generate()), Ed25519Verifier)
