"""Pluggable signatures for receipts and ledger entries.

A receipt's ``receipt_hash`` proves the receipt wasn't edited, but anyone who
can rewrite the run directory can recompute it. A signature binds a hash to a
key the host controls, so a verifier holding the matching key can tell a
genuine receipt from a recomputed one.

Two protocols keep this pluggable:

- ``ReceiptSigner``: ``algorithm``, ``key_id`` and ``sign(message) -> str``.
  Wrap a KMS / HSM / Azure Key Vault client in this to keep keys off the host.
- ``SignatureVerifier``: ``verify(message, signature) -> bool``.

Built in:

- ``HmacSigner``: HMAC-SHA256 with a shared secret (stdlib). It signs and
  verifies, so whoever can verify can also forge; use it when the verifier is
  the host itself.
- ``Ed25519Signer`` / ``Ed25519Verifier``: public-key signatures. Needs the
  ``cryptography`` package (the ``signing`` extra). Hand out the public key and
  keep the private key with the host.

What gets signed is a domain-separated hash string (see ``receipt_message`` and
``ledger_entry_message``), never raw JSON, so a signature for one kind of
object can't be replayed as another.

Pure: no I/O except the explicit ``*_from_pem_file`` helpers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

HMAC_SHA256 = "hmac-sha256"
ED25519 = "ed25519"
# HMAC keys shorter than the hash output weaken the MAC; refuse them.
MIN_HMAC_KEY_BYTES = 32

_RECEIPT_DOMAIN = "dynamic-subgraphs/receipt/v1:"
_LEDGER_DOMAIN = "dynamic-subgraphs/ledger-entry/v1:"


class Signature(BaseModel):
    """A detached signature: which algorithm, which key, and the value (base64)."""

    alg: str
    key_id: str
    value: str


@runtime_checkable
class ReceiptSigner(Protocol):
    """Anything that can sign bytes under a named key."""

    @property
    def algorithm(self) -> str: ...

    @property
    def key_id(self) -> str: ...

    def sign(self, message: bytes) -> str:
        """Return the base64 signature of ``message``."""
        ...


@runtime_checkable
class SignatureVerifier(Protocol):
    """Anything that can check a ``Signature`` over bytes."""

    def verify(self, message: bytes, signature: Signature) -> bool: ...


def receipt_message(receipt_hash: str) -> bytes:
    """The bytes signed for a receipt."""
    return (_RECEIPT_DOMAIN + receipt_hash).encode("utf-8")


def ledger_entry_message(entry_hash: str) -> bytes:
    """The bytes signed for a ledger entry."""
    return (_LEDGER_DOMAIN + entry_hash).encode("utf-8")


def make_signature(signer: ReceiptSigner, message: bytes) -> Signature:
    return Signature(
        alg=signer.algorithm, key_id=signer.key_id, value=signer.sign(message)
    )


def sign_receipt(receipt: Any, signer: ReceiptSigner) -> Signature:
    """Detached signature over a sealed receipt's ``receipt_hash``."""
    receipt_hash = str(getattr(receipt, "receipt_hash", "") or "")
    if not receipt_hash:
        raise ValueError("cannot sign an unsealed receipt (empty receipt_hash)")
    return make_signature(signer, receipt_message(receipt_hash))


def verify_receipt_signature(
    receipt: Any, signature: Signature, verifier: SignatureVerifier | KeyRing
) -> bool:
    receipt_hash = str(getattr(receipt, "receipt_hash", "") or "")
    return bool(receipt_hash) and verifier.verify(
        receipt_message(receipt_hash), signature
    )


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes | None:
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError):
        return None


# ---------------------------------------------------------------------------
# HMAC-SHA256
# ---------------------------------------------------------------------------


class HmacSigner:
    """HMAC-SHA256 with a shared secret. Signs and verifies."""

    algorithm = HMAC_SHA256

    def __init__(self, key: bytes | str, *, key_id: str = "default") -> None:
        raw = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        if len(raw) < MIN_HMAC_KEY_BYTES:
            raise ValueError(
                f"HMAC key must be at least {MIN_HMAC_KEY_BYTES} bytes, got {len(raw)}"
            )
        if not key_id:
            raise ValueError("key_id must be non-empty")
        self._key = raw
        self._key_id = key_id

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, message: bytes) -> str:
        return _b64(hmac.new(self._key, message, hashlib.sha256).digest())

    def verify(self, message: bytes, signature: Signature) -> bool:
        if signature.alg != HMAC_SHA256 or signature.key_id != self._key_id:
            return False
        given = _unb64(signature.value)
        if given is None:
            return False
        expected = hmac.new(self._key, message, hashlib.sha256).digest()
        return hmac.compare_digest(given, expected)

    def __repr__(self) -> str:  # never print the secret
        return f"HmacSigner(key_id={self._key_id!r})"


# ---------------------------------------------------------------------------
# Ed25519 (optional `cryptography` dependency)
# ---------------------------------------------------------------------------


def _crypto() -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise ImportError(
            "Ed25519 signing needs the `cryptography` package: "
            'pip install "dynamic-subgraphs[signing]"'
        ) from exc
    return ed25519


class Ed25519Verifier:
    """Checks Ed25519 signatures with a public key."""

    algorithm = ED25519

    def __init__(self, public_key: Any, *, key_id: str = "default") -> None:
        ed25519 = _crypto()
        if not isinstance(public_key, ed25519.Ed25519PublicKey):
            raise TypeError("public_key must be an Ed25519PublicKey")
        self._public = public_key
        self._key_id = key_id

    @property
    def key_id(self) -> str:
        return self._key_id

    @classmethod
    def from_public_bytes(
        cls, raw: bytes, *, key_id: str = "default"
    ) -> Ed25519Verifier:
        ed25519 = _crypto()
        return cls(ed25519.Ed25519PublicKey.from_public_bytes(raw), key_id=key_id)

    @classmethod
    def from_pem(cls, pem: bytes, *, key_id: str = "default") -> Ed25519Verifier:
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        return cls(load_pem_public_key(pem), key_id=key_id)

    @classmethod
    def from_pem_file(
        cls, path: Path | str, *, key_id: str = "default"
    ) -> Ed25519Verifier:
        return cls.from_pem(Path(path).read_bytes(), key_id=key_id)

    def public_pem(self) -> bytes:
        from cryptography.hazmat.primitives import serialization

        data: bytes = self._public.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return data

    def verify(self, message: bytes, signature: Signature) -> bool:
        if signature.alg != ED25519 or signature.key_id != self._key_id:
            return False
        raw = _unb64(signature.value)
        if raw is None:
            return False
        from cryptography.exceptions import InvalidSignature

        try:
            self._public.verify(raw, message)
        except InvalidSignature:
            return False
        return True


class Ed25519Signer:
    """Ed25519 with a private key. ``verifier()`` gives the public half."""

    algorithm = ED25519

    def __init__(self, private_key: Any, *, key_id: str = "default") -> None:
        ed25519 = _crypto()
        if not isinstance(private_key, ed25519.Ed25519PrivateKey):
            raise TypeError("private_key must be an Ed25519PrivateKey")
        if not key_id:
            raise ValueError("key_id must be non-empty")
        self._private = private_key
        self._key_id = key_id

    @property
    def key_id(self) -> str:
        return self._key_id

    @classmethod
    def generate(cls, *, key_id: str = "default") -> Ed25519Signer:
        return cls(_crypto().Ed25519PrivateKey.generate(), key_id=key_id)

    @classmethod
    def from_pem(
        cls, pem: bytes, *, key_id: str = "default", password: bytes | None = None
    ) -> Ed25519Signer:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        return cls(load_pem_private_key(pem, password=password), key_id=key_id)

    @classmethod
    def from_pem_file(
        cls, path: Path | str, *, key_id: str = "default", password: bytes | None = None
    ) -> Ed25519Signer:
        return cls.from_pem(Path(path).read_bytes(), key_id=key_id, password=password)

    def private_pem(self) -> bytes:
        from cryptography.hazmat.primitives import serialization

        data: bytes = self._private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        return data

    def sign(self, message: bytes) -> str:
        return _b64(self._private.sign(message))

    def verifier(self) -> Ed25519Verifier:
        return Ed25519Verifier(self._private.public_key(), key_id=self._key_id)

    def verify(self, message: bytes, signature: Signature) -> bool:
        return self.verifier().verify(message, signature)

    def __repr__(self) -> str:
        return f"Ed25519Signer(key_id={self._key_id!r})"


# ---------------------------------------------------------------------------
# key ring
# ---------------------------------------------------------------------------


class KeyRing:
    """Several verifiers by ``key_id``, for ledgers signed across key rotations."""

    def __init__(self, verifiers: Mapping[str, SignatureVerifier]) -> None:
        self._verifiers = dict(verifiers)

    def knows(self, key_id: str) -> bool:
        return key_id in self._verifiers

    def verify(self, message: bytes, signature: Signature) -> bool:
        verifier = self._verifiers.get(signature.key_id)
        return verifier is not None and verifier.verify(message, signature)


def verifier_for(signer: ReceiptSigner) -> SignatureVerifier | None:
    """The verifier that matches ``signer``, when it can produce one."""
    if isinstance(signer, Ed25519Signer):
        return signer.verifier()
    if isinstance(signer, SignatureVerifier):
        return signer
    return None
