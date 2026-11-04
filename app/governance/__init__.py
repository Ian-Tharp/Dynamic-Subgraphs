"""Plan-level governance: policy gate, pre-flight cost bound, approved catalog,
provenance receipts, the receipt ledger, and receipt signing.

Everything here is pure (no I/O, no model calls) except ``verify_receipt`` and
``audit_run``, which read the recorder's files, and ``ReceiptLedger``, which
appends to its JSONL file. The supervisor runs the gate after registry
validation and before compile; see ``docs/governance.md``.

Layering: imports ``app.models``, ``app.policy`` and ``app.registry`` only —
never the runtime, supervisor or SDK facade — so any host (SDK, HTTP API, an
MCP server) can call these functions directly.
"""

from app.governance.audit import RunAudit, audit_run, load_stored_receipt
from app.governance.catalog import (
    PROMPT_PLACEHOLDER,
    CatalogEntry,
    CatalogPlanner,
    CatalogSelectionError,
    SubgraphCatalog,
    build_llm_catalog_selector,
    fill_spec,
    keyword_selector,
)
from app.governance.cost import (
    CallEstimate,
    CostBound,
    CostModel,
    ModelPrice,
    TokenCap,
    estimate_plan_cost,
)
from app.governance.hashing import canonical_json, plan_hash, sha256_hex, trace_digest
from app.governance.ledger import (
    LEDGER_RELATIVE_PATH,
    LedgerEntry,
    LedgerVerification,
    ReceiptLedger,
    default_ledger_path,
    verify_ledger,
)
from app.governance.plan_gate import (
    GATES,
    GateDecision,
    PlanGateReport,
    PlanPolicyViolation,
    PolicyViolation,
    check_plan_policy,
    enforce_plan_policy,
    policy_hash,
)
from app.governance.receipt import (
    RECEIPT_FILENAME,
    CostFigures,
    GateDecisionRecord,
    ProvenanceReceipt,
    ReceiptVerification,
    build_receipt,
    policy_decision_events,
    verify_receipt,
)
from app.governance.signing import (
    Ed25519Signer,
    Ed25519Verifier,
    HmacSigner,
    KeyRing,
    ReceiptSigner,
    Signature,
    SignatureVerifier,
    sign_receipt,
    verify_receipt_signature,
)

__all__ = [
    "GATES",
    "LEDGER_RELATIVE_PATH",
    "PROMPT_PLACEHOLDER",
    "RECEIPT_FILENAME",
    "CallEstimate",
    "CatalogEntry",
    "CatalogPlanner",
    "CatalogSelectionError",
    "CostBound",
    "CostFigures",
    "CostModel",
    "Ed25519Signer",
    "Ed25519Verifier",
    "GateDecision",
    "GateDecisionRecord",
    "HmacSigner",
    "KeyRing",
    "LedgerEntry",
    "LedgerVerification",
    "ModelPrice",
    "PlanGateReport",
    "PlanPolicyViolation",
    "PolicyViolation",
    "ProvenanceReceipt",
    "ReceiptLedger",
    "ReceiptSigner",
    "ReceiptVerification",
    "RunAudit",
    "Signature",
    "SignatureVerifier",
    "SubgraphCatalog",
    "TokenCap",
    "audit_run",
    "build_llm_catalog_selector",
    "build_receipt",
    "canonical_json",
    "check_plan_policy",
    "default_ledger_path",
    "enforce_plan_policy",
    "estimate_plan_cost",
    "fill_spec",
    "keyword_selector",
    "load_stored_receipt",
    "plan_hash",
    "policy_decision_events",
    "policy_hash",
    "sha256_hex",
    "sign_receipt",
    "trace_digest",
    "verify_ledger",
    "verify_receipt",
    "verify_receipt_signature",
]
