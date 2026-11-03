"""Plan-level governance: policy gate, pre-flight cost bound, approved catalog,
and provenance receipts.

Everything here is pure (no I/O, no model calls) except ``verify_receipt``,
which reads the recorder's files. The supervisor runs the gate after registry
validation and before compile; see ``docs/governance.md``.

Layering: imports ``app.models``, ``app.policy`` and ``app.registry`` only —
never the runtime, supervisor or SDK facade — so any host (SDK, HTTP API, an
MCP server) can call these functions directly.
"""

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

__all__ = [
    "GATES",
    "PROMPT_PLACEHOLDER",
    "RECEIPT_FILENAME",
    "CallEstimate",
    "CatalogEntry",
    "CatalogPlanner",
    "CatalogSelectionError",
    "CostBound",
    "CostFigures",
    "CostModel",
    "GateDecision",
    "GateDecisionRecord",
    "ModelPrice",
    "PlanGateReport",
    "PlanPolicyViolation",
    "PolicyViolation",
    "ProvenanceReceipt",
    "ReceiptVerification",
    "SubgraphCatalog",
    "TokenCap",
    "build_llm_catalog_selector",
    "build_receipt",
    "canonical_json",
    "check_plan_policy",
    "enforce_plan_policy",
    "estimate_plan_cost",
    "fill_spec",
    "keyword_selector",
    "plan_hash",
    "policy_decision_events",
    "policy_hash",
    "sha256_hex",
    "trace_digest",
    "verify_receipt",
]
