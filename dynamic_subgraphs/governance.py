"""Plan-level governance for the SDK: policy gate, cost bound, catalog, receipts.

Re-exports the pure governance API from ``app.governance`` so SDK users (and
other hosts, such as an MCP server) have one stable import path::

    from dynamic_subgraphs.governance import (
        CostModel, SubgraphCatalog, check_plan_policy, verify_receipt,
    )

See ``docs/governance.md``.
"""

from __future__ import annotations

from app.governance import (
    GATES,
    PROMPT_PLACEHOLDER,
    RECEIPT_FILENAME,
    CallEstimate,
    CatalogEntry,
    CatalogPlanner,
    CatalogSelectionError,
    CostBound,
    CostFigures,
    CostModel,
    GateDecision,
    GateDecisionRecord,
    ModelPrice,
    PlanGateReport,
    PlanPolicyViolation,
    PolicyViolation,
    ProvenanceReceipt,
    ReceiptVerification,
    SubgraphCatalog,
    TokenCap,
    build_llm_catalog_selector,
    build_receipt,
    check_plan_policy,
    enforce_plan_policy,
    estimate_plan_cost,
    fill_spec,
    keyword_selector,
    plan_hash,
    policy_hash,
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
    "check_plan_policy",
    "enforce_plan_policy",
    "estimate_plan_cost",
    "fill_spec",
    "keyword_selector",
    "plan_hash",
    "policy_hash",
    "verify_receipt",
]
