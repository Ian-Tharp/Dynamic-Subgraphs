"""DS eval / value layer — structural governance scoring for completed runs.

Types only at this stage (Slice 7, PR1): the `EvalGate` protocol, the persisted
`EvalResult`, and the `EvalContext` a gate scores from. Gate implementations and
engine wiring land in later PRs; importing this package has no effect on a run
until an `EvalGate` is configured on `EngineConfig`.
"""

from __future__ import annotations

from dynamic_subgraphs.eval.types import (
    EVAL_SCHEMA_VERSION,
    QUALITY_FLOOR_DEFAULT,
    ComponentVerdict,
    Dimension,
    EvalContext,
    EvalGate,
    EvalReference,
    EvalResult,
    EvalTags,
    Origin,
    OverallVerdict,
    RunFingerprint,
    ScoreComponent,
    ScoreMethod,
    value_per_ktok,
    value_per_usd,
)

__all__ = [
    "EVAL_SCHEMA_VERSION",
    "QUALITY_FLOOR_DEFAULT",
    "ComponentVerdict",
    "Dimension",
    "EvalContext",
    "EvalGate",
    "EvalReference",
    "EvalResult",
    "EvalTags",
    "Origin",
    "OverallVerdict",
    "RunFingerprint",
    "ScoreComponent",
    "ScoreMethod",
    "value_per_ktok",
    "value_per_usd",
]
