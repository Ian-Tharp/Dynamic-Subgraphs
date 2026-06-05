"""DS eval / value layer — structural governance scoring for completed runs.

Exports the `EvalGate` protocol, the deterministic `DeterministicEvalGate`
implementation (and its `build_deterministic_eval_gate` factory), the persisted
`EvalResult`, and the `EvalContext` a gate scores from. The gate is off by
default: importing this package has no effect on a run until an `EvalGate` is
configured on `EngineConfig` (engine wiring lands in a later slice).
"""

from __future__ import annotations

from dynamic_subgraphs.eval.gates import (
    DeterministicEvalGate,
    build_deterministic_eval_gate,
)
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
    "DeterministicEvalGate",
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
    "build_deterministic_eval_gate",
    "value_per_ktok",
    "value_per_usd",
]
