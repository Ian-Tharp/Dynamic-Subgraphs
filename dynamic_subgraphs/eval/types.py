"""Typed seams for the DS eval / value layer (Slice 7).

Dynamic Subgraphs owns *structural / operational* governance scoring: is a
completed run a **valid, parsimonious, on-budget, grounded, successfully
executed** graph/trajectory? These types are the vocabulary that makes the
`runs/` corpus comparable — `EvalResult` is the persisted unit
(``runs/<id>/eval.json``) and `EvalContext` is what a gate scores from.

No gate *implementations* live here (those arrive in later PRs); this module is
types only, so it is pure-additive and OFF by default — nothing scores a run
until an `EvalGate` is configured on `EngineConfig`.

Layer boundary: DS scores graph/trajectory governance. Deep *semantic*
answer-quality judging is CORE's responsibility, not DS's; the optional
`goal_completion` LLM-judge dimension is a deliberate bridge, never a permanent
DS mandate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from dynamic_subgraphs.types import RunStatus

if TYPE_CHECKING:
    from app.models import GraphSpec
    from app.runtime import ExecutionResult
    from dynamic_subgraphs.usage import TokenUsage

EVAL_SCHEMA_VERSION = 1
# Below this quality a run cannot "win" a benchmark comparison regardless of how
# cheap it is — the guard against cheap-but-bad runs inverting the verdict.
QUALITY_FLOOR_DEFAULT = 0.6

# Which arm produced the run, for benchmark grouping.
Origin = Literal["invented", "authored", "routed"]
# The four scored dimensions. Three are deterministic/free; goal_completion
# needs a reference or the (bridge) LLM judge.
Dimension = Literal["plan_validity", "grounding", "goal_completion", "cost_efficiency"]
# Per-component outcome. "neutral" is the shared-constant used in paired mode so
# an inapplicable dimension doesn't renormalize one arm's denominator.
ComponentVerdict = Literal["pass", "weak", "fail", "skipped", "neutral"]
ScoreMethod = Literal["deterministic", "heuristic", "llm_judge", "reference"]
OverallVerdict = Literal["pass", "weak", "fail", "skipped"]


def value_per_ktok(quality: float, total_tokens: int) -> float | None:
    """Quality per 1k tokens — the provider-stable value axis.

    `None` when no tokens were spent. Preferred over `value_per_usd` for ranking
    because real runs can cost ~$0.001, where quality/cost explodes near zero.
    """
    if total_tokens <= 0:
        return None
    return quality / (total_tokens / 1000.0)


def value_per_usd(quality: float, cost_usd: float | None) -> float | None:
    """Quality per dollar — DIAGNOSTIC ONLY, never the benchmark verdict.

    `None` when cost is unknown (no price book) or zero. Unstable as cost → 0,
    which is exactly why it must not decide anything.
    """
    if cost_usd is None or cost_usd <= 0:
        return None
    return quality / cost_usd


class EvalReference(BaseModel):
    """Per-task gold/checklist a deterministic or judge gate scores against.

    Reference-anchored scoring (coverage of required points) is how
    `goal_completion` stays out of free-form "is this good?" preference.
    """

    checklist: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    grounding_required: bool = False


@dataclass(frozen=True)
class EvalTags:
    """Benchmark/run tags threaded through `run()` into the gate.

    The gate never invents these; the caller (or benchmark harness) supplies
    them so runs of the same logical task across arms are comparable.
    """

    task_id: str | None = None
    origin: Origin = "invented"
    reference: EvalReference | None = None


@dataclass(frozen=True)
class EvalContext:
    """Everything a gate needs to score one *completed* run.

    Assembled after the run finishes and after token/cost are known, so a gate
    is a pure function of the run's artifacts — it never drives control flow.
    """

    run_id: str
    spec: GraphSpec
    result: ExecutionResult
    prompt: str | None
    status: RunStatus
    usage: TokenUsage
    cost: float | None
    origin: Origin = "invented"
    task_id: str | None = None
    reference: EvalReference | None = None
    # Lets a gate split planner tokens out of usage.by_model for honest cost
    # attribution (a template cache would eliminate per-run planning).
    planner_model_name: str | None = None


class ScoreComponent(BaseModel):
    """One rubric dimension's score, with how it was produced."""

    dimension: Dimension
    score: float = Field(ge=0.0, le=1.0)
    verdict: ComponentVerdict
    method: ScoreMethod
    applicable: bool = True
    rationale: str = ""
    # Numeric provenance only (e.g. {"node_count": 3}); a mixed-type union
    # breaks JSON round-trip stability, so non-numeric signals go in fields.
    signals: dict[str, float] = Field(default_factory=dict)


class RunFingerprint(BaseModel):
    """Denormalized, comparable shape of a run — the corpus grouping key.

    Lets a reader answer "invented vs authored on cost-adjusted quality" from
    `runs/*/eval.json` alone, and is the motif key for the future template
    library. `topology_signature` is shape-aware (not a kind multiset);
    `instruction_sha256` is kept separate so same-shape/different-instruction
    graphs don't collide.
    """

    graph_id: str
    goal: str
    prompt_sha256: str
    node_count: int
    edge_count: int
    max_depth: int
    node_kind_histogram: dict[str, int] = Field(default_factory=dict)
    has_grounded_tool: bool = False
    planner_model: str | None = None
    worker_model: str | None = None
    topology_signature: str = ""
    instruction_sha256: str = ""
    origin: Origin = "invented"


class EvalResult(BaseModel):
    """The persisted score for one run — written to ``runs/<id>/eval.json``.

    Persists scores + fingerprint + economics only; never re-embeds the run's
    state values or trace (those already live in output.json / trace.jsonl).
    """

    schema_version: int = EVAL_SCHEMA_VERSION
    run_id: str
    scored_at: datetime
    gate: str
    rubric_id: str = "ds.default.v1"
    weights: dict[str, float] = Field(default_factory=dict)
    applicable_dimensions: list[str] = Field(default_factory=list)
    quality: float = Field(ge=0.0, le=1.0)
    components: list[ScoreComponent] = Field(default_factory=list)
    overall_verdict: OverallVerdict = "skipped"
    # Economics — copied from context, never recomputed here.
    cost_usd: float | None = None
    planner_cost_usd: float | None = None
    worker_cost_usd: float | None = None
    total_tokens: int = 0
    # Value axes (see the helpers above for which one decides the verdict).
    value_per_ktok: float | None = None
    value_per_usd: float | None = None
    quality_floor_met: bool = False
    fingerprint: RunFingerprint | None = None
    # Determinism / circularity guards.
    deterministic: bool = True
    judge_model: str | None = None
    judge_cost: float | None = None
    judge_fired: bool = False
    errors: list[dict[str, Any]] = Field(default_factory=list)


@runtime_checkable
class EvalGate(Protocol):
    """Scores a completed run. Implementations arrive in later PRs.

    A gate is post-hoc and inert to control flow — it returns a persisted
    number, never a stop/replan/ask decision. That is the hard boundary from
    `app.supervisor.iteration.LlmIterationDecider` (an in-loop acceptance gate);
    a gate must not import or call it.
    """

    name: str

    def evaluate(self, context: EvalContext) -> EvalResult: ...
