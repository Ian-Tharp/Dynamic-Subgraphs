"""Deterministic structural eval gate (Slice 7, PR2).

The default `EvalGate`: it scores a *completed* run into an `EvalResult` using
only the run's own artifacts — no LLM judge, no tokens, byte-stable across
re-runs. It owns the **structural / operational** half of DS's eval mandate:
*is this a valid, parsimonious, on-budget, grounded, successfully-executed
graph?* Deep semantic answer-quality is CORE's responsibility, not DS's (see the
package docstring in `dynamic_subgraphs.eval`).

OFF by default — importing this module does nothing to a run until an `EvalGate`
is configured on `EngineConfig` (the engine wiring lands in a following slice).
The gate is a pure function of the run's artifacts and never touches control
flow: it returns a persisted number, never a stop/replan decision.

Four rubric dimensions, all deterministic:

- ``plan_validity`` — the spec re-validates against the registry, plus budget
  adherence (node / llm-call counts vs the granted `GraphBudget`).
- ``grounding`` — when the task/plan calls for it, did a `web_search` actually
  produce results? Inapplicable runs drop out (they don't dilute the score).
- ``goal_completion`` — reference-checklist coverage when an `EvalReference` is
  supplied, else a clearly-labelled low-confidence keyword heuristic.
- ``cost_efficiency`` — a token-parsimony proxy (full marks under a free band,
  decaying after); never a dollar figure, so it is provider-pricing-stable.

The blended ``quality`` is for single-run display; the benchmark verdict is
reported per-headline (see the design spec), so the weights below are not
load-bearing for a thesis decision. They are persisted on every `EvalResult` so
any re-weighting stays reproducible.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from app.models import GraphSpec, NodeKind
from app.registry import RegistryValidationError, validate_graph_spec
from dynamic_subgraphs.eval.types import (
    QUALITY_FLOOR_DEFAULT,
    ComponentVerdict,
    EvalContext,
    EvalReference,
    EvalResult,
    OverallVerdict,
    RunFingerprint,
    ScoreComponent,
    value_per_ktok,
    value_per_usd,
)

# ds.default.v1 rubric weights. DECISION (design spec §10) — proposed, pending
# ratification; persisted on every result so re-weighting is reproducible.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "goal_completion": 0.40,
    "grounding": 0.25,
    "plan_validity": 0.20,
    "cost_efficiency": 0.15,
}
_RUBRIC_ID = "ds.default.v1"

# cost_efficiency proxy: full marks up to the free band, decaying linearly to
# zero across the scale band after it. A parsimony signal, not a price.
_EFF_FREE_TOKENS = 2_000
_EFF_SCALE_TOKENS = 30_000

# A grounded web_search output carries this marker (see app/runtime/tools.py).
_GROUNDED_TOOL = "web_search"

# Verdict thresholds, shared by component and overall scoring.
_PASS_AT = 0.75
_WEAK_AT = 0.50

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "be",
        "vs",
        "versus",
        "two",
        "compare",
        "between",
        "which",
        "what",
        "how",
        "why",
        "from",
        "by",
        "at",
        "as",
        "that",
        "this",
        "then",
        "recommend",
        "one",
        "about",
        "into",
        "over",
    }
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _component_verdict(score: float) -> ComponentVerdict:
    if score >= _PASS_AT:
        return "pass"
    if score >= _WEAK_AT:
        return "weak"
    return "fail"


def _overall_verdict(quality: float) -> OverallVerdict:
    if quality >= _PASS_AT:
        return "pass"
    if quality >= _WEAK_AT:
        return "weak"
    return "fail"


def _collect_text(value: Any) -> list[str]:
    """Flatten a state value into its string leaves (answers, snippets, etc.)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_collect_text(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_collect_text(item))
        return out
    return []


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    seen: dict[str, None] = {}
    for word in words:
        if len(word) >= 3 and word not in _STOPWORDS:
            seen.setdefault(word, None)
    return list(seen)


def _web_search_nodes(spec: GraphSpec) -> int:
    count = 0
    for node in spec.nodes:
        if node.kind == NodeKind.TOOL_CALL:
            tool = str(node.params.get("tool") or node.params.get("name") or "")
            if tool == _GROUNDED_TOOL:
                count += 1
    return count


def _grounded_outputs(values: dict[str, Any]) -> int:
    """Count run outputs that are real `web_search` results (non-empty)."""
    count = 0
    for value in values.values():
        if isinstance(value, dict) and value.get("tool") == _GROUNDED_TOOL:
            results = value.get("results") or []
            answer = value.get("answer")
            if (isinstance(results, list) and results) or answer:
                count += 1
    return count


def _topology_signature(spec: GraphSpec) -> str:
    """A shape-aware hash: kind-labelled nodes with in/out degree + edge kinds.

    Invariant to node ids and params (so the same shape collides) but sensitive
    to edge structure and degree (so a `branch` DAG never collides with a linear
    chain of the same kinds). `params["instruction"]` is hashed separately.
    """
    kind_by_id = {node.id: node.kind.value for node in spec.nodes}
    in_deg = {node.id: 0 for node in spec.nodes}
    out_deg = {node.id: 0 for node in spec.nodes}
    edge_pairs: list[str] = []
    for edge in spec.edges:
        if edge.from_ in out_deg:
            out_deg[edge.from_] += 1
        if edge.to in in_deg:
            in_deg[edge.to] += 1
        src = kind_by_id.get(edge.from_, edge.from_)
        dst = kind_by_id.get(edge.to, edge.to)
        edge_pairs.append(f"{src}>{dst}")
    node_sigs = sorted(
        f"{kind_by_id[n.id]}:{in_deg[n.id]}:{out_deg[n.id]}" for n in spec.nodes
    )
    canonical = "|".join(node_sigs) + "||" + "|".join(sorted(edge_pairs))
    return _sha256(canonical)


def _instruction_signature(spec: GraphSpec) -> str:
    parts: list[str] = []
    for node in spec.nodes:
        instruction = node.params.get("instruction")
        if isinstance(instruction, str):
            parts.append(instruction.strip())
    return _sha256("\n".join(parts))


def _graph_depth(spec: GraphSpec) -> int:
    """Longest directed path length over the real nodes (cycle-guarded)."""
    adjacency: dict[str, list[str]] = {node.id: [] for node in spec.nodes}
    for edge in spec.edges:
        if edge.from_ in adjacency:
            adjacency[edge.from_].append(edge.to)
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(node_id: str) -> int:
        if node_id not in adjacency:
            return 0
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            return 0
        visiting.add(node_id)
        best = 0
        for nxt in adjacency[node_id]:
            best = max(best, 1 + depth(nxt))
        visiting.discard(node_id)
        memo[node_id] = best
        return best

    return max((depth(node.id) for node in spec.nodes), default=0)


def _planner_tokens(
    usage_by_model: dict[str, dict[str, int]], planner: str
) -> int | None:
    """Tokens attributed to the planner model, tolerating dated snapshots."""
    if planner in usage_by_model:
        return int(usage_by_model[planner].get("total_tokens", 0))
    prefixes = [name for name in usage_by_model if name.startswith(planner)]
    if prefixes:
        longest = max(prefixes, key=len)
        return int(usage_by_model[longest].get("total_tokens", 0))
    return None


def _split_cost(
    cost: float | None,
    total_tokens: int,
    usage_by_model: dict[str, dict[str, int]],
    planner: str | None,
) -> tuple[float | None, float | None]:
    """Proportionally split total cost into planner vs worker by token share."""
    if cost is None or total_tokens <= 0 or planner is None:
        return None, None
    planner_tokens = _planner_tokens(usage_by_model, planner)
    if planner_tokens is None:
        return None, None
    planner_cost = round(cost * (planner_tokens / total_tokens), 6)
    return planner_cost, round(cost - planner_cost, 6)


def _score_plan_validity(spec: GraphSpec) -> ScoreComponent:
    try:
        validate_graph_spec(spec)
        valid = True
    except RegistryValidationError:
        valid = False
    nodes = len(spec.nodes)
    edges = len(spec.edges)
    llm_calls = sum(1 for n in spec.nodes if n.kind == NodeKind.LLM_CALL)
    budget = spec.budget
    within_budget = nodes <= budget.max_nodes and llm_calls <= budget.max_llm_calls
    if not valid:
        score, rationale = 0.0, "Spec failed registry re-validation."
    elif within_budget:
        score, rationale = 1.0, "Valid and within the granted budget."
    else:
        score, rationale = 0.6, "Valid but over the granted budget."
    return ScoreComponent(
        dimension="plan_validity",
        score=_clamp01(score),
        verdict=_component_verdict(score),
        method="deterministic",
        rationale=rationale,
        signals={
            "node_count": float(nodes),
            "edge_count": float(edges),
            "llm_call_count": float(llm_calls),
            "budget_max_nodes": float(budget.max_nodes),
            "budget_max_llm_calls": float(budget.max_llm_calls),
            "valid": 1.0 if valid else 0.0,
        },
    )


def _score_grounding(
    applicable: bool, grounded_outputs: int, web_search_nodes: int
) -> ScoreComponent:
    signals = {
        "web_search_nodes": float(web_search_nodes),
        "grounded_outputs": float(grounded_outputs),
    }
    if not applicable:
        return ScoreComponent(
            dimension="grounding",
            score=0.0,
            verdict="skipped",
            method="deterministic",
            applicable=False,
            rationale="No grounding required by the reference or the plan.",
            signals=signals,
        )
    grounded = grounded_outputs > 0
    return ScoreComponent(
        dimension="grounding",
        score=1.0 if grounded else 0.0,
        verdict="pass" if grounded else "fail",
        method="deterministic",
        rationale=(
            "Grounding required; a web_search produced results."
            if grounded
            else "Grounding required but no grounded tool output was found."
        ),
        signals=signals,
    )


def _score_goal_completion(
    output_text: str, goal: str, reference: EvalReference | None
) -> ScoreComponent:
    text = output_text.lower()
    if reference is not None and (reference.checklist or reference.must_include):
        required = [*reference.checklist, *reference.must_include]
        covered = sum(1 for item in required if item.lower() in text)
        penalty = sum(1 for bad in reference.must_not_include if bad.lower() in text)
        base = covered / len(required) if required else 0.0
        score = _clamp01(base - 0.34 * penalty)
        return ScoreComponent(
            dimension="goal_completion",
            score=score,
            verdict=_component_verdict(score),
            method="reference",
            rationale=f"Reference coverage: {covered}/{len(required)} required points.",
            signals={
                "covered": float(covered),
                "required": float(len(required)),
                "must_not_hits": float(penalty),
            },
        )
    keywords = _keywords(goal)
    hits = sum(1 for word in keywords if word in text)
    score = _clamp01(hits / len(keywords)) if keywords else 0.0
    return ScoreComponent(
        dimension="goal_completion",
        score=score,
        verdict=_component_verdict(score),
        method="heuristic",
        rationale="No reference supplied; low-confidence goal-keyword overlap.",
        signals={
            "goal_keyword_hits": float(hits),
            "goal_keywords": float(len(keywords)),
        },
    )


def _score_cost_efficiency(total_tokens: int) -> ScoreComponent:
    if total_tokens <= _EFF_FREE_TOKENS:
        score = 1.0
    else:
        score = _clamp01(1.0 - (total_tokens - _EFF_FREE_TOKENS) / _EFF_SCALE_TOKENS)
    return ScoreComponent(
        dimension="cost_efficiency",
        score=score,
        verdict=_component_verdict(score),
        method="deterministic",
        rationale="Token-parsimony proxy (provider-pricing-stable).",
        signals={"total_tokens": float(total_tokens)},
    )


def _fingerprint(context: EvalContext, has_grounded: bool) -> RunFingerprint:
    spec = context.spec
    histogram: dict[str, int] = {}
    for node in spec.nodes:
        histogram[node.kind.value] = histogram.get(node.kind.value, 0) + 1
    worker = next(
        (m for m in context.usage.by_model if m != context.planner_model_name), None
    )
    return RunFingerprint(
        graph_id=spec.graph_id,
        goal=spec.goal,
        prompt_sha256=_sha256(context.prompt or ""),
        node_count=len(spec.nodes),
        edge_count=len(spec.edges),
        max_depth=_graph_depth(spec),
        node_kind_histogram=histogram,
        has_grounded_tool=has_grounded,
        planner_model=spec.metadata.planner_model or context.planner_model_name,
        worker_model=worker,
        topology_signature=_topology_signature(spec),
        instruction_sha256=_instruction_signature(spec),
        origin=context.origin,
    )


class DeterministicEvalGate:
    """The default, token-free structural scorer (``deterministic@v1``).

    Construct once and pass to the engine (a following slice):
    ``EngineConfig(eval_gate=DeterministicEvalGate())``. Scoring is a pure
    function of the completed run, so re-scoring the same run is byte-stable
    (only ``scored_at`` differs).
    """

    name = "deterministic@v1"

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        quality_floor: float = QUALITY_FLOOR_DEFAULT,
        rubric_id: str = _RUBRIC_ID,
    ) -> None:
        self._weights = dict(weights or _DEFAULT_WEIGHTS)
        self._quality_floor = quality_floor
        self._rubric_id = rubric_id

    def _blend(self, components: list[ScoreComponent]) -> float:
        numerator = 0.0
        denominator = 0.0
        for component in components:
            if not component.applicable:
                continue
            weight = self._weights.get(component.dimension, 0.0)
            numerator += weight * component.score
            denominator += weight
        return _clamp01(numerator / denominator) if denominator > 0 else 0.0

    def evaluate(self, context: EvalContext) -> EvalResult:
        spec = context.spec
        usage = context.usage
        total_tokens = usage.total_tokens
        values: dict[str, Any] = dict(context.result.state.get("values", {}) or {})
        grounded_outputs = _grounded_outputs(values)
        has_grounded = grounded_outputs > 0
        fingerprint = _fingerprint(context, has_grounded)
        planner_cost, worker_cost = _split_cost(
            context.cost, total_tokens, usage.by_model, context.planner_model_name
        )
        scored_at = datetime.now(UTC)

        ok = context.result.ok and context.status == "ok"
        if not ok:
            paused = context.result.paused or context.status == "paused"
            return EvalResult(
                run_id=context.run_id,
                scored_at=scored_at,
                gate=self.name,
                rubric_id=self._rubric_id,
                weights=dict(self._weights),
                applicable_dimensions=[],
                quality=0.0,
                components=[],
                overall_verdict="skipped" if paused else "fail",
                cost_usd=context.cost,
                planner_cost_usd=planner_cost,
                worker_cost_usd=worker_cost,
                total_tokens=total_tokens,
                value_per_ktok=value_per_ktok(0.0, total_tokens),
                value_per_usd=value_per_usd(0.0, context.cost),
                quality_floor_met=False,
                fingerprint=fingerprint,
                deterministic=True,
            )

        output_text = "\n".join(_collect_text(values))
        web_search_nodes = _web_search_nodes(spec)
        grounding_applicable = (
            bool(context.reference and context.reference.grounding_required)
            or web_search_nodes > 0
        )
        components = [
            _score_plan_validity(spec),
            _score_grounding(grounding_applicable, grounded_outputs, web_search_nodes),
            _score_goal_completion(output_text, spec.goal, context.reference),
            _score_cost_efficiency(total_tokens),
        ]
        quality = self._blend(components)
        return EvalResult(
            run_id=context.run_id,
            scored_at=scored_at,
            gate=self.name,
            rubric_id=self._rubric_id,
            weights=dict(self._weights),
            applicable_dimensions=[c.dimension for c in components if c.applicable],
            quality=quality,
            components=components,
            overall_verdict=_overall_verdict(quality),
            cost_usd=context.cost,
            planner_cost_usd=planner_cost,
            worker_cost_usd=worker_cost,
            total_tokens=total_tokens,
            value_per_ktok=value_per_ktok(quality, total_tokens),
            value_per_usd=value_per_usd(quality, context.cost),
            quality_floor_met=quality >= self._quality_floor,
            fingerprint=fingerprint,
            deterministic=True,
        )


def build_deterministic_eval_gate(
    *,
    weights: dict[str, float] | None = None,
    quality_floor: float = QUALITY_FLOOR_DEFAULT,
) -> DeterministicEvalGate:
    """Factory mirroring the runtime's other builders."""
    return DeterministicEvalGate(weights=weights, quality_floor=quality_floor)
