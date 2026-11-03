"""Pre-flight cost bound: a worst-case token / dollar figure for a plan.

Computed from a static price table and per-node token caps — never from a
network price feed and never by calling a model. The bound answers "if every
LLM-consuming node runs and every call uses its full token cap, what is the
most this plan can spend?":

- ``llm_call`` / ``spawn_subagent`` / ``reduce(llm_summarize)``: one call each.
- ``parallel_map`` with an ``llm_call`` child: up to the effective
  ``max_fanout`` calls.
- ``spawn_subgraph`` / ``gauntlet``: open-ended (the child is planned at run
  time), so the bound adds slack calls priced at the most expensive model in
  play, up to the effective ``max_llm_calls``.

The runtime ledger already makes ``max_llm_calls`` a hard ceiling on actual
calls, so the bound keeps only the ``max_llm_calls`` most expensive calls.

The figure is an upper bound **under the declared token caps**. Output tokens
can be enforced by the provider (``max_tokens``); input size depends on upstream
values. The provenance receipt reports the bound next to the actual spend so an
overrun of the caps is visible.

Layering: imports only ``app.models``; ``app.policy`` references these types.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from app.models.graph_spec import GraphSpec, NodeSpec
from app.models.node_kinds import NodeKind


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million tokens."""

    input_per_1m: float
    output_per_1m: float

    def __post_init__(self) -> None:
        if self.input_per_1m < 0 or self.output_per_1m < 0:
            raise ValueError("ModelPrice values must be >= 0")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_per_1m
            + output_tokens / 1_000_000 * self.output_per_1m
        )


@dataclass(frozen=True)
class TokenCap:
    """The most tokens one model call is assumed to consume."""

    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("TokenCap values must be >= 0")

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


PriceInput = ModelPrice | Mapping[str, float]


def _coerce_price(value: PriceInput) -> ModelPrice:
    if isinstance(value, ModelPrice):
        return value
    return ModelPrice(
        input_per_1m=float(value.get("input_per_1m", 0.0)),
        output_per_1m=float(value.get("output_per_1m", 0.0)),
    )


@dataclass(frozen=True)
class CostModel:
    """Static price table + token caps used to bound a plan before it runs.

    Attributes:
        prices: ``{model_name: ModelPrice}``. Also accepts the SDK's
            ``{"input_per_1m": ..., "output_per_1m": ...}`` dict form. A model
            matches by exact name, then by the longest name that is a prefix
            (so ``gpt-5.4-nano`` prices ``gpt-5.4-nano-2026-03-17``).
        default_model: the model a node without ``params.model`` runs on.
        kind_models: per-kind override of ``default_model`` (e.g. the reducer
            model for ``reduce``, the subagent model for ``spawn_subagent``).
        extra_models: other models the run can reach (planner, judge). They are
            candidates for pricing open-ended ``spawn_subgraph``/``gauntlet``
            slack, which uses the most expensive model in play.
        default_cap: per-call token cap for every LLM-consuming node.
        kind_caps: per-kind override of ``default_cap``.
    """

    prices: Mapping[str, PriceInput] = field(default_factory=dict)
    default_model: str | None = None
    kind_models: Mapping[NodeKind, str] = field(default_factory=dict)
    extra_models: frozenset[str] = frozenset()
    default_cap: TokenCap = TokenCap(input_tokens=8_000, output_tokens=2_000)
    kind_caps: Mapping[NodeKind, TokenCap] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prices = {str(k): _coerce_price(v) for k, v in self.prices.items()}
        object.__setattr__(self, "prices", MappingProxyType(prices))
        object.__setattr__(
            self, "kind_models", MappingProxyType(dict(self.kind_models))
        )
        object.__setattr__(self, "kind_caps", MappingProxyType(dict(self.kind_caps)))
        object.__setattr__(self, "extra_models", frozenset(self.extra_models))

    def __hash__(self) -> int:  # mappings are unhashable; hash the canonical form
        from app.governance.hashing import canonical_json

        return hash(canonical_json(self.as_dict()))

    def price_for(self, model: str) -> ModelPrice | None:
        prices: Mapping[str, ModelPrice] = self.prices  # type: ignore[assignment]
        if model in prices:
            return prices[model]
        prefixes = [k for k in prices if model.startswith(k)]
        return prices[max(prefixes, key=len)] if prefixes else None

    def cap_for(self, kind: NodeKind) -> TokenCap:
        return self.kind_caps.get(kind, self.default_cap)

    def model_for(self, node_kind: NodeKind, declared: str | None) -> str | None:
        if declared:
            return declared
        return self.kind_models.get(node_kind, self.default_model)

    def with_role_defaults(
        self,
        *,
        default_model: str | None = None,
        kind_models: Mapping[NodeKind, str] | None = None,
        extra_models: Iterable[str] = (),
    ) -> CostModel:
        """Fill unset model routing from the run's role models (explicit wins)."""
        merged_kinds = dict(kind_models or {})
        merged_kinds.update(self.kind_models)
        return CostModel(
            prices=self.prices,
            default_model=self.default_model or default_model,
            kind_models=merged_kinds,
            extra_models=self.extra_models | frozenset(extra_models),
            default_cap=self.default_cap,
            kind_caps=self.kind_caps,
        )

    def as_dict(self) -> dict[str, Any]:
        prices: Mapping[str, ModelPrice] = self.prices  # type: ignore[assignment]
        return {
            "prices": {
                k: {"input_per_1m": v.input_per_1m, "output_per_1m": v.output_per_1m}
                for k, v in sorted(prices.items())
            },
            "default_model": self.default_model,
            "kind_models": {k.value: v for k, v in sorted(self.kind_models.items())},
            "extra_models": sorted(self.extra_models),
            "default_cap": [
                self.default_cap.input_tokens,
                self.default_cap.output_tokens,
            ],
            "kind_caps": {
                k.value: [v.input_tokens, v.output_tokens]
                for k, v in sorted(self.kind_caps.items())
            },
        }


@dataclass(frozen=True)
class CallEstimate:
    """One worst-case model call the plan may make."""

    node_id: str
    model: str | None
    tokens: int
    usd: float | None


@dataclass(frozen=True)
class CostBound:
    """Worst-case spend for a plan (see module docstring for the rules).

    ``usd`` is None when some counted call's model has no price.
    """

    llm_calls: int
    tokens: int
    usd: float | None
    open_ended: bool
    unpriced_models: tuple[str, ...]
    calls: tuple[CallEstimate, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "usd": None if self.usd is None else round(self.usd, 6),
            "open_ended": self.open_ended,
            "unpriced_models": list(self.unpriced_models),
        }


def _llm_calls_for(node: NodeSpec, max_fanout: int) -> tuple[int, str | None, bool]:
    """(calls, declared model, open_ended) for one node."""
    params = node.params
    kind = node.kind
    if kind in (NodeKind.LLM_CALL, NodeKind.SPAWN_SUBAGENT):
        return 1, params.get("model"), False
    if kind == NodeKind.REDUCE and params.get("strategy") == "llm_summarize":
        return 1, params.get("model"), False
    if kind == NodeKind.PARALLEL_MAP and params.get("child_kind") == "llm_call":
        child = params.get("child_params") or {}
        return max_fanout, child.get("model"), False
    if kind in (NodeKind.SPAWN_SUBGRAPH, NodeKind.GAUNTLET):
        return 0, None, True
    return 0, None, False


def estimate_plan_cost(
    spec: GraphSpec,
    cost_model: CostModel,
    *,
    max_llm_calls: int,
    max_fanout: int,
) -> CostBound:
    """Worst-case tokens and USD for ``spec`` under ``cost_model``.

    ``max_llm_calls`` / ``max_fanout`` are the *effective* limits (host policy
    ∧ planner request), the same ones the runtime enforces.
    """
    calls: list[CallEstimate] = []
    open_ended = False
    unpriced: set[str] = set()

    def _estimate(node_id: str, kind: NodeKind, model: str | None) -> CallEstimate:
        cap = cost_model.cap_for(kind)
        price = cost_model.price_for(model) if model else None
        if price is None:
            unpriced.add(model or "<unspecified>")
        usd = price.cost(cap.input_tokens, cap.output_tokens) if price else None
        return CallEstimate(node_id=node_id, model=model, tokens=cap.total, usd=usd)

    for node in spec.nodes:
        count, declared, node_open = _llm_calls_for(node, max_fanout)
        open_ended = open_ended or node_open
        if count:
            model = cost_model.model_for(node.kind, declared)
            est = _estimate(node.id, node.kind, model)
            calls.extend([est] * count)

    if open_ended and max_llm_calls > 0:
        # Child plans are unknown: assume every remaining call runs on the most
        # expensive model the run can reach, at the largest cap.
        pool = {
            m
            for m in (
                {cost_model.default_model}
                | set(cost_model.kind_models.values())
                | set(cost_model.extra_models)
                | {c.model for c in calls}
            )
            if m
        }
        priced = [(m, cost_model.price_for(m)) for m in sorted(pool)]
        cap = max(
            [cost_model.default_cap, *cost_model.kind_caps.values()],
            key=lambda c: c.total,
        )
        if not pool or any(p is None for _, p in priced):
            unpriced.update(m for m, p in priced if p is None)
            if not pool:
                unpriced.add("<unspecified>")
            slack = CallEstimate("<open_ended>", None, cap.total, None)
        else:
            worst = max(
                priced,
                key=lambda mp: mp[1].cost(cap.input_tokens, cap.output_tokens),  # type: ignore[union-attr]
            )
            assert worst[1] is not None
            slack = CallEstimate(
                "<open_ended>",
                worst[0],
                cap.total,
                worst[1].cost(cap.input_tokens, cap.output_tokens),
            )
        calls.extend([slack] * max_llm_calls)

    limit = max(0, max_llm_calls)
    # The runtime refuses call N+1, so only the `limit` most expensive calls can
    # happen. Bound tokens and dollars separately (each takes its own worst set).
    by_tokens = sorted(calls, key=lambda c: c.tokens, reverse=True)[:limit]
    tokens = sum(c.tokens for c in by_tokens)
    if unpriced:
        usd: float | None = None
    else:
        by_usd = sorted(calls, key=lambda c: c.usd or 0.0, reverse=True)[:limit]
        usd = sum(c.usd or 0.0 for c in by_usd)

    return CostBound(
        llm_calls=min(len(calls), limit),
        tokens=tokens,
        usd=usd,
        open_ended=open_ended,
        unpriced_models=tuple(sorted(unpriced)),
        calls=tuple(calls),
    )
