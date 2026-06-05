"""Exact per-run token usage — a small leaf module both the engine and the eval
layer import without a cycle.

`TokenUsage` is surfaced by `engine.py` (on `RunResult`) and is positioned for
the eval layer (`dynamic_subgraphs.eval`) to consume as well. Keeping it in its
own leaf module — depending only on LangChain — means neither `engine.py` nor
`eval` has to import the other to share it, which keeps the planned
`EngineConfig.eval_gate` wiring cycle-free when it lands. `engine.py` re-exports
it, so `from dynamic_subgraphs import TokenUsage` and
`from dynamic_subgraphs.engine import TokenUsage` are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.callbacks import UsageMetadataCallbackHandler


@dataclass
class TokenUsage:
    """Exact token usage for one run, read from provider-reported usage.

    Sourced from LangChain's `UsageMetadataCallbackHandler` (the providers'
    own counts — not an estimate), aggregated across every LLM call in the run
    (planner + workers + reducers + subagents). `by_model` keeps the per-model
    breakdown, which matters for hybrid runs that mix providers.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def from_handler(cls, handler: UsageMetadataCallbackHandler) -> TokenUsage:
        by_model: dict[str, dict[str, int]] = {}
        ti = to = tt = 0
        for name, usage in (handler.usage_metadata or {}).items():
            i = int(usage.get("input_tokens", 0))
            o = int(usage.get("output_tokens", 0))
            t = int(usage.get("total_tokens", i + o))
            by_model[name] = {
                "input_tokens": i,
                "output_tokens": o,
                "total_tokens": t,
            }
            ti, to, tt = ti + i, to + o, tt + t
        return cls(
            input_tokens=ti, output_tokens=to, total_tokens=tt, by_model=by_model
        )
