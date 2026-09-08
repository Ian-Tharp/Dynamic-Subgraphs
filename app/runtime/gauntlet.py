"""gauntlet: a node that runs a bounded builder/critic loop against a reference.

Matt Shumer's "gauntlet loop", as a governed node kind. Each round:

1. a **builder** child graph is planned and run one level deeper (through the
   same `ChildLauncher` `spawn_subgraph` uses), briefed with the sub-goal plus
   the single largest gap from the previous round;
2. a **critic** with a fresh context compares the builder's artifact against a
   cached reference *blind* — the A/B mapping is sealed by a coin flip outside
   the critic's context and recorded in the round log — and returns only the
   winner and one gap;
3. loop until ours wins, or a stop condition fires: the round ceiling, the
   parent's remaining LLM/wall budget, or the same gap repeated three times.

The outer graph stays an acyclic DAG: the cycle lives inside this runner, the
same way `parallel_map` hides its fan-out. Every builder round is a recorded
child run (`<parent>__gl_<name>_r<round>`); the round log with verdicts, sealed
mappings, and stop reason is returned as the node's output so it lands in the
parent's recorded `output.json`.

Losing to the bar is the expected outcome of a well-set bar; the node returns
the best artifact so far and an honest `stop_reason` rather than reinterpreting
a boundary as a win.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

from app.errors import DynamicSubgraphsError
from app.registry.validator import MAX_DEPTH_CEILING
from app.runtime.runners import NodeRunner
from app.runtime.state import render_value_for_prompt
from app.runtime.subgraph import SubgraphDepthExceeded

if TYPE_CHECKING:
    from app.runtime.budget_ledger import BudgetLedger
    from app.runtime.subgraph import ChildLauncher

logger = logging.getLogger(__name__)

StopReason = Literal[
    "won", "max_rounds", "repeated_gap", "budget", "deadline", "builder_failed"
]


class GauntletError(DynamicSubgraphsError, RuntimeError):
    """Base for gauntlet failures — each halts the parent node fail-closed."""


class GauntletNoArtifact(GauntletError):
    """The loop stopped before any builder round produced an artifact."""


@dataclass(frozen=True)
class CriticVerdict:
    """What a critic returns: the blind winner and the single largest gap."""

    winner: Literal["A", "B"]
    gap: str


class Critic(Protocol):
    """Fresh-context blind judge: sees two artifacts and the goal, nothing else."""

    def __call__(self, a: str, b: str, *, goal: str) -> CriticVerdict: ...


# ---------- critics ----------


CRITIC_SYSTEM_PROMPT = """\
You are a harsh, independent critic. You will be shown two artifacts, A and B,
produced for the same goal. You do not know which is which and must not
speculate about their provenance; judge only the craft in front of you.

Pick the artifact that better achieves the goal. Then name the SINGLE largest
gap in the LOSER — the one change that would most improve it. Be concrete and
specific; one short sentence. Do not list several gaps. Do not soften the
verdict; a tie is not allowed.
"""


class _CriticPayload(BaseModel):
    winner: Literal["A", "B"]
    gap: str = Field(min_length=1)


class _StructuredCriticModel(Protocol):
    def invoke(self, messages: list[Any], /, **kwargs: Any) -> Any: ...


def make_llm_critic(structured_model: _StructuredCriticModel) -> Critic:
    """Build a `Critic` from a chat model configured with
    `with_structured_output(_CriticPayload)` (see `build_provider_critic`)."""

    def _critic(a: str, b: str, *, goal: str) -> CriticVerdict:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = "\n".join(
            [
                f"Goal:\n{goal}",
                "",
                "=== Artifact A ===",
                a,
                "",
                "=== Artifact B ===",
                b,
            ]
        )
        payload = structured_model.invoke(
            [SystemMessage(content=CRITIC_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        if not isinstance(payload, _CriticPayload):
            raise GauntletError(
                f"critic expected _CriticPayload, got {type(payload).__name__}"
            )
        return CriticVerdict(winner=payload.winner, gap=payload.gap)

    return _critic


def build_provider_critic(provider: Any, ref: Any) -> Critic:
    """Build an LLM critic from a `ModelProvider` + `ModelRef` (assembly seam).

    The structured model is built on the first verdict, not at assembly time,
    so wiring a supervisor never constructs a judge model a run may not use.
    """
    built: list[Critic] = []

    def _critic(a: str, b: str, *, goal: str) -> CriticVerdict:
        if not built:
            built.append(
                make_llm_critic(provider.build_structured_output(ref, _CriticPayload))
            )
        return built[0](a, b, goal=goal)

    return _critic


def make_length_critic() -> Critic:
    """Token-free stand-in: the longer artifact wins. For mock runs and tests."""

    def _critic(a: str, b: str, *, goal: str) -> CriticVerdict:
        del goal
        winner: Literal["A", "B"] = "A" if len(a) >= len(b) else "B"
        return CriticVerdict(winner=winner, gap="shorter than the other artifact")

    return _critic


# ---------- the runner ----------


def _seal_mapping(seed: str) -> bool:
    """Coin flip, sealed outside the critic's context. True => ours is A.

    Seeded from the run lineage so a replay reproduces the same blinding.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return random.Random(digest).random() < 0.5


def _same_gap(a: str, b: str) -> bool:
    # ponytail: word-set Jaccard >= 0.6 counts as the same gap; an LLM critic
    # rephrases, so exact match would never fire. Upgrade to a critic-emitted
    # gap label if this proves too loose or too tight.
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= 0.6


def _artifact_text(values: Mapping[str, Any], artifact_key: str) -> str:
    """The builder child's artifact as text: the declared key if present,
    otherwise the whole produced-values dict rendered."""
    if artifact_key in values:
        return render_value_for_prompt(values[artifact_key])
    return render_value_for_prompt(dict(values))


def build_gauntlet_runner(
    launcher: ChildLauncher,
    critic: Critic,
    *,
    depth_ceiling: int = MAX_DEPTH_CEILING,
    ledger_registry: Mapping[str, BudgetLedger] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> NodeRunner:
    """Build the NodeRunner for `gauntlet` nodes.

    Reserves the parent's remaining node/LLM allowance once (like
    `spawn_subgraph`), then spends it round by round: each builder child is
    granted what is left minus one call reserved for its critic, and the loop
    stops on `budget` before launching a round it cannot afford to judge.
    """

    def _runner(state: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
        metadata = state.get("metadata", {}) or {}
        depth = int(metadata.get("graph_depth", 0))
        effective_ceiling = min(
            int(metadata.get("budget_max_depth", depth_ceiling)), depth_ceiling
        )
        if depth + 1 > effective_ceiling:
            raise SubgraphDepthExceeded(
                f"gauntlet at depth {depth} would exceed the nesting "
                f"ceiling {effective_ceiling}"
            )

        parent_run_id = str(metadata.get("run_id", "root"))
        name = str(params["name"])
        sub_goal = str(params["sub_goal"])
        reference_key = str(params["reference_key"])
        artifact_key = str(params.get("artifact_key", "artifact"))
        max_rounds = int(params.get("max_rounds", 3))
        inputs_from = list(params.get("inputs_from", []) or [])
        parent_values = state.get("values", {}) or {}
        if reference_key not in parent_values:
            raise GauntletError(
                f"gauntlet reference_key {reference_key!r} is not in state.values"
            )
        reference = render_value_for_prompt(parent_values[reference_key])
        inputs = {k: parent_values[k] for k in inputs_from if k in parent_values}
        deadline = metadata.get("wall_deadline_monotonic")
        replay_of = metadata.get("replay_of")

        counters = state.get("counters", {}) or {}
        budgets: dict[str, int | None] = {
            "llm_calls": metadata.get("budget_max_llm_calls"),
            "nodes": metadata.get("budget_max_nodes"),
        }
        consumed = {
            "llm_calls": int(counters.get("llm_calls_consumed", 0)),
            "nodes": int(counters.get("nodes_executed", 0)),
        }
        ledger = (
            ledger_registry.get(parent_run_id) if ledger_registry is not None else None
        )
        if ledger is not None:
            grant = ledger.reserve(budgets=budgets, consumed=consumed)
        else:
            grant = {
                key: (None if cap is None else max(0, cap - consumed[key]))
                for key, cap in budgets.items()
            }

        spent = {"llm_calls": 0, "nodes": 0}
        rounds: list[dict[str, Any]] = []
        best: str | None = None
        gap: str | None = None
        gap_repeats = 0
        stop_reason: StopReason = "max_rounds"

        def _remaining(key: str) -> int | None:
            cap = grant[key]
            return None if cap is None else max(0, cap - spent[key])

        try:
            for round_no in range(1, max_rounds + 1):
                if deadline is not None and clock() >= float(deadline):
                    stop_reason = "deadline"
                    break
                llm_left = _remaining("llm_calls")
                # A round needs at least one builder call plus one critic call.
                if llm_left is not None and llm_left < 2:
                    stop_reason = "budget"
                    break
                nodes_left = _remaining("nodes")
                if nodes_left is not None and nodes_left < 1:
                    stop_reason = "budget"
                    break

                brief = sub_goal
                if gap is not None:
                    brief += f"\n\nThe previous attempt lost to the reference. Largest gap: {gap}"
                    if gap_repeats >= 1:
                        brief += (
                            "\nThis gap was reported before. Measure it against the "
                            "artifact before changing anything; if the evidence "
                            "contradicts it, say so instead of complying."
                        )
                if best is not None:
                    brief += f"\n\nPrevious attempt (improve on it):\n{best}"

                child_run_id = f"{parent_run_id}__gl_{name}_r{round_no}"
                result = launcher(
                    brief,
                    run_id=child_run_id,
                    graph_depth=depth + 1,
                    parent_run_id=parent_run_id,
                    inputs=dict(inputs),
                    max_llm_calls=None if llm_left is None else llm_left - 1,
                    max_nodes=nodes_left,
                    max_depth=effective_ceiling,
                    wall_deadline=deadline,
                    replay_of=f"{replay_of}__gl_{name}_r{round_no}"
                    if replay_of
                    else None,
                )
                spent["llm_calls"] += int(result.counters.get("llm_calls_consumed", 0))
                spent["nodes"] += int(result.counters.get("nodes_executed", 0))
                if result.status != "ok":
                    rounds.append(
                        {
                            "round": round_no,
                            "builder": "failed",
                            "error": result.response,
                        }
                    )
                    stop_reason = "builder_failed"
                    break
                artifact = _artifact_text(result.values, artifact_key)

                if _remaining("llm_calls") == 0:
                    best = artifact
                    stop_reason = "budget"
                    rounds.append({"round": round_no, "verdict": "unjudged"})
                    break

                ours_is_a = _seal_mapping(
                    f"{replay_of or parent_run_id}:{name}:{round_no}"
                )
                a, b = (artifact, reference) if ours_is_a else (reference, artifact)
                verdict = critic(a, b, goal=sub_goal)
                spent["llm_calls"] += 1
                ours_won = (verdict.winner == "A") == ours_is_a
                rounds.append(
                    {
                        "round": round_no,
                        "child_run_id": child_run_id,
                        "mapping": {
                            "A": "ours" if ours_is_a else "bar",
                            "B": "bar" if ours_is_a else "ours",
                        },
                        "winner": "ours" if ours_won else "bar",
                        "gap": verdict.gap,
                    }
                )
                best = artifact
                if ours_won:
                    stop_reason = "won"
                    break
                if gap is not None and _same_gap(gap, verdict.gap):
                    gap_repeats += 1
                else:
                    gap_repeats = 0
                gap = verdict.gap
                if gap_repeats >= 2:
                    stop_reason = "repeated_gap"
                    break
        finally:
            if ledger is not None:
                ledger.refund(reserved=grant, spent=spent)

        if best is None:
            last_error = rounds[-1].get("error") if rounds else None
            detail = f"; last builder error: {last_error}" if last_error else ""
            raise GauntletNoArtifact(
                f"gauntlet {name!r} produced no artifact "
                f"(stop_reason={stop_reason}{detail})"
            )
        return {
            "result": {
                "artifact": best,
                "winner": "ours" if stop_reason == "won" else "bar",
                "stop_reason": stop_reason,
                "gap": gap,
                "rounds": rounds,
            },
            "__spend__": {
                "llm_calls_consumed": spent["llm_calls"],
                "nodes_executed": spent["nodes"],
            },
        }

    return _runner
