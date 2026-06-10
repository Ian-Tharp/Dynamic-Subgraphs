"""Read + compare the scored runs corpus (Slice 7 PR6).

`EvalCorpus` turns `runs/*/eval.json` into comparable memory: load, group by
task (`prompt_sha256`) or shape (`topology_signature`), and run the paired
Pareto comparison the benchmark verdict uses (design spec §3/§7): quality
floor first, then Pareto dominance on (quality, cost) — USD when every run
priced, tokens otherwise. Refuses mismatched `rubric_id` /
`applicable_dimensions` pairs: incomparable pairs are excluded from any
verdict, never silently scored.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from dynamic_subgraphs.eval import EvalResult


@dataclass(frozen=True)
class OriginComparison:
    """The paired verdict for one task (grouped by prompt_sha256)."""

    task_key: str
    runs: tuple[EvalResult, ...]
    pareto_winner: str | None  # origin name; None = tie / nobody dominates
    incomparable: bool = False
    reason: str = ""


class EvalCorpus:
    """Reader over a runs directory's eval.json corpus."""

    def __init__(self, root_dir: Path | str = "runs") -> None:
        self._root = Path(root_dir)

    def load(self, run_id: str) -> EvalResult | None:
        path = self._root / run_id / "eval.json"
        if not path.exists():
            return None
        try:
            return EvalResult.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, ValidationError, OSError):
            return None

    def all(self) -> Iterator[EvalResult]:
        if not self._root.exists():
            return
        for child in sorted(self._root.iterdir()):
            if child.is_dir():
                result = self.load(child.name)
                if result is not None:
                    yield result

    def by_task(self) -> dict[str, list[EvalResult]]:
        groups: dict[str, list[EvalResult]] = {}
        for r in self.all():
            if r.fingerprint is not None:
                groups.setdefault(r.fingerprint.prompt_sha256, []).append(r)
        return groups

    def by_topology(self) -> dict[str, list[EvalResult]]:
        groups: dict[str, list[EvalResult]] = {}
        for r in self.all():
            if r.fingerprint is not None:
                groups.setdefault(r.fingerprint.topology_signature, []).append(r)
        return groups

    def compare_origins(self, task_key: str) -> OriginComparison:
        runs = tuple(self.by_task().get(task_key, []))
        if len(runs) < 2:
            return OriginComparison(task_key, runs, None, True, "needs >= 2 runs")
        if len({r.rubric_id for r in runs}) > 1:
            return OriginComparison(task_key, runs, None, True, "rubric_id mismatch")
        if len({tuple(sorted(r.applicable_dimensions)) for r in runs}) > 1:
            return OriginComparison(
                task_key, runs, None, True, "applicable_dimensions mismatch"
            )
        eligible = [r for r in runs if r.quality_floor_met]
        if not eligible:
            return OriginComparison(task_key, runs, None, False, "no run met floor")

        # USD is the cost axis only when EVERY eligible run is priced —
        # mixing axes across runs would make dominance incoherent.
        all_priced = all(r.cost_usd is not None for r in eligible)

        def cost_axis(r: EvalResult) -> float:
            if all_priced:
                assert r.cost_usd is not None
                return r.cost_usd
            return float(r.total_tokens)

        def dominates(a: EvalResult, b: EvalResult) -> bool:
            ge = a.quality >= b.quality and cost_axis(a) <= cost_axis(b)
            strict = a.quality > b.quality or cost_axis(a) < cost_axis(b)
            return ge and strict

        winners = [
            r for r in eligible if all(r is o or dominates(r, o) for o in eligible)
        ]
        # A winner must have a fingerprint (guaranteed by by_task grouping, but
        # we check anyway to satisfy the type checker — fingerprint is Optional).
        if len(winners) == 1 and winners[0].fingerprint is not None:
            winner: str | None = winners[0].fingerprint.origin
        else:
            winner = None
        return OriginComparison(task_key, runs, winner, False, "")
