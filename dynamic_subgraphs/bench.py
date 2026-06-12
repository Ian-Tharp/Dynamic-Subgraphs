"""The 3-arm dynamic-vs-authored benchmark harness (Slice 7 PR8).

Consumes the eval layer (never re-implements scoring). Arms (design spec §7):
  - "invented" (Arm A): the live LLM planner plans per task.
  - "routed"   (Arm B, PRIMARY baseline): RouterLibrary picks a reviewed,
    frozen GraphSpec per task_type; runs via RunConfig.fixed_spec on the
    REAL worker path after {prompt} substitution (fill_spec).
  - "authored" (Arm C, sanity floor): one frozen linear graph for every task.

Persists runs/bench/<bench_id>/bench.jsonl (one ArmRunRow per line) and
report.json (row count + pilot CV table). The KEEP/KILL decision rule is
deliberately NOT encoded here — it is pre-registered after the pilot
(spec §0/§7) and applied in the results doc, not by code that could be
quietly edited post-hoc.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.errors import DynamicSubgraphsError
from app.models.graph_spec import GraphSpec
from dynamic_subgraphs.engine import DynamicSubgraphs, EngineConfig, Model
from dynamic_subgraphs.eval.types import EvalReference, Origin
from dynamic_subgraphs.recording import Recording
from dynamic_subgraphs.types import Planner

ARMS: tuple[str, ...] = ("invented", "routed", "authored")


class BenchIntegrityError(DynamicSubgraphsError, RuntimeError):
    """Raised when a benchmark run would record corrupt evidence.

    Two cases abort-fast rather than pollute the verdict: (1) a run that
    succeeded (status=="ok") but produced no EvalResult — an infrastructure
    failure in the gate that would otherwise be silently booked as a
    quality-0 row, biasing the decision toward KILL; (2) a reused bench_id
    whose directory already holds rows from a prior attempt, which would
    mislead corpus readers. Abort-fast beats polluted verdicts.
    """


# ---------------------------------------------------------------------------
# BenchTask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchTask:
    """One benchmark task: a prompt + task type + gold reference."""

    task_id: str
    task_type: str
    prompt: str
    reference: EvalReference

    @classmethod
    def from_jsonl(cls, path: Path) -> list[BenchTask]:
        """Load tasks from a JSONL file — one JSON object per line."""
        tasks: list[BenchTask] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(
                cls(
                    task_id=obj["task_id"],
                    task_type=obj["task_type"],
                    prompt=obj["prompt"],
                    reference=EvalReference.model_validate(obj["reference"]),
                )
            )
        return tasks


# ---------------------------------------------------------------------------
# fill_spec
# ---------------------------------------------------------------------------


def _fill_value(value: Any, prompt: str) -> Any:
    """Recursively replace the literal substring ``{prompt}`` in every string."""
    if isinstance(value, str):
        return value.replace("{prompt}", prompt)
    if isinstance(value, dict):
        return {k: _fill_value(v, prompt) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_value(item, prompt) for item in value]
    return value


def fill_spec(spec: GraphSpec, prompt: str) -> GraphSpec:
    """Return a deep-copied ``GraphSpec`` with ``{prompt}`` substituted.

    Uses ``str.replace`` — never ``str.format`` — so user prompts containing
    braces, quotes, or backslashes are safe. Substitution walks every node's
    ``params`` dict recursively (instructions, tool args, any future nested
    child_params). The original ``spec`` object is never mutated.
    """
    copy = spec.model_copy(deep=True)
    for node in copy.nodes:
        node.params = _fill_value(node.params, prompt)
    return copy


# ---------------------------------------------------------------------------
# RouterLibrary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterLibrary:
    """Deterministic router: task_type -> frozen GraphSpec.

    ``by_task_type`` maps task-type strings to the reviewed, author-frozen spec
    for that type. ``single_fixed`` is the Arm C sanity-floor spec used for
    every task regardless of type.
    """

    by_task_type: dict[str, GraphSpec]
    single_fixed: GraphSpec

    @classmethod
    def from_dir(cls, library_dir: Path, single_fixed_path: Path) -> RouterLibrary:
        """Load all ``*.json`` specs from ``library_dir`` (keyed by stem) and the
        single fixed-floor spec from ``single_fixed_path``.

        The stem of each file is used as the task_type key (e.g.
        ``compare.json`` -> ``"compare"``).
        """
        by_task_type: dict[str, GraphSpec] = {}
        for json_path in sorted(library_dir.glob("*.json")):
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            by_task_type[json_path.stem] = GraphSpec.model_validate(raw)
        single_fixed = GraphSpec.model_validate(
            json.loads(single_fixed_path.read_text(encoding="utf-8"))
        )
        return cls(by_task_type=by_task_type, single_fixed=single_fixed)

    def route(self, task: BenchTask) -> GraphSpec:
        """Deterministic router: task_type lookup; ``'summarize'`` is the fallback."""
        return self.by_task_type.get(task.task_type, self.by_task_type["summarize"])


# ---------------------------------------------------------------------------
# ArmRunRow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRunRow:
    """One benchmark data row: task x arm x repeat -> eval metrics."""

    bench_id: str
    task_id: str
    task_type: str
    arm: str
    repeat: int
    run_id: str
    status: str
    quality: float
    quality_floor_met: bool
    value_per_ktok: float | None
    cost_usd: float | None
    planner_cost_usd: float | None
    worker_cost_usd: float | None
    total_tokens: int
    node_count: int | None
    topology_signature: str | None
    applicable_dimensions: list[str] = field(default_factory=list)
    structural_quality: float | None = None
    goal_score: float | None = None


# ---------------------------------------------------------------------------
# BenchReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchReport:
    """The full benchmark output: all rows + pilot CV stats."""

    bench_id: str
    rows: tuple[ArmRunRow, ...]
    pilot: dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _origin(arm: str) -> Origin:
    """Map arm name to the eval-layer Origin literal."""
    if arm == "invented":
        return "invented"
    if arm == "routed":
        return "routed"
    return "authored"


def _headlines(
    eval_result: Any,
) -> tuple[float | None, float | None]:
    """Split two pre-registered headline metrics from the EvalResult components.

    Returns:
        structural_quality: weight-blended score over applicable non-goal_completion
            components (using ``eval_result.weights``).
        goal_score: the ``goal_completion`` component's score iff its method is
            ``"reference"`` or ``"llm_judge"``; else None.
    """
    components: list[Any] = list(eval_result.components)
    weights: dict[str, float] = dict(eval_result.weights)

    # Structural quality: applicable non-goal_completion components, blended.
    structural_num = 0.0
    structural_den = 0.0
    goal_score: float | None = None

    for comp in components:
        if comp.dimension == "goal_completion":
            if comp.method in ("reference", "llm_judge"):
                goal_score = float(comp.score)
        else:
            if comp.applicable:
                w = weights.get(comp.dimension, 0.0)
                structural_num += w * float(comp.score)
                structural_den += w

    structural_quality: float | None = None
    if structural_den > 0:
        structural_quality = structural_num / structural_den

    return structural_quality, goal_score


def _pilot_stats(rows: list[ArmRunRow]) -> dict[str, Any]:
    """Per-task CV of ``value_per_ktok`` in the INVENTED arm.

    For each task, computes CV = stdev / mean over value_per_ktok values from
    invented-arm rows, then derives the recommended repeats for the main run
    using the normal approximation (95%, pre-registered 20% relative effect):
        repeats = max(3, ceil((1.96 * cv / 0.20) ** 2) + 1)

    Every per-task entry carries ``samples`` (usable value_per_ktok count) and
    ``dropped_none`` (invented-arm rows whose value_per_ktok was None), so a
    thin CV is visible at a glance. ``cv`` is None when there are fewer than 2
    usable values or mean <= 0.
    """
    # Group invented-arm value_per_ktok by task_id, tracking dropped Nones.
    by_task: dict[str, list[float]] = {}
    dropped: dict[str, int] = {}
    for row in rows:
        if row.arm == "invented":
            by_task.setdefault(row.task_id, [])
            dropped.setdefault(row.task_id, 0)
            if row.value_per_ktok is not None:
                by_task[row.task_id].append(row.value_per_ktok)
            else:
                dropped[row.task_id] += 1

    per_task_cv: dict[str, Any] = {}
    for task_id, values in by_task.items():
        entry: dict[str, Any] = {
            "cv": None,
            "samples": len(values),
            "dropped_none": dropped[task_id],
        }
        if len(values) >= 2:
            mean = statistics.mean(values)
            if mean > 0:
                stdev = statistics.stdev(values)
                cv = stdev / mean
                entry["cv"] = cv
                entry["mean_value_per_ktok"] = mean
                entry["stdev_value_per_ktok"] = stdev
                entry["recommended_repeats"] = max(
                    3, math.ceil((1.96 * cv / 0.20) ** 2) + 1
                )
        per_task_cv[task_id] = entry

    return {
        "per_task_cv": per_task_cv,
        "note": "pre-register the final decision rule before the main run",
    }


def _persist(report: BenchReport, runs_dir: Path) -> None:
    """Write ``bench.jsonl`` and ``report.json`` under ``runs_dir/bench/<bench_id>/``."""
    bench_dir = runs_dir / "bench" / report.bench_id
    bench_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = bench_dir / "bench.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in report.rows:
            fh.write(json.dumps(asdict(row)) + "\n")

    report_payload: dict[str, Any] = {
        "bench_id": report.bench_id,
        "row_count": len(report.rows),
        "pilot": report.pilot,
    }
    report_path = bench_dir / "report.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    tasks: list[BenchTask],
    repeats: int,
    runs_dir: Path | str,
    router_library: RouterLibrary,
    bench_id: str,
    planner: Planner = "llm",
    model: Model | None = None,
    planner_model: Model | None = None,
) -> BenchReport:
    """Run all tasks across all three arms for ``repeats`` repetitions.

    One pinned worker model is shared across ALL arms (spec §7). Arms:
    - ``"invented"``: live LLM planner plans per task (fixed_spec=None).
    - ``"routed"``: RouterLibrary selects a frozen spec; substituted via
      fill_spec; runs on the REAL worker path.
    - ``"authored"``: the single fixed-floor spec; substituted via fill_spec;
      runs on the REAL worker path.

    Runs are sequential (no threading). Plan/validation failures synthesize
    quality=0.0 / floor=False rows (the plan-failure rule).

    Raises:
        BenchIntegrityError: if ``bench_id`` reuses a non-empty bench dir, or
            if a run succeeds (status=="ok") without an EvalResult — an
            infrastructure failure that must not be booked as quality-0 data.
    """
    from dynamic_subgraphs.eval import DeterministicEvalGate

    runs_dir = Path(runs_dir)

    # Refuse a reused bench_id BEFORE spending any tokens: stale rows from a
    # prior attempt under the same id would mislead corpus readers.
    bench_dir = runs_dir / "bench" / bench_id
    if bench_dir.is_dir() and any(bench_dir.iterdir()):
        raise BenchIntegrityError(
            f"bench dir {bench_dir} already exists and is non-empty — "
            "bench_id must be unique per run; stale run dirs from a prior "
            "attempt would mislead corpus readers."
        )

    gate = DeterministicEvalGate(grounding_applicability="reference_only")
    engine = DynamicSubgraphs(
        EngineConfig(
            planner=planner,
            model=model,
            planner_model=planner_model,
            runs_dir=runs_dir,
            eval_gate=gate,
        )
    )

    rows: list[ArmRunRow] = []

    for task in tasks:
        for arm in ARMS:
            for repeat in range(repeats):
                run_id = f"{bench_id}-{task.task_id}-{arm}-r{repeat}"

                # Determine fixed_spec for routed/authored arms.
                fixed: GraphSpec | None
                if arm == "routed":
                    fixed = fill_spec(router_library.route(task), task.prompt)
                elif arm == "authored":
                    fixed = fill_spec(router_library.single_fixed, task.prompt)
                else:
                    fixed = None

                result = engine.run(
                    task.prompt,
                    run_id=run_id,
                    record=Recording.evaluated(),
                    fixed_spec=fixed,
                    task_id=task.task_id,
                    origin=_origin(arm),
                    reference=task.reference,
                )

                eval_result = result.eval
                if eval_result is None and result.status == "ok":
                    # Disjoint from the legit case below: the run succeeded
                    # (and spent money) but the gate threw and the engine
                    # swallowed+logged it. Booking this as quality-0 would
                    # silently bias the verdict toward KILL.
                    raise BenchIntegrityError(
                        f"run {run_id!r} succeeded but produced no EvalResult — "
                        "the gate failed on a successful run (see engine warning "
                        "logs). Aborting: a decision-grade benchmark must not "
                        "book infrastructure failures as quality-0 data."
                    )
                if eval_result is None:
                    # Legit failure (plan/validation/compile/execution — any
                    # non-ok status): synthesize a zero row.
                    quality = 0.0
                    floor_met = False
                    vpktok: float | None = None
                    cost_usd: float | None = result.cost
                    planner_cost: float | None = None
                    worker_cost: float | None = None
                    node_count: int | None = (
                        len(result.plan.nodes) if result.plan is not None else None
                    )
                    topo_sig: str | None = None
                    applicable_dims: list[str] = []
                    structural_q: float | None = None
                    goal_s: float | None = None
                else:
                    quality = float(eval_result.quality)
                    floor_met = bool(eval_result.quality_floor_met)
                    vpktok = eval_result.value_per_ktok
                    cost_usd = eval_result.cost_usd
                    planner_cost = eval_result.planner_cost_usd
                    worker_cost = eval_result.worker_cost_usd
                    node_count = (
                        eval_result.fingerprint.node_count
                        if eval_result.fingerprint is not None
                        else None
                    )
                    topo_sig = (
                        eval_result.fingerprint.topology_signature
                        if eval_result.fingerprint is not None
                        else None
                    )
                    applicable_dims = list(eval_result.applicable_dimensions)
                    structural_q, goal_s = _headlines(eval_result)

                row = ArmRunRow(
                    bench_id=bench_id,
                    task_id=task.task_id,
                    task_type=task.task_type,
                    arm=arm,
                    repeat=repeat,
                    run_id=run_id,
                    status=str(result.status),
                    quality=quality,
                    quality_floor_met=floor_met,
                    value_per_ktok=vpktok,
                    cost_usd=cost_usd,
                    planner_cost_usd=planner_cost,
                    worker_cost_usd=worker_cost,
                    total_tokens=result.usage.total_tokens,
                    node_count=node_count,
                    topology_signature=topo_sig,
                    applicable_dimensions=applicable_dims,
                    structural_quality=structural_q,
                    goal_score=goal_s,
                )
                rows.append(row)

    pilot = _pilot_stats(rows)
    report = BenchReport(
        bench_id=bench_id,
        rows=tuple(rows),
        pilot=pilot,
    )
    _persist(report, runs_dir)
    return report
