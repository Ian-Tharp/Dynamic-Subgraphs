"""The frozen §4 decision rule, as code (internal).

Applies ``docs/evals/bench-dynamic-vs-fixed-2026-06.md`` §4 (registered
2026-06-11, frozen) to a main-run ``bench.jsonl``. Every choice the prose
leaves open is fixed below and was committed **before** any paid main-run
data existed, so the analysis can't be picked after seeing results. Changing
any constant or operationalization here after the paid run is a
re-registration and has to be recorded in the results doc.

Operationalization (one line per forking path):

* **Pairing.** Rows are keyed by ``ArmRunRow.arm``, not origin: A1 and A2 are
  both ``invented``. Each dynamic sub-arm is compared with the ``routed`` arm.
* **Per-task aggregate.** For each (task, arm): the mean over repeats of the
  headline score, blended quality, full cost, worker-only cost and tokens.
  Failed runs count as quality 0 (pilot rule); a missing headline on a row
  also counts as 0.
* **Floor.** An arm meets the floor on a task iff its mean blended quality is
  >= 0.6. A task is floor-eligible iff at least one arm meets it. If only one
  arm meets it, that arm wins the task (as ``corpus.compare_origins`` does:
  only eligible runs compete). If neither does, the task is excluded.
* **Pareto.** Both arms eligible: an arm wins iff it is >= on headline score
  and <= on cost, strictly better on one. Anything else is a tie (including
  trade-offs: better quality at higher cost).
* **Cost axis.** USD when every row of both arms on the task is priced, else
  total tokens for that task (flagged in the report).
* **Full cost** = ``cost_usd`` (planner included). **Worker-only cost** =
  ``worker_cost_usd`` when the planner split exists, else ``cost_usd`` (no
  deduction; the router has no planner, so its worker cost is its full cost).
  On the token axis the two bases are identical.
* **Win share** = wins / floor-eligible tasks (ties stay in the denominator,
  as §4 reads: "≥60% of floor-eligible tasks").
* **Sign test** = exact one-sided binomial over decisive tasks (ties
  excluded): p = P(X >= wins | n = wins_a + wins_b, 0.5).
* **Bar** = win share >= 0.60 AND p < 0.05.
* **Bootstrap CI** = 95% percentile interval of the win share, resampling
  floor-eligible tasks with replacement, 10,000 resamples,
  ``random.Random(20260611)`` (the registration date) fresh per interval.
* **Verdict**, in order: KEEP iff the dynamic arm clears the bar on full cost.
  KILL iff the router clears it on full AND worker-only cost.
  INCONCLUSIVE-leaning-template-cache iff the dynamic arm clears it on
  worker-only cost (but not full). Otherwise INCONCLUSIVE.
* **λ cut** (secondary, never decisive): per run
  ``adjusted = headline - 0.05 · total_tokens / 10,000``; per task the higher
  mean adjusted score wins (equal = tie), over the same floor-eligible tasks.
* **Strata** get the same computation but are descriptive only: 4 tasks per
  stratum give a best one-sided p of 1/16 = 0.0625, so no stratum can clear
  the bar.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from dynamic_subgraphs.bench import ArmRunRow

RULE = "docs/evals/bench-dynamic-vs-fixed-2026-06.md §4 (frozen 2026-06-11)"
QUALITY_FLOOR = 0.6
WIN_SHARE = 0.60
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260611
CI_LEVEL = 0.95
LAMBDA = 0.05
LAMBDA_TOKENS = 10_000
HEADLINES: tuple[str, ...] = ("structural", "goal")
ALL = "all"

KEEP = "KEEP"
KILL = "KILL"
INCONCLUSIVE = "INCONCLUSIVE"
LEANING = "INCONCLUSIVE-leaning-template-cache"
VERDICTS: tuple[str, ...] = (KEEP, KILL, INCONCLUSIVE, LEANING)

Outcome = Literal["dynamic", "baseline", "tie", "ineligible"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def sign_test_p(wins: int, n: int) -> float:
    """Exact one-sided sign-test p: P(X >= wins) for X ~ Binomial(n, 0.5).

    ``n`` is the number of decisive (non-tie) tasks. n == 0 gives 1.0.
    """
    if n < 0 or wins < 0 or wins > n:
        raise ValueError(f"need 0 <= wins <= n, got wins={wins}, n={n}")
    if n == 0:
        return 1.0
    tail: int = sum(math.comb(n, k) for k in range(wins, n + 1))
    return tail / (1 << n)


def min_wins_for_p(n: int, alpha: float = ALPHA) -> int | None:
    """Fewest wins out of ``n`` decisive tasks with p < ``alpha``; None if none."""
    for w in range(n + 1):
        if sign_test_p(w, n) < alpha:
            return w
    return None


def required_wins(eligible: int, ties: int) -> int | None:
    """Fewest wins that clear the full bar (share AND sign test).

    ``eligible`` floor-eligible tasks of which ``ties`` are ties, so
    ``eligible - ties`` are decisive. None if no win count can clear it.
    """
    decisive = eligible - ties
    by_p = min_wins_for_p(decisive)
    if by_p is None:
        return None
    by_share = math.ceil(WIN_SHARE * eligible - 1e-9)
    need = max(by_p, by_share)
    return need if need <= decisive else None


def bootstrap_ci(
    indicators: Sequence[int],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean of a 0/1 per-task indicator."""
    m = len(indicators)
    if m == 0:
        return None
    rng = random.Random(seed)
    pool = list(indicators)
    means = sorted(sum(rng.choices(pool, k=m)) / m for _ in range(resamples))
    tail = (1 - level) / 2
    lo = means[math.floor(tail * (resamples - 1))]
    hi = means[math.ceil((1 - tail) * (resamples - 1))]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Per-task aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmTask:
    """One arm's mean metrics on one task, across its repeats."""

    runs: int
    quality: float
    structural: float
    goal: float
    tokens: float
    full_usd: float | None  # None when any repeat is unpriced
    worker_usd: float | None
    missing_headline: int

    def headline(self, name: str) -> float:
        return self.structural if name == "structural" else self.goal

    @property
    def floor_met(self) -> bool:
        return self.quality >= QUALITY_FLOOR


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def aggregate(rows: Sequence[ArmRunRow]) -> ArmTask:
    """Aggregate one arm's repeats on one task."""
    if not rows:
        raise ValueError("no rows to aggregate")
    missing = sum(
        1
        for r in rows
        if r.status == "ok" and (r.structural_quality is None or r.goal_score is None)
    )

    def score(v: float | None, r: ArmRunRow) -> float:
        return float(v) if (v is not None and r.status == "ok") else 0.0

    priced = all(r.cost_usd is not None for r in rows)
    full = [float(r.cost_usd) for r in rows if r.cost_usd is not None]
    worker = [
        float(r.worker_cost_usd if r.worker_cost_usd is not None else r.cost_usd)
        for r in rows
        if r.cost_usd is not None
    ]
    return ArmTask(
        runs=len(rows),
        quality=_mean([r.quality if r.status == "ok" else 0.0 for r in rows]),
        structural=_mean([score(r.structural_quality, r) for r in rows]),
        goal=_mean([score(r.goal_score, r) for r in rows]),
        tokens=_mean([float(r.total_tokens) for r in rows]),
        full_usd=_mean(full) if priced else None,
        worker_usd=_mean(worker) if priced else None,
        missing_headline=missing,
    )


def adjusted(rows: Sequence[ArmRunRow], headline: str) -> float:
    """Mean λ-penalized headline score: score - 0.05 · tokens / 10k."""
    vals = []
    for r in rows:
        raw = r.structural_quality if headline == "structural" else r.goal_score
        s = float(raw) if (raw is not None and r.status == "ok") else 0.0
        vals.append(s - LAMBDA * (r.total_tokens / LAMBDA_TOKENS))
    return _mean(vals)


def pareto(qa: float, ca: float, qb: float, cb: float) -> Outcome:
    """Paired Pareto on (quality up, cost down). 'dynamic' means a wins."""
    if qa >= qb and ca <= cb and (qa > qb or ca < cb):
        return "dynamic"
    if qb >= qa and cb <= ca and (qb > qa or cb < ca):
        return "baseline"
    return "tie"


@dataclass(frozen=True)
class TaskComparison:
    """One dynamic arm vs the baseline on one task, for one headline."""

    task_id: str
    task_type: str
    headline: str
    cost_axis: Literal["usd", "tokens"]
    dynamic_floor_met: bool
    baseline_floor_met: bool
    full: Outcome
    worker: Outcome
    lambda_cut: Outcome


def compare_task(
    task_id: str,
    task_type: str,
    headline: str,
    dyn_rows: Sequence[ArmRunRow],
    base_rows: Sequence[ArmRunRow],
) -> TaskComparison:
    """Apply floor -> Pareto (both cost bases) -> λ cut for one task/headline."""
    d, b = aggregate(dyn_rows), aggregate(base_rows)
    usd = d.full_usd is not None and b.full_usd is not None
    axis: Literal["usd", "tokens"] = "usd" if usd else "tokens"

    def outcome(cost_d: float, cost_b: float) -> Outcome:
        if not d.floor_met and not b.floor_met:
            return "ineligible"
        if d.floor_met != b.floor_met:
            return "dynamic" if d.floor_met else "baseline"
        return pareto(d.headline(headline), cost_d, b.headline(headline), cost_b)

    if usd:
        assert d.full_usd is not None and b.full_usd is not None
        assert d.worker_usd is not None and b.worker_usd is not None
        full = outcome(d.full_usd, b.full_usd)
        worker = outcome(d.worker_usd, b.worker_usd)
    else:
        full = worker = outcome(d.tokens, b.tokens)

    if full == "ineligible":
        lam: Outcome = "ineligible"
    else:
        ad, ab = adjusted(dyn_rows, headline), adjusted(base_rows, headline)
        lam = "dynamic" if ad > ab else "baseline" if ab > ad else "tie"
    return TaskComparison(
        task_id=task_id,
        task_type=task_type,
        headline=headline,
        cost_axis=axis,
        dynamic_floor_met=d.floor_met,
        baseline_floor_met=b.floor_met,
        full=full,
        worker=worker,
        lambda_cut=lam,
    )


# ---------------------------------------------------------------------------
# Verdict lines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideStats:
    """One side's evidence on one cost basis (or the λ cut)."""

    wins: int
    losses: int
    ties: int
    eligible: int  # wins + losses + ties
    decisive: int  # wins + losses: the sign test's n
    share: float | None  # wins / eligible
    p: float  # exact one-sided sign test
    ci: tuple[float, float] | None
    meets_bar: bool


def side_stats(
    outcomes: Sequence[Outcome], side: Outcome, with_ci: bool = True
) -> SideStats:
    other: Outcome = "baseline" if side == "dynamic" else "dynamic"
    elig = [o for o in outcomes if o != "ineligible"]
    wins = sum(1 for o in elig if o == side)
    losses = sum(1 for o in elig if o == other)
    ties = len(elig) - wins - losses
    p = sign_test_p(wins, wins + losses)
    share = wins / len(elig) if elig else None
    ci = bootstrap_ci([1 if o == side else 0 for o in elig]) if with_ci else None
    meets = share is not None and share >= WIN_SHARE and p < ALPHA
    return SideStats(
        wins=wins,
        losses=losses,
        ties=ties,
        eligible=len(elig),
        decisive=wins + losses,
        share=share,
        p=p,
        ci=ci,
        meets_bar=meets,
    )


def decide(dyn_full: bool, dyn_worker: bool, base_full: bool, base_worker: bool) -> str:
    """§4 verdict from the four bar checks (see module docstring)."""
    if dyn_full:
        return KEEP
    if base_full and base_worker:
        return KILL
    if dyn_worker:
        return LEANING
    return INCONCLUSIVE


@dataclass(frozen=True)
class VerdictLine:
    """One §4 verdict line: dynamic sub-arm vs baseline, headline, scope."""

    dynamic_arm: str
    baseline_arm: str
    headline: str
    scope: str  # "all" or a task_type (stratum)
    decisive_scope: bool  # False for strata (descriptive only)
    tasks: int
    ineligible: int
    token_axis_tasks: int
    full: SideStats  # dynamic arm, full cost
    worker: SideStats  # dynamic arm, worker-only cost
    baseline_full: SideStats  # router, full cost
    baseline_worker: SideStats  # router, worker-only cost
    lambda_cut: SideStats  # dynamic arm, λ-penalized (never decisive)
    verdict: str


def verdict_line(
    dynamic_arm: str,
    baseline_arm: str,
    headline: str,
    scope: str,
    comps: Sequence[TaskComparison],
) -> VerdictLine:
    full = [c.full for c in comps]
    worker = [c.worker for c in comps]
    d_full = side_stats(full, "dynamic")
    d_worker = side_stats(worker, "dynamic")
    b_full = side_stats(full, "baseline")
    b_worker = side_stats(worker, "baseline")
    lam = side_stats([c.lambda_cut for c in comps], "dynamic", with_ci=False)
    return VerdictLine(
        dynamic_arm=dynamic_arm,
        baseline_arm=baseline_arm,
        headline=headline,
        scope=scope,
        decisive_scope=scope == ALL,
        tasks=len(comps),
        ineligible=sum(1 for c in comps if c.full == "ineligible"),
        token_axis_tasks=sum(1 for c in comps if c.cost_axis == "tokens"),
        full=d_full,
        worker=d_worker,
        baseline_full=b_full,
        baseline_worker=b_worker,
        lambda_cut=lam,
        verdict=decide(
            d_full.meets_bar, d_worker.meets_bar, b_full.meets_bar, b_worker.meets_bar
        ),
    )


# ---------------------------------------------------------------------------
# Whole-report analysis
# ---------------------------------------------------------------------------


def load_rows(path: Path | str) -> list[ArmRunRow]:
    """Read ``bench.jsonl`` into rows (unknown or missing fields raise)."""
    rows: list[ArmRunRow] = []
    for ln, line in enumerate(Path(path).read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(ArmRunRow(**json.loads(line)))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{ln}: not an ArmRunRow ({exc})") from exc
    return rows


def analyze(
    rows: Iterable[ArmRunRow],
    *,
    dynamic_arms: Sequence[str],
    baseline_arm: str = "routed",
    headline_arm: str | None = None,
    expected_repeats: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Apply §4 to ``rows``. Returns a JSON-safe report dict.

    Raises ``ValueError`` on evidence that can't be scored honestly: rows from
    more than one bench_id, duplicate (task, arm, repeat) keys, or a named arm
    with no rows. Tasks missing one side of a pair are excluded and listed in
    ``warnings`` (a partial run is not decision-grade).
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no rows")
    bench_ids = sorted({r.bench_id for r in rows})
    if len(bench_ids) > 1:
        raise ValueError(f"rows span several bench_ids: {bench_ids}")
    seen: set[tuple[str, str, int]] = set()
    for r in rows:
        key = (r.task_id, r.arm, r.repeat)
        if key in seen:
            raise ValueError(f"duplicate row {key}")
        seen.add(key)
    arms_present = {r.arm for r in rows}
    for a in (*dynamic_arms, baseline_arm):
        if a not in arms_present:
            raise ValueError(f"arm {a!r} has no rows (present: {sorted(arms_present)})")

    by: dict[tuple[str, str], list[ArmRunRow]] = {}
    task_type: dict[str, str] = {}
    for r in rows:
        by.setdefault((r.task_id, r.arm), []).append(r)
        task_type[r.task_id] = r.task_type
    tasks = sorted(task_type)
    strata = sorted(set(task_type.values()))

    warnings: list[str] = []
    registered_sizing: bool | None = None
    if expected_repeats is not None:
        registered_sizing = True
        for (tid, arm), rs in sorted(by.items()):
            want = expected_repeats.get(task_type[tid])
            if want is not None and len(rs) != want:
                registered_sizing = False
                warnings.append(
                    f"{tid}/{arm}: {len(rs)} repeats, registered {want} "
                    "(not the registered sizing; never report a verdict from it)"
                )
    if all(r.total_tokens == 0 for r in rows):
        warnings.append(
            "no row spent any tokens: mock (--dry-run) data, the verdicts "
            "only show the analyzer runs and mean nothing"
        )
    missing = sum(1 for r in rows if r.status == "ok" and r.goal_score is None)
    if missing:
        warnings.append(f"{missing} ok rows lack goal_score (scored as 0)")

    lines: list[VerdictLine] = []
    per_task: list[dict[str, Any]] = []
    for dyn in dynamic_arms:
        for h in HEADLINES:
            comps: list[TaskComparison] = []
            for tid in tasks:
                d_rows, b_rows = by.get((tid, dyn)), by.get((tid, baseline_arm))
                if not d_rows or not b_rows:
                    warnings.append(f"{tid}: unpaired for {dyn} vs {baseline_arm}")
                    continue
                if len(d_rows) != len(b_rows):
                    warnings.append(
                        f"{tid}: {dyn} has {len(d_rows)} repeats, "
                        f"{baseline_arm} has {len(b_rows)} (paired design broken)"
                    )
                comps.append(compare_task(tid, task_type[tid], h, d_rows, b_rows))
            lines.append(verdict_line(dyn, baseline_arm, h, ALL, comps))
            for s in strata:
                sc = [c for c in comps if c.task_type == s]
                lines.append(verdict_line(dyn, baseline_arm, h, s, sc))
            per_task.extend({"dynamic_arm": dyn, **asdict(c)} for c in comps)
    token_tasks = sum(1 for c in per_task if c["cost_axis"] == "tokens")
    if token_tasks:
        warnings.append(
            f"{token_tasks} task comparisons fell back to the token cost axis "
            "(unpriced rows): worker-only cost = full cost there"
        )

    return {
        "rule": RULE,
        "constants": {
            "quality_floor": QUALITY_FLOOR,
            "win_share": WIN_SHARE,
            "alpha": ALPHA,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "lambda": LAMBDA,
            "lambda_tokens": LAMBDA_TOKENS,
        },
        "bench_id": bench_ids[0],
        "rows": len(rows),
        "tasks": len(tasks),
        "strata": strata,
        "dynamic_arms": list(dynamic_arms),
        "baseline_arm": baseline_arm,
        "headline_arm": headline_arm,
        "registered_sizing": registered_sizing,
        "warnings": list(dict.fromkeys(warnings)),
        "lines": [asdict(line) for line in lines],
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _ci(ci: Any) -> str:
    return "-" if ci is None else f"[{100 * ci[0]:.0f}%, {100 * ci[1]:.0f}%]"


def _side(s: Mapping[str, Any]) -> str:
    return f"{s['wins']}/{s['decisive']} (p={s['p']:.4f})"


def to_markdown(report: Mapping[str, Any]) -> str:
    """Render the verdict lines as a markdown table (one row per line)."""
    out = [
        f"# §4 verdict — `{report['bench_id']}`",
        "",
        f"Rule: {report['rule']}. Rows: {report['rows']}, tasks: {report['tasks']}. "
        f"Headline arm: `{report['headline_arm']}` vs `{report['baseline_arm']}`. "
        f"Registered sizing: {report['registered_sizing']}.",
        "",
        "Wins/ties/n are floor-eligible tasks; p is the exact one-sided sign "
        "test over decisive tasks (ties excluded); CI is the 95% bootstrap "
        "interval of the win share. Strata are descriptive only (best "
        "possible p with 4 tasks is 0.0625).",
        "",
        "| arm | headline | scope | n | wins | ties | losses | share | p | 95% CI "
        "| worker-only wins/n | router full wins/n | router worker wins/n "
        "| λ cut wins/n | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ln in report["lines"]:
        f = ln["full"]
        verdict = ln["verdict"] if ln["decisive_scope"] else f"({ln['verdict']})"
        if ln["scope"] == ALL and ln["dynamic_arm"] == report["headline_arm"]:
            verdict = f"**{verdict}**"
        out.append(
            f"| {ln['dynamic_arm']} | {ln['headline']} | {ln['scope']} "
            f"| {f['eligible']} | {f['wins']} | {f['ties']} | {f['losses']} "
            f"| {_pct(f['share'])} | {f['p']:.4f} | {_ci(f['ci'])} "
            f"| {_side(ln['worker'])} | {_side(ln['baseline_full'])} "
            f"| {_side(ln['baseline_worker'])} | {_side(ln['lambda_cut'])} "
            f"| {verdict} |"
        )
    if report["warnings"]:
        out += ["", "**Warnings**", ""]
        out += [f"- {w}" for w in report["warnings"]]
    return "\n".join(out) + "\n"


def analyze_file(
    path: Path | str,
    *,
    dynamic_arms: Sequence[str],
    baseline_arm: str = "routed",
    headline_arm: str | None = None,
    expected_repeats: Mapping[str, int] | None = None,
    out_dir: Path | str | None = None,
) -> tuple[dict[str, Any], str]:
    """Analyze a ``bench.jsonl``; write ``verdict.json`` + ``verdict.md``.

    Outputs go next to the input unless ``out_dir`` is given.
    """
    report = analyze(
        load_rows(path),
        dynamic_arms=dynamic_arms,
        baseline_arm=baseline_arm,
        headline_arm=headline_arm,
        expected_repeats=expected_repeats,
    )
    md = to_markdown(report)
    dest = Path(out_dir) if out_dir is not None else Path(path).parent
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "verdict.json").write_text(json.dumps(report, indent=2), "utf-8")
    (dest / "verdict.md").write_text(md, "utf-8")
    return report, md
