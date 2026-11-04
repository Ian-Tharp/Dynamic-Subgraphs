"""One-command runner for the main dynamic-vs-router benchmark (internal).

Reads a committed run config (``docs/evals/bench-main-v1.config.json``) that
pins the pack, arms, planner/worker models and per-stratum repeats, then
calls :func:`dynamic_subgraphs.bench.run_benchmark`. Four modes:

    # Print the run plan + cost estimate. No engine, no network.
    python -m dynamic_subgraphs._bench_main --config <cfg> --estimate

    # Full schedule offline: mock planner + mock runners, zero tokens.
    python -m dynamic_subgraphs._bench_main --config <cfg> --dry-run

    # The paid live run. Requires --live AND the provider keys. Stops before
    # spending past spend_cap_usd (or --max-spend-usd), or when a STOP file
    # appears in runs/bench/<bench_id>/.
    python -m dynamic_subgraphs._bench_main --config <cfg> --live

    # Apply the frozen §4 rule to a finished run's rows.
    python -m dynamic_subgraphs._bench_main --config <cfg> --analyze <bench.jsonl>

Internal tooling, not public API (underscore module, see
docs/api-stability.md). The §4 rule lives in ``_bench_verdict``, committed
before the paid run so the analysis can't be chosen after seeing the data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dynamic_subgraphs._bench_verdict import analyze_file
from dynamic_subgraphs.bench import (
    BenchArm,
    BenchReport,
    BenchSpendAbort,
    BenchTask,
    RouterLibrary,
    SpendGuard,
    resolve_repeats,
    run_benchmark,
)
from dynamic_subgraphs.engine import Model
from dynamic_subgraphs.eval.types import Origin

# Keys a live run needs: worker/planner (OpenAI) and grounded search (Tavily;
# without it the runtime silently falls back to a weaker search backend,
# which would change Arm B's research evidence vs the pilot).
LIVE_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY", "TAVILY_API_KEY")

_ORIGINS: tuple[str, ...] = ("invented", "routed", "authored")


@dataclass(frozen=True)
class MainRunConfig:
    """A parsed, validated main-run config."""

    bench_id: str
    pack: Path
    router_library: Path
    single_fixed: Path
    worker_model: Model
    arms: tuple[BenchArm, ...]
    repeats_by_task_type: dict[str, int]
    repeats_cap: int
    usd_per_run: dict[str, float]
    safety_margin: float
    seconds_per_run: float
    spend_cap_usd: float | None = None
    headline_arm: str | None = None

    @classmethod
    def load(cls, path: Path | str, root: Path | str = ".") -> MainRunConfig:
        """Load and validate a config; relative paths resolve against ``root``."""
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        base = Path(root)
        arms: list[BenchArm] = []
        for a in raw["arms"]:
            origin = a["origin"]
            if origin not in _ORIGINS:
                raise ValueError(f"arm {a['name']!r}: unknown origin {origin!r}")
            planner = a.get("planner_model")
            if origin == "invented" and planner is None:
                raise ValueError(f"invented arm {a['name']!r} must pin planner_model")
            arms.append(
                BenchArm(
                    name=a["name"],
                    origin=cast(Origin, origin),
                    planner_model=_model(planner) if planner else None,
                )
            )
        cap = int(raw["repeats_cap"])
        repeats = {k: int(v) for k, v in raw["repeats_by_task_type"].items()}
        over = {k: v for k, v in repeats.items() if v > cap or v < 1}
        if over:
            raise ValueError(f"repeats outside [1, cap={cap}]: {over}")
        est = raw["cost_estimate"]
        usd = {k: float(v) for k, v in est["usd_per_run"].items()}
        missing = {a.name for a in arms} - set(usd)
        if missing:
            raise ValueError(f"cost_estimate.usd_per_run missing arms: {missing}")
        spend_cap = raw.get("spend_cap_usd")
        if spend_cap is not None and not float(spend_cap) > 0:
            raise ValueError(f"spend_cap_usd must be > 0, got {spend_cap}")
        headline = raw.get("headline_arm")
        invented = {a.name for a in arms if a.origin == "invented"}
        if headline is not None and headline not in invented:
            raise ValueError(f"headline_arm {headline!r} is not an invented arm")
        return cls(
            bench_id=raw["bench_id"],
            pack=base / raw["pack"],
            router_library=base / raw["router_library"],
            single_fixed=base / raw["single_fixed"],
            worker_model=_model(raw["worker_model"]),
            arms=tuple(arms),
            repeats_by_task_type=repeats,
            repeats_cap=cap,
            usd_per_run=usd,
            safety_margin=float(est["safety_margin"]),
            seconds_per_run=float(est["seconds_per_run"]),
            spend_cap_usd=float(spend_cap) if spend_cap is not None else None,
            headline_arm=headline,
        )

    @property
    def dynamic_arms(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.arms if a.origin == "invented")

    @property
    def baseline_arm(self) -> str:
        routed = [a.name for a in self.arms if a.origin == "routed"]
        if len(routed) != 1:
            raise ValueError(f"need exactly one routed arm, got {routed}")
        return routed[0]


def _model(obj: Mapping[str, Any]) -> Model:
    return Model(obj["provider"], obj["model"])


@dataclass(frozen=True)
class RunPlan:
    """What a run will do, computed without running anything."""

    tasks: int
    arms: int
    runs: int
    runs_by_task_type: dict[str, int]
    estimated_usd: float
    estimated_usd_with_margin: float
    estimated_hours: float


def plan_run(
    cfg: MainRunConfig, tasks: Sequence[BenchTask], max_repeats: int | None = None
) -> RunPlan:
    """Count runs and estimate cost/time for ``tasks`` under ``cfg``."""
    repeats = _effective_repeats(cfg, max_repeats)
    per_type: dict[str, int] = {}
    usd = 0.0
    for t in tasks:
        n = resolve_repeats(t, repeats)
        per_type[t.task_type] = per_type.get(t.task_type, 0) + n * len(cfg.arms)
        usd += n * sum(cfg.usd_per_run[a.name] for a in cfg.arms)
    runs = sum(per_type.values())
    return RunPlan(
        tasks=len(tasks),
        arms=len(cfg.arms),
        runs=runs,
        runs_by_task_type=per_type,
        estimated_usd=usd,
        estimated_usd_with_margin=usd * cfg.safety_margin,
        estimated_hours=runs * cfg.seconds_per_run / 3600,
    )


def _effective_repeats(cfg: MainRunConfig, max_repeats: int | None) -> dict[str, int]:
    if max_repeats is None:
        return dict(cfg.repeats_by_task_type)
    return {k: min(v, max_repeats) for k, v in cfg.repeats_by_task_type.items()}


def run_main(
    cfg: MainRunConfig,
    *,
    mode: str,
    runs_dir: Path | str,
    bench_id: str | None = None,
    max_repeats: int | None = None,
    env: Mapping[str, str] | None = None,
    max_spend_usd: float | None = None,
) -> BenchReport:
    """Execute the configured run. ``mode`` is ``"dry-run"`` or ``"live"``.

    Dry-run uses the mock planner and mock runners (zero tokens) on the exact
    same pack, arms and repeat schedule as the live run.

    Both modes run under a :class:`SpendGuard` (the kill-switch):
    ``max_spend_usd`` overrides the config's ``spend_cap_usd``. A live run
    refuses to start without a cap, and aborts if a successful run comes back
    unpriced. Creating ``<runs_dir>/bench/<bench_id>/STOP`` stops the run
    before its next task run.
    """
    if mode not in ("dry-run", "live"):
        raise ValueError(f"unknown mode {mode!r}")
    cap = max_spend_usd if max_spend_usd is not None else cfg.spend_cap_usd
    if mode == "live" and cap is None:
        raise SystemExit(
            "live run refused: no spend cap (set spend_cap_usd in the config "
            "or pass --max-spend-usd). Nothing was spent."
        )
    if mode == "live":
        env = os.environ if env is None else env
        missing = [k for k in LIVE_ENV_VARS if not env.get(k)]
        if missing:
            raise SystemExit(
                f"live run refused: missing env vars {missing}. Nothing was spent."
            )
    tasks = BenchTask.from_jsonl(cfg.pack)
    library = RouterLibrary.from_dir(cfg.router_library, cfg.single_fixed)
    bid = bench_id or cfg.bench_id
    guard = (
        SpendGuard(
            cap_usd=cap,
            fallback_usd_per_run=cfg.usd_per_run,
            stop_file=Path(runs_dir) / "bench" / bid / "STOP",
            require_priced=mode == "live",
        )
        if cap is not None
        else None
    )
    return run_benchmark(
        tasks=tasks,
        repeats=_effective_repeats(cfg, max_repeats),
        runs_dir=runs_dir,
        router_library=library,
        bench_id=bid,
        planner="mock" if mode == "dry-run" else "llm",
        model=cfg.worker_model,
        arms=cfg.arms,
        spend_guard=guard,
    )


def _summarize(report: BenchReport) -> dict[str, Any]:
    status: dict[str, dict[str, int]] = {}
    for r in report.rows:
        by = status.setdefault(r.arm, {})
        by[r.status] = by.get(r.status, 0) + 1
    return {"bench_id": report.bench_id, "rows": len(report.rows), "status": status}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m dynamic_subgraphs._bench_main")
    p.add_argument("--config", required=True, type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimate", action="store_true", help="plan + cost only")
    mode.add_argument("--dry-run", action="store_true", help="mock, zero tokens")
    mode.add_argument("--live", action="store_true", help="PAID live run")
    mode.add_argument(
        "--analyze",
        type=Path,
        metavar="BENCH_JSONL",
        help="apply the frozen §4 rule; writes verdict.json + verdict.md",
    )
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--bench-id", default=None, help="override config bench_id")
    p.add_argument(
        "--max-repeats",
        type=int,
        default=None,
        help="cap every stratum's repeats (smoke runs only; not the registered sizing)",
    )
    p.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help="kill-switch: stop before spend exceeds this (overrides spend_cap_usd)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="--analyze output dir (default: next to the bench.jsonl)",
    )
    args = p.parse_args(argv)

    cfg = MainRunConfig.load(args.config)
    if args.analyze is not None:
        _, md = analyze_file(
            args.analyze,
            dynamic_arms=cfg.dynamic_arms,
            baseline_arm=cfg.baseline_arm,
            headline_arm=cfg.headline_arm,
            expected_repeats=cfg.repeats_by_task_type,
            out_dir=args.out_dir,
        )
        print(md)
        dest = args.out_dir or args.analyze.parent
        print(f"report: {dest / 'verdict.json'}")
        return 0

    tasks = BenchTask.from_jsonl(cfg.pack)
    plan = plan_run(cfg, tasks, args.max_repeats)
    print(
        f"plan: {plan.tasks} tasks x {plan.arms} arms -> {plan.runs} runs "
        f"{plan.runs_by_task_type}"
    )
    print(
        f"estimate: ${plan.estimated_usd:.2f} "
        f"(${plan.estimated_usd_with_margin:.2f} with x{cfg.safety_margin} margin), "
        f"~{plan.estimated_hours:.1f} h sequential"
    )
    cap = args.max_spend_usd if args.max_spend_usd is not None else cfg.spend_cap_usd
    print(f"kill-switch: spend cap {'none' if cap is None else f'${cap:.2f}'}")
    if args.estimate:
        return 0

    try:
        report = _execute(cfg, args)
    except BenchSpendAbort as exc:
        print(f"ABORTED by kill-switch: {exc}", file=sys.stderr)
        print("completed rows are in bench.jsonl; see spend.json", file=sys.stderr)
        return 3
    print(json.dumps(_summarize(report), indent=2))
    return 0


def _execute(cfg: MainRunConfig, args: argparse.Namespace) -> BenchReport:
    if args.live:
        # LangSmith quota was exhausted during the pilot; tracing only adds
        # retry latency for dropped data, so keep it off (pilot parity).
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        runs_dir = args.runs_dir or Path("runs")
        report = run_main(
            cfg,
            mode="live",
            runs_dir=runs_dir,
            bench_id=args.bench_id,
            max_repeats=args.max_repeats,
            max_spend_usd=args.max_spend_usd,
        )
    else:
        runs_dir = args.runs_dir or Path(tempfile.mkdtemp(prefix="bench-dry-"))
        report = run_main(
            cfg,
            mode="dry-run",
            runs_dir=runs_dir,
            bench_id=args.bench_id or f"{cfg.bench_id}-dryrun",
            max_repeats=args.max_repeats,
            max_spend_usd=args.max_spend_usd,
        )
    print(f"rows: {Path(runs_dir) / 'bench' / report.bench_id / 'bench.jsonl'}")
    return report


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
