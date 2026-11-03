"""One-command runner for the main dynamic-vs-router benchmark (internal).

Reads a committed run config (``docs/evals/bench-main-v1.config.json``) that
pins the pack, arms, planner/worker models and per-stratum repeats, then
calls :func:`dynamic_subgraphs.bench.run_benchmark`. Three modes:

    # Print the run plan + cost estimate. No engine, no network.
    python -m dynamic_subgraphs._bench_main --config <cfg> --estimate

    # Full schedule offline: mock planner + mock runners, zero tokens.
    python -m dynamic_subgraphs._bench_main --config <cfg> --dry-run

    # The paid live run. Requires --live AND the provider keys.
    python -m dynamic_subgraphs._bench_main --config <cfg> --live

Internal tooling, not public API (underscore module, see
docs/api-stability.md). The KEEP/KILL rule is NOT applied here; it lives in
the results doc so code can't quietly re-score the verdict.
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

from dynamic_subgraphs.bench import (
    BenchArm,
    BenchReport,
    BenchTask,
    RouterLibrary,
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
        )


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
) -> BenchReport:
    """Execute the configured run. ``mode`` is ``"dry-run"`` or ``"live"``.

    Dry-run uses the mock planner and mock runners (zero tokens) on the exact
    same pack, arms and repeat schedule as the live run.
    """
    if mode not in ("dry-run", "live"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "live":
        env = os.environ if env is None else env
        missing = [k for k in LIVE_ENV_VARS if not env.get(k)]
        if missing:
            raise SystemExit(
                f"live run refused: missing env vars {missing}. Nothing was spent."
            )
    tasks = BenchTask.from_jsonl(cfg.pack)
    library = RouterLibrary.from_dir(cfg.router_library, cfg.single_fixed)
    return run_benchmark(
        tasks=tasks,
        repeats=_effective_repeats(cfg, max_repeats),
        runs_dir=runs_dir,
        router_library=library,
        bench_id=bench_id or cfg.bench_id,
        planner="mock" if mode == "dry-run" else "llm",
        model=cfg.worker_model,
        arms=cfg.arms,
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
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--bench-id", default=None, help="override config bench_id")
    p.add_argument(
        "--max-repeats",
        type=int,
        default=None,
        help="cap every stratum's repeats (smoke runs only; not the registered sizing)",
    )
    args = p.parse_args(argv)

    cfg = MainRunConfig.load(args.config)
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
    if args.estimate:
        return 0

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
        )
    else:
        runs_dir = args.runs_dir or Path(tempfile.mkdtemp(prefix="bench-dry-"))
        report = run_main(
            cfg,
            mode="dry-run",
            runs_dir=runs_dir,
            bench_id=args.bench_id or f"{cfg.bench_id}-dryrun",
            max_repeats=args.max_repeats,
        )
    print(json.dumps(_summarize(report), indent=2))
    print(f"rows: {Path(runs_dir) / 'bench' / report.bench_id / 'bench.jsonl'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
