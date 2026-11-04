"""The live-run spend kill-switch (``SpendGuard``). Offline; spends nothing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dynamic_subgraphs import DynamicSubgraphs, EvalReference
from dynamic_subgraphs._bench_main import MainRunConfig, main, run_main
from dynamic_subgraphs.bench import (
    ArmRunRow,
    BenchSpendAbort,
    BenchTask,
    RouterLibrary,
    SpendGuard,
    run_benchmark,
)
from dynamic_subgraphs.engine import RunResult
from dynamic_subgraphs.usage import TokenUsage

CONFIG = Path("docs/evals/bench-main-v1.config.json")
LIB = RouterLibrary.from_dir(
    Path("docs/evals/router-library-v1"), Path("docs/evals/fixed-host-graph-v1.json")
)


def _row(cost: float | None, tokens: int = 100, status: str = "ok") -> ArmRunRow:
    return ArmRunRow(
        bench_id="b",
        task_id="t",
        task_type="compare",
        arm="routed",
        repeat=0,
        run_id="b-t-routed-r0",
        status=status,
        quality=0.0,
        quality_floor_met=False,
        value_per_ktok=None,
        cost_usd=cost,
        planner_cost_usd=None,
        worker_cost_usd=None,
        total_tokens=tokens,
        node_count=None,
        topology_signature=None,
    )


def test_guard_books_cost_and_stops_before_the_cap() -> None:
    g = SpendGuard(cap_usd=0.01, fallback_usd_per_run={"routed": 0.003})
    g.before_run("routed", "r0")
    g.charge(_row(0.004))
    g.before_run("routed", "r1")  # 0.004 + 0.003 <= 0.01
    g.charge(_row(0.004))
    with pytest.raises(BenchSpendAbort, match="spend cap"):
        g.before_run("routed", "r2")  # 0.008 + 0.003 > 0.01
    assert g.aborted and g.to_dict()["spent_usd"] == 0.008


def test_guard_charges_estimate_for_unpriced_spend_only() -> None:
    g = SpendGuard(cap_usd=1.0, fallback_usd_per_run={"routed": 0.002})
    g.charge(_row(None, tokens=0))  # mock run: nothing spent
    g.charge(_row(None, tokens=50, status="plan_failed"))
    assert g.spent_usd == 0.002 and g.unpriced_runs == 1 and g.runs == 2


def test_guard_live_refuses_untrackable_spend() -> None:
    g = SpendGuard(cap_usd=1.0, require_priced=True)
    g.charge(_row(None, tokens=50, status="plan_failed"))  # failure: estimate
    with pytest.raises(BenchSpendAbort, match="no price"):
        g.charge(_row(None, tokens=50))


def test_guard_stop_file(tmp_path: Path) -> None:
    stop = tmp_path / "STOP"
    g = SpendGuard(cap_usd=1.0, stop_file=stop)
    g.before_run("routed", "r0")
    stop.touch()
    with pytest.raises(BenchSpendAbort, match="stop file"):
        g.before_run("routed", "r1")


def test_guard_rejects_non_positive_cap() -> None:
    with pytest.raises(ValueError):
        SpendGuard(cap_usd=0)


def _priced_failure(*args: Any, **kwargs: Any) -> RunResult:
    return RunResult(
        run_id=kwargs["run_id"],
        status="execution_failed",
        response=None,
        usage=TokenUsage(input_tokens=80, output_tokens=20, total_tokens=100),
        cost=0.004,
    )


def test_run_benchmark_aborts_mid_run_and_keeps_rows(tmp_path: Path) -> None:
    tasks = [
        BenchTask(f"t{i}", "summarize", "Summarize: sky.", EvalReference())
        for i in range(2)
    ]
    guard = SpendGuard(cap_usd=0.01)
    with (
        patch.object(DynamicSubgraphs, "run", side_effect=_priced_failure),
        pytest.raises(BenchSpendAbort),
    ):
        run_benchmark(
            tasks=tasks,
            repeats=1,
            runs_dir=tmp_path,
            router_library=LIB,
            bench_id="kill",
            planner="mock",
            spend_guard=guard,
        )
    bench = tmp_path / "bench" / "kill"
    rows = (bench / "bench.jsonl").read_text("utf-8").splitlines()
    assert len(rows) == 3  # 3 x $0.004 = $0.012 > $0.01: run 4 never starts
    spend = json.loads((bench / "spend.json").read_text("utf-8"))
    assert spend["spent_usd"] == 0.012 and spend["aborted"]
    assert not (bench / "report.json").exists()


def test_config_pins_cap_at_top_of_registered_envelope() -> None:
    cfg = MainRunConfig.load(CONFIG)
    assert cfg.spend_cap_usd == 8.0
    assert cfg.headline_arm == "invented-a2"
    assert cfg.dynamic_arms == ("invented-a1", "invented-a2")
    assert cfg.baseline_arm == "routed"


def test_live_run_refused_without_a_cap(tmp_path: Path) -> None:
    cfg = replace(MainRunConfig.load(CONFIG), spend_cap_usd=None)
    env = {"OPENAI_API_KEY": "x", "TAVILY_API_KEY": "x"}
    with (
        patch.object(DynamicSubgraphs, "run", side_effect=AssertionError("ran")),
        pytest.raises(SystemExit, match="no spend cap"),
    ):
        run_main(cfg, mode="live", runs_dir=tmp_path, env=env)


def test_cli_kill_switch_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Cap below one A1 run's estimate: stops before the first run.
    rc = main(
        [
            "--config",
            str(CONFIG),
            "--dry-run",
            "--max-repeats",
            "1",
            "--runs-dir",
            str(tmp_path),
            "--max-spend-usd",
            "0.001",
        ]
    )
    assert rc == 3
    err = capsys.readouterr().err
    assert "ABORTED by kill-switch" in err
    spend = json.loads(
        (tmp_path / "bench" / "main-v1-dryrun" / "spend.json").read_text("utf-8")
    )
    assert spend["runs"] == 0 and spend["cap_usd"] == 0.001
