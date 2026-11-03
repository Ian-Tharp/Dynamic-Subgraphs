"""The main-run config + one-command runner (offline; never spends tokens)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dynamic_subgraphs import DynamicSubgraphs, Model
from dynamic_subgraphs._bench_main import (
    LIVE_ENV_VARS,
    MainRunConfig,
    main,
    plan_run,
    run_main,
)
from dynamic_subgraphs.bench import BenchTask

CONFIG = Path("docs/evals/bench-main-v1.config.json")
MAIN_TYPES = {"compare", "research", "summarize", "extract", "plan", "synthesize"}


def _no_spend(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("engine.run must not be called")


def test_config_pins_arms_and_models() -> None:
    cfg = MainRunConfig.load(CONFIG)
    assert [a.name for a in cfg.arms] == [
        "invented-a1",
        "invented-a2",
        "routed",
        "authored",
    ]
    nano = Model("openai", "gpt-5.4-nano-2026-03-17")
    assert cfg.worker_model == nano
    a1, a2, b, c = cfg.arms
    assert (a1.origin, a1.planner_model) == ("invented", nano)
    # A2: a stronger, explicitly pinned (dated-snapshot) planner.
    assert a2.origin == "invented"
    assert a2.planner_model == Model("openai", "gpt-5.4-mini-2026-03-17")
    assert a2.planner_model != a1.planner_model
    assert (b.origin, c.origin) == ("routed", "authored")
    assert b.planner_model is None and c.planner_model is None


def test_config_repeats_follow_pilot_derivation_and_cap() -> None:
    """Rule §4: per-task repeats from the §3 CV table, capped at 20."""
    cfg = MainRunConfig.load(CONFIG)
    assert cfg.repeats_cap == 20
    assert set(cfg.repeats_by_task_type) == MAIN_TYPES
    assert cfg.repeats_by_task_type == {
        "compare": 20,  # derived 25 -> cap
        "research": 20,  # max(35, 7) -> cap
        "summarize": 3,
        "extract": 6,
        "plan": 3,
        "synthesize": 20,  # new type, no pilot CV -> cap (upper bound)
    }


def test_config_pack_matches_the_committed_main_pack() -> None:
    cfg = MainRunConfig.load(CONFIG)
    assert cfg.pack == Path("docs/evals/bench-tasks-main-v1.jsonl")
    assert cfg.router_library.is_dir() and cfg.single_fixed.is_file()


def test_full_run_plan_and_estimate() -> None:
    cfg = MainRunConfig.load(CONFIG)
    plan = plan_run(cfg, BenchTask.from_jsonl(cfg.pack))
    # 4 tasks/type x sum(repeats)=72 x 4 arms
    assert plan.runs == 4 * 72 * 4 == 1152
    assert plan.runs_by_task_type["synthesize"] == 4 * 20 * 4
    # Inside the rule's registered envelope (~$4-8) even with the margin.
    assert 2.0 < plan.estimated_usd < plan.estimated_usd_with_margin <= 8.0


def test_dry_run_of_main_command_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exact main-run CLI, in --dry-run (mock planner + mock runners).

    Repeats are capped at 1 to keep CI fast; the full 1,152-run dry-run is
    the same code path (documented in docs/evals/bench-main-run-v1.md).
    """
    rc = main(
        [
            "--config",
            str(CONFIG),
            "--dry-run",
            "--max-repeats",
            "1",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "24 tasks x 4 arms -> 96 runs" in out

    jsonl = tmp_path / "bench" / "main-v1-dryrun" / "bench.jsonl"
    rows = [json.loads(ln) for ln in jsonl.read_text("utf-8").splitlines()]
    assert len(rows) == 96
    assert {r["status"] for r in rows} == {"ok"}
    assert {r["task_type"] for r in rows} == MAIN_TYPES
    planners = {r["arm"]: r["planner_model"] for r in rows}
    assert planners == {
        "invented-a1": "openai/gpt-5.4-nano-2026-03-17",
        "invented-a2": "openai/gpt-5.4-mini-2026-03-17",
        "routed": None,
        "authored": None,
    }
    assert (tmp_path / "bench" / "main-v1-dryrun" / "report.json").is_file()


def test_estimate_mode_runs_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(DynamicSubgraphs, "run", side_effect=_no_spend):
        assert main(["--config", str(CONFIG), "--estimate"]) == 0
    assert "1152 runs" in capsys.readouterr().out


def test_live_run_refused_without_keys_before_any_spend(tmp_path: Path) -> None:
    cfg = MainRunConfig.load(CONFIG)
    with (
        patch.object(DynamicSubgraphs, "run", side_effect=_no_spend),
        pytest.raises(SystemExit, match="missing env vars"),
    ):
        run_main(cfg, mode="live", runs_dir=tmp_path, env={})
    assert set(LIVE_ENV_VARS) == {"OPENAI_API_KEY", "TAVILY_API_KEY"}


def test_invented_arm_without_pinned_planner_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text("utf-8"))
    del raw["arms"][1]["planner_model"]
    bad = tmp_path / "cfg.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must pin planner_model"):
        MainRunConfig.load(bad)


def test_repeats_above_cap_rejected(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text("utf-8"))
    raw["repeats_by_task_type"]["compare"] = 25
    bad = tmp_path / "cfg.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="cap=20"):
        MainRunConfig.load(bad)
