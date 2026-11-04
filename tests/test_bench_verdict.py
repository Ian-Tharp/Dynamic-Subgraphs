"""Golden tests for the frozen §4 analyzer (``dynamic_subgraphs._bench_verdict``).

Synthetic ``bench.jsonl`` fixtures are built to land on each of the four
verdicts, plus the bar arithmetic the owner signs off on before the paid run
(see docs/evals/bench-main-run-v1.md §6).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from dynamic_subgraphs._bench_main import MainRunConfig, main
from dynamic_subgraphs._bench_verdict import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    INCONCLUSIVE,
    KEEP,
    KILL,
    LEANING,
    analyze,
    bootstrap_ci,
    decide,
    load_rows,
    min_wins_for_p,
    pareto,
    required_wins,
    sign_test_p,
)
from dynamic_subgraphs.bench import ArmRunRow

CONFIG = Path("docs/evals/bench-main-v1.config.json")
STRATA = ("compare", "research", "summarize", "extract", "plan", "synthesize")
DYN = ("invented-a1", "invented-a2")

# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

# A per-arm, per-task scenario: (quality, full_cost, planner_cost). Headline
# scores are set equal to blended quality so one number drives everything.
Arm = tuple[float, float, float]

ROUTER: Arm = (0.80, 0.0010, 0.0)
# Dynamic outcomes against ROUTER, by construction:
WIN: Arm = (0.90, 0.0008, 0.0002)  # better and cheaper on both bases
LOSE: Arm = (0.70, 0.0020, 0.0002)  # worse and dearer even worker-only
TIE: Arm = (0.90, 0.0020, 0.0002)  # better but dearer on both bases -> tie
# Dearer on full cost only because of the planner: worker-only it wins.
TEMPLATE: Arm = (0.90, 0.0014, 0.0006)
LOW_ROUTER: Arm = (0.30, 0.0010, 0.0)
LOW_DYN: Arm = (0.20, 0.0010, 0.0002)


def _row(
    task_id: str,
    task_type: str,
    arm: str,
    repeat: int,
    spec: Arm,
    *,
    status: str = "ok",
) -> ArmRunRow:
    q, cost, planner = spec
    ok = status == "ok"
    return ArmRunRow(
        bench_id="golden",
        task_id=task_id,
        task_type=task_type,
        arm=arm,
        repeat=repeat,
        run_id=f"golden-{task_id}-{arm}-r{repeat}",
        status=status,
        quality=q if ok else 0.0,
        quality_floor_met=ok and q >= 0.6,
        value_per_ktok=None,
        cost_usd=cost,
        planner_cost_usd=planner if planner else None,
        worker_cost_usd=round(cost - planner, 6) if planner else None,
        total_tokens=1000,
        node_count=1,
        topology_signature=None,
        structural_quality=q if ok else None,
        goal_score=q if ok else None,
    )


def build(
    dyn_outcomes: list[Arm],
    *,
    router: list[Arm] | None = None,
    repeats: int = 2,
) -> list[ArmRunRow]:
    """24 tasks (4 per stratum, in order); both dynamic arms share outcomes."""
    assert len(dyn_outcomes) == 24
    router = router or [ROUTER] * 24
    rows: list[ArmRunRow] = []
    for i, (d, b) in enumerate(zip(dyn_outcomes, router, strict=True)):
        tt = STRATA[i // 4]
        tid = f"{tt}-{i % 4 + 1}"
        for rep in range(repeats):
            for arm in DYN:
                rows.append(_row(tid, tt, arm, rep, d))
            rows.append(_row(tid, tt, "routed", rep, b))
    return rows


def run(rows: list[ArmRunRow]) -> dict[str, Any]:
    return analyze(rows, dynamic_arms=DYN, headline_arm="invented-a2")


def line(report: dict[str, Any], arm: str, headline: str, scope: str) -> Any:
    (hit,) = [
        ln
        for ln in report["lines"]
        if (ln["dynamic_arm"], ln["headline"], ln["scope"]) == (arm, headline, scope)
    ]
    return hit


def headline_verdicts(report: dict[str, Any]) -> set[str]:
    return {ln["verdict"] for ln in report["lines"] if ln["scope"] == "all"}


# ---------------------------------------------------------------------------
# Arithmetic the owner signs off on (runbook §6)
# ---------------------------------------------------------------------------


def test_sign_test_is_exact_binomial_tail() -> None:
    assert sign_test_p(4, 4) == 1 / 16 == 0.0625
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(17, 24) == pytest.approx(536155 / 2**24)
    assert sign_test_p(17, 24) < 0.05 < sign_test_p(16, 24)
    with pytest.raises(ValueError):
        sign_test_p(5, 4)


def test_stratum_of_four_can_never_reach_p_below_alpha() -> None:
    """4 tasks: best one-sided p is 0.0625, so per-stratum is descriptive."""
    assert min_wins_for_p(4) is None
    assert required_wins(4, 0) is None


def test_headline_bar_is_17_of_24_not_60_percent() -> None:
    assert min_wins_for_p(24) == 17  # 71% > 60%: the sign test binds
    assert math.ceil(0.6 * 24) == 15
    assert required_wins(24, 0) == 17


def test_ties_move_the_bar() -> None:
    # Sign-test bar over decisive tasks only:
    assert min_wins_for_p(20) == 15  # 4 ties
    assert min_wins_for_p(16) == 12  # 8 ties
    # With the 60%-of-eligible share, 8 ties make the share bind instead:
    # 12/16 decisive passes p, but 12/24 = 50% < 60%, so 15 wins are needed.
    assert required_wins(24, 4) == 15
    assert required_wins(24, 8) == 15
    assert required_wins(24, 10) is None  # 14 decisive; 15 needed > 14


def test_pareto_and_decide() -> None:
    assert pareto(0.9, 1.0, 0.8, 1.0) == "dynamic"
    assert pareto(0.8, 1.0, 0.8, 2.0) == "dynamic"
    assert pareto(0.8, 2.0, 0.9, 1.0) == "baseline"
    assert pareto(0.9, 2.0, 0.8, 1.0) == "tie"  # trade-off
    assert pareto(0.8, 1.0, 0.8, 1.0) == "tie"  # identical
    assert decide(True, True, False, False) == KEEP
    assert decide(False, False, True, True) == KILL
    assert decide(False, False, True, False) == INCONCLUSIVE  # not "even worker"
    assert decide(False, True, False, False) == LEANING
    assert decide(False, False, False, False) == INCONCLUSIVE


def test_bootstrap_is_seeded_and_golden() -> None:
    ind = [1] * 18 + [0] * 6
    a = bootstrap_ci(ind)
    assert a == bootstrap_ci(ind) == bootstrap_ci(ind, BOOTSTRAP_RESAMPLES, 20260611)
    assert BOOTSTRAP_SEED == 20260611 and BOOTSTRAP_RESAMPLES == 10_000
    assert a == (0.5833333333333334, 0.9166666666666666)
    assert bootstrap_ci([]) is None
    assert bootstrap_ci([1, 1, 1, 1]) == (1.0, 1.0)


# ---------------------------------------------------------------------------
# Golden verdicts
# ---------------------------------------------------------------------------


def test_golden_keep_with_ties_and_floor_ineligible_tasks() -> None:
    # 17 wins, 4 ties, 1 loss, 2 tasks where neither arm meets the floor.
    outcomes = [WIN] * 17 + [TIE] * 4 + [LOSE] + [LOW_DYN] * 2
    router = [ROUTER] * 22 + [LOW_ROUTER] * 2
    report = run(build(outcomes, router=router))
    ln = line(report, "invented-a2", "goal", "all")
    f = ln["full"]
    assert (f["wins"], f["ties"], f["losses"], f["eligible"]) == (17, 4, 1, 22)
    assert ln["ineligible"] == 2 and ln["tasks"] == 24
    assert f["p"] == pytest.approx(sign_test_p(17, 18))
    assert f["share"] == pytest.approx(17 / 22)
    lo, hi = f["ci"]
    assert lo < 17 / 22 < hi
    assert ln["verdict"] == KEEP
    assert headline_verdicts(report) == {KEEP}
    # The λ cut is reported but never decides.
    assert ln["lambda_cut"]["wins"] == 21 and "meets_bar" in ln["lambda_cut"]


def test_golden_kill_requires_router_win_even_worker_only() -> None:
    outcomes = [LOSE] * 18 + [TIE] * 6
    report = run(build(outcomes))
    ln = line(report, "invented-a2", "structural", "all")
    assert ln["baseline_full"]["wins"] == 18
    assert ln["baseline_worker"]["wins"] == 18
    assert ln["baseline_worker"]["meets_bar"]
    assert ln["verdict"] == KILL
    assert headline_verdicts(report) == {KILL}


def test_golden_leaning_template_cache() -> None:
    """Loses (ties) on full cost because of the planner; wins worker-only."""
    outcomes = [TEMPLATE] * 20 + [TIE] * 4
    report = run(build(outcomes))
    ln = line(report, "invented-a2", "goal", "all")
    assert ln["full"]["wins"] == 0 and ln["full"]["ties"] == 24
    assert ln["worker"]["wins"] == 20 and ln["worker"]["meets_bar"]
    assert ln["verdict"] == LEANING
    assert headline_verdicts(report) == {LEANING}


def test_golden_inconclusive_mixed() -> None:
    outcomes = [WIN, LOSE] * 12
    report = run(build(outcomes))
    assert headline_verdicts(report) == {INCONCLUSIVE}
    f = line(report, "invented-a1", "goal", "all")["full"]
    assert (f["wins"], f["losses"]) == (12, 12)


def test_one_fewer_win_flips_keep_to_inconclusive() -> None:
    keep = run(build([WIN] * 17 + [LOSE] * 7))
    near = run(build([WIN] * 16 + [LOSE] * 8))
    assert line(keep, "invented-a2", "goal", "all")["verdict"] == KEEP
    assert line(near, "invented-a2", "goal", "all")["verdict"] == INCONCLUSIVE


def test_share_bar_binds_with_many_ties() -> None:
    """12 wins / 8 ties / 4 losses: p=0.038 passes, 50% share does not."""
    report = run(build([WIN] * 12 + [TIE] * 8 + [LOSE] * 4))
    f = line(report, "invented-a2", "goal", "all")["full"]
    assert f["p"] < 0.05 and f["share"] == 0.5
    assert not f["meets_bar"]
    assert line(report, "invented-a2", "goal", "all")["verdict"] == INCONCLUSIVE


def test_floor_only_one_arm_met_counts_as_that_arm_winning() -> None:
    # Dynamic below floor, router above: router wins regardless of cost.
    cheap_low = (0.50, 0.0001, 0.00005)
    report = run(build([cheap_low] * 24))
    ln = line(report, "invented-a2", "goal", "all")
    assert ln["baseline_full"]["wins"] == 24 and ln["verdict"] == KILL


def test_failed_runs_score_zero() -> None:
    rows = build([WIN] * 24)
    # Fail every repeat of the dynamic arm on one task: mean quality 0.
    rows = [
        _row(r.task_id, r.task_type, r.arm, r.repeat, WIN, status="plan_failed")
        if (r.task_id == "compare-1" and r.arm in DYN)
        else r
        for r in rows
    ]
    report = run(rows)
    (pt,) = [
        c
        for c in report["per_task"]
        if c["task_id"] == "compare-1"
        and c["dynamic_arm"] == "invented-a2"
        and c["headline"] == "goal"
    ]
    assert not pt["dynamic_floor_met"] and pt["full"] == "baseline"


def test_strata_are_descriptive_and_never_decide() -> None:
    report = run(build([WIN] * 24))
    for s in STRATA:
        ln = line(report, "invented-a2", "goal", s)
        assert ln["full"]["wins"] == 4 and ln["full"]["p"] == 0.0625
        assert not ln["decisive_scope"] and ln["verdict"] == INCONCLUSIVE


def test_arms_keyed_by_name_not_origin() -> None:
    """A1 and A2 are both 'invented'; each gets its own lines."""
    rows = build([WIN] * 24)
    rows = [
        _row(r.task_id, r.task_type, r.arm, r.repeat, LOSE)
        if r.arm == "invented-a1"
        else r
        for r in rows
    ]
    report = run(rows)
    assert line(report, "invented-a1", "goal", "all")["verdict"] == KILL
    assert line(report, "invented-a2", "goal", "all")["verdict"] == KEEP


def test_unpriced_rows_fall_back_to_tokens_and_warn() -> None:
    rows = [
        ArmRunRow(**{**asdict(r), "cost_usd": None, "worker_cost_usd": None})
        for r in build([WIN] * 24)
    ]
    report = run(rows)
    assert any("token cost axis" in w for w in report["warnings"])
    assert {c["cost_axis"] for c in report["per_task"]} == {"tokens"}


def test_refuses_mixed_bench_ids_duplicates_and_missing_arms() -> None:
    rows = build([WIN] * 24)
    with pytest.raises(ValueError, match="bench_ids"):
        run([*rows, ArmRunRow(**{**asdict(rows[0]), "bench_id": "other"})])
    with pytest.raises(ValueError, match="duplicate"):
        run([*rows, rows[0]])
    with pytest.raises(ValueError, match="no rows"):
        run([r for r in rows if r.arm != "routed"])


def test_sizing_check_flags_capped_runs() -> None:
    report = analyze(
        build([WIN] * 24, repeats=1),
        dynamic_arms=DYN,
        expected_repeats=MainRunConfig.load(CONFIG).repeats_by_task_type,
    )
    assert report["registered_sizing"] is False
    assert any("not the registered sizing" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write(rows: list[ArmRunRow], path: Path) -> Path:
    path.write_text("".join(json.dumps(asdict(r)) + "\n" for r in rows), "utf-8")
    return path


def test_cli_analyze_writes_json_and_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jsonl = _write(build([WIN] * 17 + [LOSE] * 7), tmp_path / "bench.jsonl")
    assert main(["--config", str(CONFIG), "--analyze", str(jsonl)]) == 0
    out = capsys.readouterr().out
    assert "| invented-a2 | goal | all | 24 | 17 | 0 | 7 |" in out
    assert "**KEEP**" in out
    report = json.loads((tmp_path / "verdict.json").read_text("utf-8"))
    assert report["headline_arm"] == "invented-a2"
    assert report["dynamic_arms"] == list(DYN)
    assert (tmp_path / "verdict.md").read_text("utf-8").startswith("# §4 verdict")
    assert len(load_rows(jsonl)) == 24 * 2 * 3


def test_cli_analyze_on_dry_run_output(tmp_path: Path) -> None:
    """The analyzer accepts real harness output (capped dry-run for CI speed;
    the full 1,152-row dry-run is the same path, see the runbook)."""
    runs = tmp_path / "runs"
    args = ["--config", str(CONFIG), "--runs-dir", str(runs)]
    assert main([*args, "--dry-run", "--max-repeats", "1"]) == 0
    jsonl = runs / "bench" / "main-v1-dryrun" / "bench.jsonl"
    assert main(["--config", str(CONFIG), "--analyze", str(jsonl)]) == 0
    report = json.loads((jsonl.parent / "verdict.json").read_text("utf-8"))
    assert report["rows"] == 96 and report["tasks"] == 24
    # 2 sub-arms x 2 headlines x (all + 6 strata)
    assert len(report["lines"]) == 2 * 2 * 7
    assert report["registered_sizing"] is False  # --max-repeats 1
    assert any("mock" in w for w in report["warnings"])
