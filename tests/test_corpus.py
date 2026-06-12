"""EvalCorpus reader + paired Pareto comparison (Slice 7 PR6)."""

from datetime import UTC, datetime
from pathlib import Path

from dynamic_subgraphs.corpus import EvalCorpus, OriginComparison
from dynamic_subgraphs.eval import EvalResult
from dynamic_subgraphs.eval.types import Origin, RunFingerprint


def _eval_result(
    run_id: str,
    *,
    origin: Origin,
    quality: float,
    tokens: int,
    cost: float | None = None,
    dims: tuple[str, ...] = ("plan_validity", "goal_completion", "cost_efficiency"),
    rubric: str = "ds.default.v1",
    floor_met: bool = True,
) -> EvalResult:
    return EvalResult(
        run_id=run_id,
        scored_at=datetime.now(UTC),
        gate="deterministic@v1",
        rubric_id=rubric,
        weights={},
        applicable_dimensions=list(dims),
        quality=quality,
        components=[],
        overall_verdict="pass",
        cost_usd=cost,
        planner_cost_usd=None,
        worker_cost_usd=None,
        total_tokens=tokens,
        value_per_ktok=quality / (tokens / 1000.0),
        value_per_usd=None,
        quality_floor_met=floor_met,
        fingerprint=RunFingerprint(
            graph_id="g",
            goal="goal",
            prompt_sha256="p" * 64,
            node_count=2,
            edge_count=2,
            max_depth=1,
            node_kind_histogram={"llm_call": 2},
            has_grounded_tool=False,
            planner_model=None,
            worker_model="m",
            topology_signature="t" * 64,
            instruction_sha256="i" * 64,
            origin=origin,
        ),
        deterministic=True,
    )


def _write(tmp_path: Path, result: EvalResult) -> None:
    d = tmp_path / result.run_id
    d.mkdir(parents=True)
    (d / "eval.json").write_text(result.model_dump_json(), encoding="utf-8")


def test_all_skips_unscored_dirs(tmp_path) -> None:
    _write(tmp_path, _eval_result("a", origin="invented", quality=0.8, tokens=1000))
    (tmp_path / "unscored").mkdir()
    corpus = EvalCorpus(root_dir=tmp_path)
    assert [r.run_id for r in corpus.all()] == ["a"]


def test_corrupt_eval_json_skipped(tmp_path) -> None:
    _write(tmp_path, _eval_result("a", origin="invented", quality=0.8, tokens=1000))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "eval.json").write_text("{not json", encoding="utf-8")
    assert [r.run_id for r in EvalCorpus(root_dir=tmp_path).all()] == ["a"]


def test_no_fingerprint_loads_but_excluded_from_groups(tmp_path) -> None:
    _write(tmp_path, _eval_result("a", origin="invented", quality=0.8, tokens=1000))
    bare = _eval_result("b", origin="routed", quality=0.7, tokens=900)
    bare = bare.model_copy(update={"fingerprint": None})
    _write(tmp_path, bare)
    corpus = EvalCorpus(root_dir=tmp_path)
    # Still a valid result -> visible in all()...
    assert [r.run_id for r in corpus.all()] == ["a", "b"]
    # ...but ungroupable without a fingerprint, so excluded from both groupings.
    assert [r.run_id for rs in corpus.by_task().values() for r in rs] == ["a"]
    assert [r.run_id for rs in corpus.by_topology().values() for r in rs] == ["a"]


def test_grouping(tmp_path) -> None:
    _write(tmp_path, _eval_result("a", origin="invented", quality=0.8, tokens=1000))
    _write(tmp_path, _eval_result("b", origin="routed", quality=0.7, tokens=900))
    corpus = EvalCorpus(root_dir=tmp_path)
    assert set(corpus.by_topology()) == {"t" * 64}
    assert set(corpus.by_task()) == {"p" * 64}


def test_compare_origins_pareto_win(tmp_path) -> None:
    # invented: better quality AND cheaper (tokens) -> pareto winner
    _write(tmp_path, _eval_result("a", origin="invented", quality=0.9, tokens=800))
    _write(tmp_path, _eval_result("b", origin="routed", quality=0.7, tokens=1000))
    cmp = EvalCorpus(root_dir=tmp_path).compare_origins("p" * 64)
    assert isinstance(cmp, OriginComparison)
    assert cmp.pareto_winner == "invented"
    assert cmp.incomparable is False


def test_compare_origins_tradeoff_is_a_tie(tmp_path) -> None:
    # better quality but MORE tokens vs cheaper-but-worse: nobody dominates
    _write(tmp_path, _eval_result("a", origin="invented", quality=0.9, tokens=2000))
    _write(tmp_path, _eval_result("b", origin="routed", quality=0.7, tokens=500))
    cmp = EvalCorpus(root_dir=tmp_path).compare_origins("p" * 64)
    assert cmp.pareto_winner is None
    assert cmp.incomparable is False


def test_below_floor_cannot_win(tmp_path) -> None:
    _write(
        tmp_path,
        _eval_result("a", origin="invented", quality=0.9, tokens=500, floor_met=False),
    )
    _write(tmp_path, _eval_result("b", origin="routed", quality=0.65, tokens=1000))
    cmp = EvalCorpus(root_dir=tmp_path).compare_origins("p" * 64)
    assert cmp.pareto_winner == "routed"


def test_usd_cost_preferred_over_tokens(tmp_path) -> None:
    # same quality; a costs less USD though more tokens -> a dominates on USD axis
    _write(
        tmp_path,
        _eval_result("a", origin="invented", quality=0.8, tokens=2000, cost=0.001),
    )
    _write(
        tmp_path,
        _eval_result("b", origin="routed", quality=0.8, tokens=1000, cost=0.004),
    )
    cmp = EvalCorpus(root_dir=tmp_path).compare_origins("p" * 64)
    assert cmp.pareto_winner == "invented"


def test_mismatched_rubric_refused(tmp_path) -> None:
    _write(tmp_path, _eval_result("a", origin="invented", quality=0.9, tokens=800))
    _write(
        tmp_path,
        _eval_result(
            "b", origin="routed", quality=0.7, tokens=900, rubric="ds.other.v9"
        ),
    )
    cmp = EvalCorpus(root_dir=tmp_path).compare_origins("p" * 64)
    assert cmp.incomparable is True and cmp.reason == "rubric_id mismatch"


def test_mismatched_dimensions_refused(tmp_path) -> None:
    _write(tmp_path, _eval_result("a", origin="invented", quality=0.9, tokens=800))
    _write(
        tmp_path,
        _eval_result(
            "b",
            origin="routed",
            quality=0.7,
            tokens=900,
            dims=("plan_validity", "grounding", "goal_completion", "cost_efficiency"),
        ),
    )
    cmp = EvalCorpus(root_dir=tmp_path).compare_origins("p" * 64)
    assert cmp.incomparable is True and cmp.reason == "applicable_dimensions mismatch"


def test_single_run_incomparable(tmp_path) -> None:
    _write(tmp_path, _eval_result("a", origin="invented", quality=0.9, tokens=800))
    cmp = EvalCorpus(root_dir=tmp_path).compare_origins("p" * 64)
    assert cmp.incomparable is True
