"""The pilot task pack is loadable, referenced, heterogeneous (PR8c)."""

from pathlib import Path

from dynamic_subgraphs.bench import BenchTask, RouterLibrary, run_benchmark

PACK = Path("docs/evals/bench-tasks-pilot-v1.jsonl")

_LIBRARY_DIR = Path("docs/evals/router-library-v1")
_SINGLE_FIXED = Path("docs/evals/fixed-host-graph-v1.json")


def test_pilot_pack_loads_six_heterogeneous_tasks() -> None:
    tasks = BenchTask.from_jsonl(PACK)
    assert len(tasks) == 6
    assert len({t.task_type for t in tasks}) >= 4
    assert len({t.task_id for t in tasks}) == 6


def test_references_are_mandatory_and_well_formed() -> None:
    for t in BenchTask.from_jsonl(PACK):
        required = [*t.reference.checklist, *t.reference.must_include]
        assert required, f"{t.task_id}: empty reference"
        for item in required:
            assert item == item.lower(), (
                f"{t.task_id}: checklist must be lowercase (substring matching)"
            )
            assert len(item) <= 40, (
                f"{t.task_id}: checklist items must be short surface tokens"
            )


def test_grounding_flags_split() -> None:
    tasks = BenchTask.from_jsonl(PACK)
    assert any(t.reference.grounding_required for t in tasks)
    assert any(not t.reference.grounding_required for t in tasks)
    # research tasks and ONLY research tasks require grounding
    for t in tasks:
        assert t.reference.grounding_required == (t.task_type == "research"), t.task_id


def test_no_brace_literals_in_prompts() -> None:
    for t in BenchTask.from_jsonl(PACK):
        assert "{" not in t.prompt and "}" not in t.prompt, t.task_id


def test_self_contained_material_tasks() -> None:
    by_id = {t.task_id: t for t in BenchTask.from_jsonl(PACK)}
    summarize = next(t for t in by_id.values() if t.task_type == "summarize")
    assert len(summarize.prompt) > 1500, (
        "summarize material must be inline (~400-600 words)"
    )
    extract = next(t for t in by_id.values() if t.task_type == "extract")
    assert len(extract.prompt) > 300, "extract record must be inline"
    # checklist values must actually appear in the extract material (they're the answers)
    for item in extract.reference.checklist:
        assert item in extract.prompt.lower(), (
            f"extract checklist {item!r} not in material"
        )


# The router's research spec appends these fixed query suffixes (see
# docs/evals/router-library-v1/research.json); a checklist item that is a
# substring of either could score free points off query text echoed into
# search results rather than off genuine answer content.
_ROUTER_QUERY_SUFFIXES = ("recent developments", "limitations criticism")


def test_question_tasks_avoid_prompt_echo() -> None:
    """Checklist items echoed in the prompt are free points for lazy answers.

    Question-style tasks (compare/research/plan) may keep at most 2 deliberate
    subject anchors; everything else must be a term only a genuine answer
    surfaces. Extract/summarize are exempt — their material is inline by
    design. Research checklists additionally must not collide with the
    router's fixed query suffixes.
    """
    for t in BenchTask.from_jsonl(PACK):
        if t.task_type not in ("compare", "research", "plan"):
            continue
        prompt = t.prompt.lower()
        echoed = [item for item in t.reference.checklist if item in prompt]
        assert len(echoed) <= 2, f"{t.task_id}: prompt-echoed checklist items {echoed}"
        if t.task_type == "research":
            for item in t.reference.checklist:
                for suffix in _ROUTER_QUERY_SUFFIXES:
                    assert item not in suffix, (
                        f"{t.task_id}: checklist item {item!r} is a substring of "
                        f"router query suffix {suffix!r}"
                    )


def test_pack_runs_offline_through_harness(tmp_path: Path) -> None:
    """6 tasks x 3 arms x 1 repeat = 18 rows; no exceptions (token-free smoke)."""
    lib = RouterLibrary.from_dir(_LIBRARY_DIR, _SINGLE_FIXED)
    tasks = BenchTask.from_jsonl(PACK)

    report = run_benchmark(
        tasks=tasks,
        repeats=1,
        runs_dir=tmp_path,
        router_library=lib,
        bench_id="pack-smoke",
        planner="mock",
    )

    assert len(report.rows) == 18, f"expected 18 rows, got {len(report.rows)}"
