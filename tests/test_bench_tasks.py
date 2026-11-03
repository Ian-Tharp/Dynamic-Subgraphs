"""Task-pack integrity: the pilot pack (PR8c) and the main-run pack."""

from collections import Counter
from pathlib import Path
from typing import Any

from app.models import NodeKind
from dynamic_subgraphs.bench import BenchTask, RouterLibrary, fill_spec, run_benchmark

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


# ---------------------------------------------------------------------------
# Main-run pack (docs/evals/bench-tasks-main-v1.jsonl) — frozen rule §4:
# 24 tasks, 4 per type across 6 types, incl. the topology-eliciting stratum.
# ---------------------------------------------------------------------------

MAIN_PACK = Path("docs/evals/bench-tasks-main-v1.jsonl")
MAIN_TYPES = {"compare", "research", "summarize", "extract", "plan", "synthesize"}
# Question-style prompts carry no inline answer material, so echoed checklist
# items are free points. "synthesize" (the topology stratum) has inline data,
# but its checklist is DERIVED results, so it is echo-guarded too.
_ECHO_GUARDED = ("compare", "research", "plan", "synthesize")
_MATERIAL_TYPES = ("summarize", "extract")
# Tavily rejects queries over 400 characters with HTTP 400 (the pilot's
# authored-arm failure mode on long material prompts).
TAVILY_MAX_QUERY_CHARS = 400


def _web_search_queries(spec: Any) -> list[str]:
    return [
        n.params["args"]["query"]
        for n in spec.nodes
        if n.kind == NodeKind.TOOL_CALL and n.params.get("tool_name") == "web_search"
    ]


def test_main_pack_is_24_tasks_stratified_4_per_type() -> None:
    tasks = BenchTask.from_jsonl(MAIN_PACK)
    assert len(tasks) == 24
    assert len({t.task_id for t in tasks}) == 24, "task_ids must be unique"
    assert len({t.prompt for t in tasks}) == 24, "prompts must be unique"
    counts = Counter(t.task_type for t in tasks)
    assert set(counts) == MAIN_TYPES
    assert all(n == 4 for n in counts.values()), counts
    for t in tasks:
        assert t.task_id.startswith(f"main-{t.task_type}-"), t.task_id


def test_main_pack_has_no_overlap_with_pilot() -> None:
    main = BenchTask.from_jsonl(MAIN_PACK)
    pilot = BenchTask.from_jsonl(PACK)
    assert not {t.task_id for t in main} & {t.task_id for t in pilot}
    assert not {t.prompt for t in main} & {t.prompt for t in pilot}
    # No reused material either: no pilot prompt's opening sentence reappears.
    for p in pilot:
        opening = p.prompt.split(".")[0].lower()
        for m in main:
            assert opening not in m.prompt.lower(), (m.task_id, p.task_id)


def test_main_pack_references_well_formed() -> None:
    for t in BenchTask.from_jsonl(MAIN_PACK):
        assert len(t.reference.checklist) >= 5, f"{t.task_id}: thin checklist"
        assert len(set(t.reference.checklist)) == len(t.reference.checklist)
        for item in [*t.reference.checklist, *t.reference.must_include]:
            assert item == item.lower() and item.strip() == item, (t.task_id, item)
            assert 2 <= len(item) <= 40, (t.task_id, item)
        assert t.reference.gold_answer is None


def test_main_pack_grounding_is_research_only() -> None:
    for t in BenchTask.from_jsonl(MAIN_PACK):
        assert t.reference.grounding_required == (t.task_type == "research"), t.task_id


def test_main_pack_no_brace_literals() -> None:
    for t in BenchTask.from_jsonl(MAIN_PACK):
        assert "{" not in t.prompt and "}" not in t.prompt, t.task_id


def test_main_pack_no_checklist_echo() -> None:
    """At most 2 deliberate anchors may appear in the prompt; the rest must be
    things only a genuine (or, for synthesize, correctly computed) answer
    surfaces. Research items must not collide with router query suffixes."""
    for t in BenchTask.from_jsonl(MAIN_PACK):
        if t.task_type not in _ECHO_GUARDED:
            continue
        prompt = t.prompt.lower()
        echoed = [i for i in t.reference.checklist if i in prompt]
        assert len(echoed) <= 2, f"{t.task_id}: prompt-echoed items {echoed}"
        assert len(t.reference.checklist) - len(echoed) >= 3, t.task_id
        if t.task_type == "research":
            for item in t.reference.checklist:
                for suffix in _ROUTER_QUERY_SUFFIXES:
                    assert item not in suffix, (t.task_id, item, suffix)


def test_main_pack_material_is_inline_and_answers_present() -> None:
    for t in BenchTask.from_jsonl(MAIN_PACK):
        if t.task_type not in _MATERIAL_TYPES:
            continue
        if t.task_type == "summarize":
            # ~250-400 words: enough to summarize, short of the pilot's
            # ~2,800-char prompts where the nano planner's output degraded.
            assert 1200 <= len(t.prompt) <= 2400, (t.task_id, len(t.prompt))
        else:
            assert 300 <= len(t.prompt) <= 1500, (t.task_id, len(t.prompt))
        for item in t.reference.checklist:
            assert item in t.prompt.lower(), f"{t.task_id}: {item!r} not in material"


def test_main_pack_topology_stratum_shape() -> None:
    """The synthesize stratum: self-contained multi-item inputs that need
    per-item work then a reduce/decision, so fan-out or multi-stage plans
    plausibly beat the router's single llm_call (summarize fallback)."""
    lib = RouterLibrary.from_dir(_LIBRARY_DIR, _SINGLE_FIXED)
    for t in BenchTask.from_jsonl(MAIN_PACK):
        if t.task_type != "synthesize":
            continue
        assert not t.reference.grounding_required
        assert 600 <= len(t.prompt) <= 2400, (t.task_id, len(t.prompt))
        routed = lib.route(t)
        assert routed.graph_id == "router-summarize-v1"
        assert len(routed.nodes) == 1, "baseline here is one router llm_call"


def test_main_pack_prompts_fit_tavily_query_limit() -> None:
    """Every routed web_search query must fit Tavily's 400-char limit, and so
    must the authored arm's query for every question-style task. Material
    tasks (summarize/extract/synthesize) are EXPECTED to exceed it on the
    authored arm — that is the sanity-floor shape failure the pilot recorded,
    and it fails fast before any LLM spend."""
    lib = RouterLibrary.from_dir(_LIBRARY_DIR, _SINGLE_FIXED)
    for t in BenchTask.from_jsonl(MAIN_PACK):
        for q in _web_search_queries(fill_spec(lib.route(t), t.prompt)):
            assert len(q) <= TAVILY_MAX_QUERY_CHARS, (t.task_id, len(q))
        authored = _web_search_queries(fill_spec(lib.single_fixed, t.prompt))
        if t.task_type in ("compare", "research", "plan"):
            assert all(len(q) <= TAVILY_MAX_QUERY_CHARS for q in authored), t.task_id
