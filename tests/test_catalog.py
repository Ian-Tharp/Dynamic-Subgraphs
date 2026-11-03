"""Approved-subgraph catalog mode (app.governance.catalog): governed routing."""

from __future__ import annotations

import pytest

from app.governance import (
    CatalogEntry,
    CatalogPlanner,
    CatalogSelectionError,
    SubgraphCatalog,
    build_llm_catalog_selector,
    check_plan_policy,
    fill_spec,
    keyword_selector,
)
from app.models import GraphSpec, NodeKind
from app.models.graph_spec import GraphBudget
from app.policy import ExecutionPolicy
from app.supervisor.planner import build_demo_spec
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig
from dynamic_subgraphs.bench import fill_spec as bench_fill_spec


@pytest.fixture
def summarize(make_node, make_edge) -> GraphSpec:
    return GraphSpec(
        graph_id="summarize-v1",
        goal="summarize",
        nodes=[
            make_node(
                "summarize",
                outputs=["summary"],
                params={"instruction": "Summarize: {prompt}"},
            )
        ],
        edges=[make_edge("START", "summarize"), make_edge("summarize", "END")],
    )


@pytest.fixture
def research(make_node, make_edge) -> GraphSpec:
    return GraphSpec(
        graph_id="research-v1",
        goal="research",
        nodes=[
            make_node(
                "search",
                NodeKind.TOOL_CALL,
                outputs=["hits"],
                params={"tool_name": "web_search", "args": {"query": "{prompt}"}},
            ),
            make_node(
                "answer",
                inputs=["hits"],
                outputs=["answer"],
                params={
                    "instruction": "Answer '{prompt}' from the hits; quote {prompt}"
                },
            ),
        ],
        edges=[
            make_edge("START", "search"),
            make_edge("search", "answer"),
            make_edge("answer", "END"),
        ],
    )


@pytest.fixture
def catalog(summarize, research) -> SubgraphCatalog:
    return SubgraphCatalog(
        entries=(
            CatalogEntry(
                "summarize", summarize, "Summarize text", ("summarize", "tl;dr")
            ),
            CatalogEntry("research", research, "Search the web", ("search", "find")),
        )
    )


# ---------- matching ----------


def test_filled_entry_matches_its_template(catalog, research) -> None:
    spec = fill_spec(research, "who won {the} cup?")

    entry = catalog.match(spec)

    assert entry is not None
    assert entry.entry_id == "research"


def test_placeholder_must_take_one_value_everywhere(catalog, research) -> None:
    spec = fill_spec(research, "cats")
    # Tamper: one occurrence of the prompt slot differs from the others.
    spec.nodes[1].params["instruction"] = "Answer 'cats' from the hits; quote dogs"

    assert catalog.match(spec) is None


def test_edited_param_or_extra_node_does_not_match(
    catalog, research, make_node, make_edge
) -> None:
    changed = fill_spec(research, "x")
    changed.nodes[0].params["tool_name"] = "create_follow_up_task"
    extra = fill_spec(research, "x")
    extra.nodes.append(make_node("leak", outputs=["z"]))
    extra.edges.append(make_edge("answer", "leak"))

    assert catalog.match(changed) is None
    assert catalog.match(extra) is None


def test_budget_and_metadata_do_not_affect_matching(catalog, summarize) -> None:
    spec = fill_spec(summarize, "text").model_copy(
        update={"budget": GraphBudget(max_nodes=2, max_llm_calls=1)}
    )
    spec.metadata.purpose = "anything"

    assert catalog.match(spec) is not None


def test_entries_carry_a_stable_content_hash(catalog, summarize) -> None:
    entry = catalog.get("summarize")

    assert entry.spec_hash == CatalogEntry("x", summarize).spec_hash
    assert catalog.by_hash(entry.spec_hash) is entry


def test_catalog_rejects_duplicates_and_empty(summarize) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SubgraphCatalog(
            entries=(CatalogEntry("a", summarize), CatalogEntry("a", summarize))
        )
    with pytest.raises(ValueError, match="at least one"):
        SubgraphCatalog(entries=())


def test_from_specs_keys_by_graph_id(summarize, research) -> None:
    assert SubgraphCatalog.from_specs([summarize, research]).ids == (
        "summarize-v1",
        "research-v1",
    )


def test_bench_fill_spec_is_the_shared_implementation(summarize) -> None:
    assert bench_fill_spec(summarize, "p") == fill_spec(summarize, "p")


# ---------- the catalog gate ----------


def test_gate_refuses_a_plan_outside_the_catalog(catalog) -> None:
    report = check_plan_policy(
        build_demo_spec(), ExecutionPolicy(approved_catalog=catalog)
    )

    assert not report.allowed
    assert {v.code for v in report.violations} == {"plan_not_in_catalog"}
    assert not report.repairable


def test_gate_records_the_matched_entry(catalog, research) -> None:
    report = check_plan_policy(
        fill_spec(research, "q"), ExecutionPolicy(approved_catalog=catalog)
    )

    assert report.allowed
    assert report.catalog_entry_id == "research"
    assert report.catalog_entry_hash == catalog.get("research").spec_hash


def test_catalog_is_part_of_the_policy_hash(catalog, summarize) -> None:
    a = check_plan_policy(build_demo_spec(), ExecutionPolicy(approved_catalog=catalog))
    b = check_plan_policy(
        build_demo_spec(),
        ExecutionPolicy(approved_catalog=SubgraphCatalog.from_specs([summarize])),
    )

    assert a.policy_hash != b.policy_hash


# ---------- governed routing: the planner selects, never synthesizes ----------


def test_keyword_selector_picks_best_overlap_and_falls_back_to_first(catalog) -> None:
    assert keyword_selector("please search and find X", catalog.entries) == "research"
    assert keyword_selector("hello", catalog.entries) == "summarize"


def test_catalog_planner_returns_a_filled_entry(catalog) -> None:
    spec = CatalogPlanner(catalog)("find the latest release")

    assert spec.graph_id == "research-v1"
    assert spec.nodes[0].params["args"]["query"] == "find the latest release"
    assert spec.metadata.purpose == "catalog:research"


def test_selector_returning_unknown_id_fails(summarize) -> None:
    catalog = SubgraphCatalog.from_specs([summarize], selector=lambda p, e: "made-up")

    with pytest.raises(CatalogSelectionError):
        CatalogPlanner(catalog)("x")


def test_llm_selector_is_constrained_to_catalog_ids(catalog) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    select = build_llm_catalog_selector(
        FakeListChatModel(responses=["research", "nope"])
    )

    assert select("anything", catalog.entries) == "research"
    with pytest.raises(CatalogSelectionError):
        select("anything", catalog.entries)


def test_engine_in_catalog_mode_routes_to_an_approved_entry(catalog) -> None:
    engine = DynamicSubgraphs(
        EngineConfig(planner="mock", policy=ExecutionPolicy(approved_catalog=catalog))
    )

    result = engine.run("summarize this memo")

    assert result.ok, result.errors
    assert result.plan is not None
    assert result.plan.graph_id == "summarize-v1"
    assert result.receipt is not None
    assert result.receipt.catalog_entry_id == "summarize"
    assert result.receipt.catalog_entry_hash == catalog.get("summarize").spec_hash


def test_engine_refuses_a_fixed_spec_outside_the_catalog(catalog) -> None:
    engine = DynamicSubgraphs(
        EngineConfig(planner="mock", policy=ExecutionPolicy(approved_catalog=catalog))
    )

    result = engine.run("x", fixed_spec=build_demo_spec())

    assert result.status == "policy_denied"
    assert result.plan_attempts == 1  # catalog misses are not repaired


def test_nested_child_plans_are_gated_by_the_catalog(
    catalog, make_node, make_edge
) -> None:
    from app.governance import PlanPolicyViolation
    from app.runtime import LangGraphExecutor
    from app.runtime.subgraph import make_child_launcher

    launch = make_child_launcher(
        planner=lambda goal: build_demo_spec(),  # not an approved entry
        executor=LangGraphExecutor(),
        policy=ExecutionPolicy(approved_catalog=catalog),
    )

    with pytest.raises(PlanPolicyViolation):
        launch("dig", run_id="p__sg_dig", graph_depth=2, parent_run_id="p", inputs={})
