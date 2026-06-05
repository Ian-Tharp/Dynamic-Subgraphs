"""LLMPlanner coverage with a fake structured-output model.

These tests never touch a real model. They verify:
- happy path: a valid spec round-trips through the validator and is returned;
- the planner retries when the model emits an invalid spec, and the second
  attempt receives the validator's issues as a HumanMessage;
- the planner raises PlannerError after max_retries exhausted;
- the planner refuses non-GraphSpec returns from the model.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.supervisor.llm_planner import LLMPlanner, PlannerError

# ---------- fake model ----------


class FakeStructuredModel:
    """Mimics a `with_structured_output(GraphSpec)` runnable.

    Yields scripted return values from `responses` in order. Captures every
    invocation's messages so tests can assert the planner re-sent validator
    issues on retry.
    """

    def __init__(self, responses: Iterable[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list, /, **kwargs: Any) -> Any:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("FakeStructuredModel ran out of scripted responses")
        return self._responses.pop(0)


# ---------- planner guidance (prompt append) ----------


def test_guidance_is_appended_to_the_default_planner_prompt() -> None:
    # Arrange / Act
    planner = LLMPlanner(
        FakeStructuredModel([]),
        guidance="Prefer parallel_map over deep recursion.",
    )

    # Assert — guidance present, contract preserved (not replaced).
    assert "Additional planning guidance" in planner._system_prompt
    assert "Prefer parallel_map over deep recursion." in planner._system_prompt
    assert "Hard rules:" in planner._system_prompt


def test_no_guidance_leaves_the_default_prompt_unchanged() -> None:
    planner = LLMPlanner(FakeStructuredModel([]))
    assert "Additional planning guidance" not in planner._system_prompt


def test_guidance_appends_even_to_a_full_system_prompt_override() -> None:
    planner = LLMPlanner(
        FakeStructuredModel([]),
        system_prompt="CUSTOM CONTRACT",
        guidance="STEER",
    )
    assert planner._system_prompt.startswith("CUSTOM CONTRACT")
    assert "STEER" in planner._system_prompt


# ---------- spec helpers ----------


def _valid_spec() -> GraphSpec:
    return GraphSpec(
        graph_id="planner-spec",
        goal="single step demo",
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["draft"],
                params={"instruction": "do the thing"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
    )


def _invalid_spec_dangling_edge() -> GraphSpec:
    return GraphSpec(
        graph_id="planner-bad-spec",
        goal="will fail",
        nodes=[
            NodeSpec(
                id="step",
                kind=NodeKind.LLM_CALL,
                outputs=["draft"],
                params={"instruction": "x"},
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "step"}),
            EdgeSpec.model_validate({"from": "step", "to": "ghost"}),
            EdgeSpec.model_validate({"from": "step", "to": "END"}),
        ],
    )


# ---------- happy path ----------


def test_planner_returns_validated_spec_on_first_attempt() -> None:
    model = FakeStructuredModel([_valid_spec()])
    planner = LLMPlanner(model)

    spec = planner("compare two sources")

    assert spec.graph_id == "planner-spec"
    assert len(model.calls) == 1


def test_first_invocation_includes_system_and_human_messages() -> None:
    model = FakeStructuredModel([_valid_spec()])
    planner = LLMPlanner(model)

    planner("a prompt")

    first_call = model.calls[0]
    assert isinstance(first_call[0], SystemMessage)
    assert isinstance(first_call[1], HumanMessage)
    assert "a prompt" in first_call[1].content


# ---------- retry on invalid spec ----------


def test_planner_retries_and_recovers() -> None:
    model = FakeStructuredModel([_invalid_spec_dangling_edge(), _valid_spec()])
    planner = LLMPlanner(model, max_retries=2)

    spec = planner("prompt")

    assert spec.graph_id == "planner-spec"
    assert len(model.calls) == 2
    # The second call should include a repair message with the dangling_edge issue.
    repair_msg = model.calls[1][-1]
    assert isinstance(repair_msg, HumanMessage)
    assert "dangling_edge" in repair_msg.content


def test_planner_gives_up_after_max_retries() -> None:
    model = FakeStructuredModel(
        [
            _invalid_spec_dangling_edge(),
            _invalid_spec_dangling_edge(),
            _invalid_spec_dangling_edge(),
        ]
    )
    planner = LLMPlanner(model, max_retries=2)

    with pytest.raises(PlannerError) as exc:
        planner("prompt")

    assert "3 attempt" in str(exc.value)
    assert len(model.calls) == 3


# ---------- defensive ----------


def test_planner_rejects_non_graphspec_return() -> None:
    model = FakeStructuredModel(["not a spec"])  # type: ignore[list-item]
    planner = LLMPlanner(model)

    with pytest.raises(PlannerError, match="did not return a GraphSpec"):
        planner("prompt")


# ---------- prompt templating ----------


def test_default_reduce_strategies_advertise_only_deterministic_set() -> None:
    model = FakeStructuredModel([_valid_spec()])

    planner = LLMPlanner(model)

    # The strategy list interpolated into `one of [...]` should not include
    # llm_summarize when the planner is built with the default set.
    assert 'one of ["concat", "merge_dict"]' in planner._system_prompt


def test_widened_reduce_strategies_advertise_llm_summarize() -> None:
    model = FakeStructuredModel([_valid_spec()])

    planner = LLMPlanner(
        model,
        executable_reduce_strategies={"concat", "merge_dict", "llm_summarize"},
    )

    assert 'one of ["concat", "llm_summarize", "merge_dict"]' in planner._system_prompt
