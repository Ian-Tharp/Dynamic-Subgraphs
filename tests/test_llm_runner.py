"""OpenAILlmRunner unit tests with a fake chat model — no real tokens."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.runtime import LlmReduceRunner, OpenAILlmRunner
from app.runtime.state import make_initial_state


class FakeChatModel:
    """Stand-in for BaseChatModel — yields scripted AIMessages."""

    def __init__(self, responses: Iterable[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list, /, **kwargs: Any) -> Any:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("FakeChatModel ran out of scripted responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        # AIMessage or any duck-typed response with .content is returned as-is.
        return nxt


# ---------- happy path ----------


def test_runner_returns_model_content_in_result() -> None:
    model = FakeChatModel([AIMessage(content="Real summary of A.")])
    runner = OpenAILlmRunner(model)

    output = runner(make_initial_state(), {"instruction": "Summarize source A."})

    assert output == {"result": "Real summary of A."}


def test_runner_sends_system_and_human_messages() -> None:
    model = FakeChatModel([AIMessage(content="x")])
    runner = OpenAILlmRunner(model)

    runner(make_initial_state(), {"instruction": "Do something."})

    sent = model.calls[0]
    assert isinstance(sent[0], SystemMessage)
    assert isinstance(sent[1], HumanMessage)
    assert "Do something." in sent[1].content


# ---------- upstream state injection ----------


def test_runner_injects_upstream_values_into_user_message() -> None:
    model = FakeChatModel([AIMessage(content="x")])
    runner = OpenAILlmRunner(model)
    state = make_initial_state()
    state["values"]["summary_a"] = "alpha summary"
    state["values"]["summary_b"] = "beta summary"

    runner(state, {"instruction": "Compare them."})

    user_msg_content = model.calls[0][1].content
    assert "Upstream state:" in user_msg_content
    assert "summary_a" in user_msg_content
    assert "alpha summary" in user_msg_content
    assert "summary_b" in user_msg_content
    assert "beta summary" in user_msg_content


def test_runner_omits_upstream_section_when_state_is_empty() -> None:
    model = FakeChatModel([AIMessage(content="x")])
    runner = OpenAILlmRunner(model)

    runner(make_initial_state(), {"instruction": "Just answer."})

    user_msg_content = model.calls[0][1].content
    assert "Upstream state" not in user_msg_content


def test_runner_renders_non_string_values_via_json() -> None:
    model = FakeChatModel([AIMessage(content="x")])
    runner = OpenAILlmRunner(model)
    state = make_initial_state()
    state["values"]["counts"] = {"a": 1, "b": 2}
    state["values"]["items"] = ["x", "y"]

    runner(state, {"instruction": "Reduce."})

    user_msg_content = model.calls[0][1].content
    assert '"a": 1' in user_msg_content
    assert '"x"' in user_msg_content


# ---------- defensive ----------


def test_runner_coerces_non_string_content_to_string() -> None:
    class _NonStringResponse:
        content = 42  # langchain types enforce str; some adapters won't.

    model = FakeChatModel([_NonStringResponse()])
    runner = OpenAILlmRunner(model)

    output = runner(make_initial_state(), {"instruction": "x"})

    assert output["result"] == "42"


def test_runner_propagates_model_exceptions() -> None:
    model = FakeChatModel([RuntimeError("upstream API down")])
    runner = OpenAILlmRunner(model)

    with pytest.raises(RuntimeError, match="upstream API down"):
        runner(make_initial_state(), {"instruction": "x"})


# ---------- LlmReduceRunner ----------


def test_llm_reduce_delegates_concat_to_deterministic_runner() -> None:
    # No model invocations expected for concat strategy.
    model = FakeChatModel([])
    runner = LlmReduceRunner(model)
    state = make_initial_state()
    state["values"]["a"] = "alpha"
    state["values"]["b"] = "beta"

    output = runner(
        state,
        {
            "strategy": "concat",
            "input_keys": ["a", "b"],
            "output_key": "joined",
        },
    )

    assert output == {"joined": "alpha\nbeta"}
    assert model.calls == []  # model was never invoked


def test_llm_reduce_delegates_merge_dict_to_deterministic_runner() -> None:
    model = FakeChatModel([])
    runner = LlmReduceRunner(model)
    state = make_initial_state()
    state["values"]["a"] = {"x": 1}
    state["values"]["b"] = {"y": 2}

    output = runner(
        state,
        {
            "strategy": "merge_dict",
            "input_keys": ["a", "b"],
            "output_key": "merged",
        },
    )

    assert output == {"merged": {"x": 1, "y": 2}}
    assert model.calls == []


def test_llm_reduce_summarize_routes_response_into_output_key() -> None:
    model = FakeChatModel([AIMessage(content="Synthesized recommendation.")])
    runner = LlmReduceRunner(model)
    state = make_initial_state()
    state["values"]["summary_a"] = "alpha"
    state["values"]["summary_b"] = "beta"

    output = runner(
        state,
        {
            "strategy": "llm_summarize",
            "input_keys": ["summary_a", "summary_b"],
            "output_key": "recommendation",
            "instruction": "Pick the stronger source.",
        },
    )

    assert output == {"recommendation": "Synthesized recommendation."}


def test_llm_reduce_summarize_sends_inputs_and_instruction_to_model() -> None:
    model = FakeChatModel([AIMessage(content="ok")])
    runner = LlmReduceRunner(model)
    state = make_initial_state()
    state["values"]["summary_a"] = "alpha value"
    state["values"]["summary_b"] = "beta value"

    runner(
        state,
        {
            "strategy": "llm_summarize",
            "input_keys": ["summary_a", "summary_b"],
            "output_key": "recommendation",
            "instruction": "Compare and pick one.",
        },
    )

    sent = model.calls[0]
    assert isinstance(sent[0], SystemMessage)
    user_msg = sent[1].content
    assert "Compare and pick one." in user_msg
    assert "summary_a" in user_msg and "alpha value" in user_msg
    assert "summary_b" in user_msg and "beta value" in user_msg


def test_llm_reduce_summarize_propagates_model_exceptions() -> None:
    model = FakeChatModel([RuntimeError("model down")])
    runner = LlmReduceRunner(model)
    state = make_initial_state()
    state["values"]["a"] = "x"

    with pytest.raises(RuntimeError, match="model down"):
        runner(
            state,
            {
                "strategy": "llm_summarize",
                "input_keys": ["a"],
                "output_key": "out",
                "instruction": "summarize",
            },
        )
