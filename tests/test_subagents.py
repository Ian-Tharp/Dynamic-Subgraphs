"""End-to-end coverage for spawn_subagent: subagent factory, runner, default echo, e2e."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.registry import Registry, validate_graph_spec
from app.runtime import (
    DEFAULT_SUBAGENT_SYSTEM_PROMPTS,
    LangGraphExecutor,
    Subagent,
    make_llm_subagent,
    make_spawn_subagent_runner,
    run_spawn_subagent,
)
from app.runtime.state import make_initial_state


# ---------- fake chat model ----------


class FakeChatModel:
    def __init__(self, responses: Iterable[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list, /, **kwargs: Any) -> Any:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("FakeChatModel out of scripted responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


# ---------- make_llm_subagent ----------


def test_llm_subagent_sends_system_and_human_messages() -> None:
    model = FakeChatModel([AIMessage(content="ok")])

    subagent = make_llm_subagent(
        system_prompt="You are a critic.", chat_model=model
    )
    subagent("review the draft", context={})

    sent = model.calls[0]
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == "You are a critic."
    assert isinstance(sent[1], HumanMessage)
    assert "review the draft" in sent[1].content


def test_llm_subagent_renders_context_into_human_message() -> None:
    model = FakeChatModel([AIMessage(content="ok")])

    subagent = make_llm_subagent(
        system_prompt="role", chat_model=model
    )
    subagent(
        "analyze",
        context={"draft": "the candidate text", "facts": ["a", "b"]},
    )

    user_msg = model.calls[0][1].content
    assert "Context:" in user_msg
    assert "draft" in user_msg and "the candidate text" in user_msg
    assert "facts" in user_msg and '"a"' in user_msg  # JSON-rendered list


def test_llm_subagent_omits_context_block_when_context_empty() -> None:
    model = FakeChatModel([AIMessage(content="ok")])

    subagent = make_llm_subagent(system_prompt="role", chat_model=model)
    subagent("just answer", context={})

    user_msg = model.calls[0][1].content
    assert "Context:" not in user_msg


def test_llm_subagent_coerces_non_string_response_content() -> None:
    class _R:
        content = ["a", "b"]

    model = FakeChatModel([_R()])

    subagent = make_llm_subagent(system_prompt="role", chat_model=model)
    out = subagent("x", context={})

    assert isinstance(out, str)
    assert "a" in out and "b" in out


def test_llm_subagent_propagates_model_exceptions() -> None:
    model = FakeChatModel([RuntimeError("API down")])

    subagent = make_llm_subagent(system_prompt="role", chat_model=model)

    with pytest.raises(RuntimeError, match="API down"):
        subagent("x", context={})


# ---------- make_spawn_subagent_runner ----------


def test_spawn_subagent_runner_returns_result_for_known_agent() -> None:
    def echo_critic(task: str, *, context):
        del context
        return f"critic: {task}"

    runner = make_spawn_subagent_runner({"critic": echo_critic})

    out = runner(
        make_initial_state(),
        {"agent_name": "critic", "task": "review"},
    )

    assert out == {"result": "critic: review"}


def test_spawn_subagent_runner_passes_state_values_as_context() -> None:
    seen: dict[str, Any] = {}

    def capturing(task: str, *, context):
        seen["task"] = task
        seen["context"] = dict(context)
        return "ok"

    runner = make_spawn_subagent_runner({"critic": capturing})
    state = make_initial_state()
    state["values"]["draft"] = "v1"

    runner(state, {"agent_name": "critic", "task": "review"})

    assert seen["task"] == "review"
    assert seen["context"] == {"draft": "v1"}


def test_spawn_subagent_runner_raises_clear_error_for_unknown_agent() -> None:
    runner = make_spawn_subagent_runner({"critic": lambda t, *, context: "x"})

    with pytest.raises(RuntimeError, match="No subagent registered"):
        runner(
            make_initial_state(),
            {"agent_name": "ghost_agent", "task": "x"},
        )


# ---------- default echo runner ----------


def test_default_spawn_subagent_runner_produces_predictable_echo() -> None:
    out = run_spawn_subagent(
        make_initial_state(),
        {"agent_name": "critic", "task": "review the draft"},
    )

    assert out == {
        "result": "<subagent:critic>review the draft</subagent>"
    }


# ---------- DEFAULT_SUBAGENT_SYSTEM_PROMPTS ----------


def test_default_system_prompts_cover_default_allowlist() -> None:
    allowlist = Registry().subagents

    for name in allowlist:
        assert name in DEFAULT_SUBAGENT_SYSTEM_PROMPTS, (
            f"missing default system prompt for allowlisted subagent {name!r}"
        )


# ---------- end-to-end ----------


def test_end_to_end_spawn_subagent_executes_with_echo_runner() -> None:
    spec = GraphSpec(
        graph_id="subagent-e2e",
        goal="exercise spawn_subagent end-to-end with the default echo runner",
        nodes=[
            NodeSpec(
                id="seed",
                kind=NodeKind.LLM_CALL,
                outputs=["draft"],
                params={"instruction": "draft a proposal"},
            ),
            NodeSpec(
                id="review",
                kind=NodeKind.SPAWN_SUBAGENT,
                inputs=["draft"],
                outputs=["critique"],
                params={
                    "agent_name": "critic",
                    "task": "critique the upstream draft",
                },
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "seed"}),
            EdgeSpec.model_validate({"from": "seed", "to": "review"}),
            EdgeSpec.model_validate({"from": "review", "to": "END"}),
        ],
    )
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor()

    result = executor.execute(executor.compile(validated), run_id="sub-e2e")

    assert result.ok is True
    assert (
        result.state["values"]["critique"]
        == "<subagent:critic>critique the upstream draft</subagent>"
    )


def test_end_to_end_with_wired_subagent_routes_real_string() -> None:
    def specialist(task: str, *, context) -> str:
        return f"<specialist task={task} ctx_keys={sorted(context.keys())}>"

    spec = GraphSpec(
        graph_id="subagent-wired",
        goal="wire a custom subagent and run end-to-end",
        nodes=[
            NodeSpec(
                id="seed",
                kind=NodeKind.LLM_CALL,
                outputs=["topic"],
                params={"instruction": "pick a topic"},
            ),
            NodeSpec(
                id="analyze",
                kind=NodeKind.SPAWN_SUBAGENT,
                inputs=["topic"],
                outputs=["analysis"],
                params={
                    "agent_name": "document_specialist",
                    "task": "analyze the topic",
                },
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "seed"}),
            EdgeSpec.model_validate({"from": "seed", "to": "analyze"}),
            EdgeSpec.model_validate({"from": "analyze", "to": "END"}),
        ],
    )
    validated = validate_graph_spec(spec)
    executor = LangGraphExecutor(
        runners={
            NodeKind.SPAWN_SUBAGENT: make_spawn_subagent_runner(
                {"document_specialist": specialist}
            ),
        }
    )

    result = executor.execute(executor.compile(validated), run_id="sub-wired")

    assert result.ok is True
    analysis = result.state["values"]["analysis"]
    assert "<specialist task=analyze the topic" in analysis
    assert "topic" in analysis  # context keys leaked through
