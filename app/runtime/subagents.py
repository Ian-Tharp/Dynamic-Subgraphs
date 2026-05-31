"""Subagent infrastructure for `spawn_subagent` nodes.

A subagent is any callable `(task, *, context) -> str` registered under a
name in the registry's allowlist. The `spawn_subagent` node looks the
subagent up by `agent_name`, hands it the planner-provided `task` and
the parent run's `state.values` as context, and routes the single-string
result into the node's first declared output key.

Patterns established here mirror the LLM-call runner pattern:

- `Subagent` is a Protocol so any concrete implementation works.
- `make_llm_subagent(...)` builds one backed by any `BaseChatModel`.
- `build_openai_subagents(...)` is the convenience factory that wires
  ChatOpenAI lazily — callers using mocks or alternate providers don't
  pay the langchain-openai import cost.
- `make_spawn_subagent_runner(subagents)` returns the `NodeRunner` the
  executor wires for `NodeKind.SPAWN_SUBAGENT`.

The default subagent registry (`document_specialist`, `critic`) matches
the registry's `DEFAULT_SUBAGENTS` allowlist. Override the system prompts
or the names by passing your own `system_prompts` to the factories.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.models import DynamicRunState
from app.runtime.state import render_value_for_prompt


class Subagent(Protocol):
    """Callable that takes a task plus context and returns a single string result."""

    def __call__(self, task: str, *, context: Mapping[str, Any]) -> str: ...


DEFAULT_SUBAGENT_SYSTEM_PROMPTS: dict[str, str] = {
    "document_specialist": (
        "You are a document analysis specialist embedded in a larger "
        "workflow. Given a task and any context produced by upstream "
        "nodes, produce structured findings — key claims, supporting "
        "evidence, ambiguities, and gaps. Keep responses concise and "
        "decision-ready. Respond with the analysis only — no preamble."
    ),
    "critic": (
        "You are a critical reviewer embedded in a larger workflow. "
        "Given a task and any context produced by upstream nodes, "
        "identify weaknesses, missing perspectives, and concrete "
        "improvements. Be constructive and specific. Respond with "
        "your critique only — no preamble."
    ),
}


def make_llm_subagent(
    *,
    system_prompt: str,
    chat_model: BaseChatModel,
) -> Subagent:
    """Build a subagent that delegates to a chat model with a fixed role prompt."""

    def _call(task: str, *, context: Mapping[str, Any]) -> str:
        user_parts: list[str] = [f"Task:\n{task}"]
        if context:
            rendered = "\n".join(
                f"- {key}: {render_value_for_prompt(value)}"
                for key, value in sorted(context.items())
            )
            user_parts.append(f"\nContext:\n{rendered}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(user_parts)),
        ]
        response = chat_model.invoke(messages)
        text = getattr(response, "content", None)
        if not isinstance(text, str):
            text = str(text)
        return text

    return _call


def make_spawn_subagent_runner(
    subagents: Mapping[str, Subagent],
) -> Callable[[DynamicRunState, Mapping[str, Any]], dict[str, Any]]:
    """Build the `NodeRunner` the executor wires for `NodeKind.SPAWN_SUBAGENT`."""

    registry: dict[str, Subagent] = dict(subagents)

    def _runner(
        state: DynamicRunState,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        agent_name = str(params["agent_name"])
        task = str(params["task"])

        if agent_name not in registry:
            raise RuntimeError(
                f"No subagent registered for '{agent_name}'. Available: "
                f"{sorted(registry.keys())}"
            )

        subagent = registry[agent_name]
        context = dict(state.get("values", {}) or {})
        result = subagent(task, context=context)
        return {"result": result}

    return _runner


def build_openai_subagents(
    model: str = "gpt-5.4-nano",
    *,
    system_prompts: Mapping[str, str] | None = None,
    temperature: float | None = None,
) -> dict[str, Subagent]:
    """Build the default subagent registry backed by ChatOpenAI.

    Requires `OPENAI_API_KEY`. The langchain-openai import is local so
    callers using mocks or other providers don't pay the import cost.
    """

    from app.runtime.chat_models import build_openai_chat

    prompts: Mapping[str, str] = (
        system_prompts
        if system_prompts is not None
        else DEFAULT_SUBAGENT_SYSTEM_PROMPTS
    )

    chat = build_openai_chat(model, temperature=temperature)

    return {
        name: make_llm_subagent(system_prompt=prompt, chat_model=chat)
        for name, prompt in prompts.items()
    }


def build_openai_spawn_subagent_runner(
    model: str = "gpt-5.4-nano",
    *,
    system_prompts: Mapping[str, str] | None = None,
    temperature: float | None = None,
) -> Callable[[DynamicRunState, Mapping[str, Any]], dict[str, Any]]:
    """One-shot factory: build OpenAI-backed subagents + wrap as NodeRunner."""
    subagents = build_openai_subagents(
        model=model,
        system_prompts=system_prompts,
        temperature=temperature,
    )
    return make_spawn_subagent_runner(subagents)


