"""Lazy chat-model factories — keep optional provider imports off the hot path.

This module is the single place provider-specific construction lives. The
runtime/planner/subagent modules import the factory and stay
provider-agnostic; they don't import `langchain_openai` directly. When we
add other providers (Anthropic, Ollama, …), they land here as new
factories with the same shape.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel


def build_openai_chat(
    model: str,
    *,
    temperature: float | None = None,
    **extra_kwargs: Any,
) -> BaseChatModel:
    """Construct a ChatOpenAI with our project conventions.

    Requires `OPENAI_API_KEY` in the environment. The `langchain_openai`
    import is local so callers that use mocks or alternate providers
    don't pay the dependency cost.
    """

    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {"model": model, **extra_kwargs}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)
