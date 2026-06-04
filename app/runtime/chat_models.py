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

from app.runtime.model_providers import ModelRef, OpenAIModelProvider


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

    return OpenAIModelProvider().build_chat(
        ModelRef(
            provider="openai",
            model=model,
            temperature=temperature,
            extra_kwargs=extra_kwargs,
        )
    )
