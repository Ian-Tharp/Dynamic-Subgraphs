"""Provider-neutral chat model construction.

The runtime consumes chat models through LangChain's ``BaseChatModel`` and
structured-output runnables. This module keeps vendor-specific construction and
credential checks in one place so planner/runner wiring can select providers
without hard-coding OpenAI throughout the SDK surface.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class ModelRef:
    """A concrete model choice for one runtime role.

    `base_url` and `api_key` make bring-your-own-endpoint / BYO-key first
    class (e.g. a local LM Studio or Ollama server, or a proxy) without
    burying them in `extra_kwargs`. `structured_method` selects how the
    provider coerces structured output — local OpenAI-compatible servers
    often reject the forced `tool_choice` of ``"function_calling"`` and need
    ``"json_schema"`` instead.
    """

    provider: str
    model: str
    temperature: float | None = None
    base_url: str | None = None
    api_key: str | None = None
    structured_method: str | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        model = self.model.strip()
        if not provider:
            raise ValueError("ModelRef.provider must be non-empty")
        if not model:
            raise ValueError("ModelRef.model must be non-empty")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)

    @property
    def qualified_name(self) -> str:
        return f"{self.provider}:{self.model}"

    @classmethod
    def openai_compatible(
        cls,
        model: str,
        *,
        base_url: str,
        api_key: str = "not-needed",
        structured_method: str = "json_schema",
        temperature: float | None = None,
        **extra_kwargs: Any,
    ) -> ModelRef:
        """Target any OpenAI-compatible server (local or proxied).

        Defaults to ``structured_method="json_schema"`` because local servers
        commonly reject the forced ``tool_choice`` of ``"function_calling"``.
        """

        return cls(
            provider="openai",
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            structured_method=structured_method,
            extra_kwargs=extra_kwargs,
        )

    @classmethod
    def lmstudio(
        cls,
        model: str,
        *,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        structured_method: str = "json_schema",
        temperature: float | None = None,
        **extra_kwargs: Any,
    ) -> ModelRef:
        """Target a local LM Studio server (OpenAI-compatible).

        Note: small local models often make poor *planners* — they emit
        invalid GraphSpecs and fail planning. Prefer them as the worker model
        behind a capable planner (see docs/recipes.md).
        """

        return cls.openai_compatible(
            model,
            base_url=base_url,
            api_key=api_key,
            structured_method=structured_method,
            temperature=temperature,
            **extra_kwargs,
        )

    @classmethod
    def ollama(
        cls,
        model: str,
        *,
        base_url: str | None = None,
        structured_method: str | None = None,
        temperature: float | None = None,
        **extra_kwargs: Any,
    ) -> ModelRef:
        """Target a local Ollama server (no API key).

        Note: small local models often make poor *planners* — prefer them as
        the worker model behind a capable planner (see docs/recipes.md).
        """

        return cls(
            provider="ollama",
            model=model,
            temperature=temperature,
            base_url=base_url,
            structured_method=structured_method,
            extra_kwargs=extra_kwargs,
        )

    def client_kwargs(self) -> dict[str, Any]:
        """Chat-client kwargs common to LangChain providers.

        Merges `extra_kwargs` first so first-class fields win on conflict.
        """

        kwargs: dict[str, Any] = dict(self.extra_kwargs)
        kwargs["model"] = self.model
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        return kwargs


class ModelProvider(Protocol):
    """Build provider-specific model clients behind a stable runtime contract."""

    name: str

    def required_env_vars(self) -> tuple[str, ...]: ...

    def build_chat(self, ref: ModelRef) -> BaseChatModel: ...

    def build_structured_output(self, ref: ModelRef, schema: type) -> Any: ...


class MissingModelProviderCredential(RuntimeError):
    """Raised when a selected provider is registered but not configured."""


class OpenAIModelProvider:
    """OpenAI provider adapter.

    Imports ``langchain_openai`` lazily so mock/local execution and alternate
    providers do not pay that dependency at import time.
    """

    name = "openai"

    def required_env_vars(self) -> tuple[str, ...]:
        return ("OPENAI_API_KEY",)

    def build_chat(self, ref: ModelRef) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**ref.client_kwargs())

    def build_structured_output(self, ref: ModelRef, schema: type) -> Any:
        # Function-calling handles GraphSpec's open ``params`` dict more
        # reliably than strict JSON-schema mode — but local OpenAI-compatible
        # servers (LM Studio, vLLM) reject forced tool_choice, so callers can
        # opt into "json_schema" via ModelRef.structured_method.
        method = ref.structured_method or "function_calling"
        return self.build_chat(ref).with_structured_output(schema, method=method)


class AnthropicModelProvider:
    """Anthropic (Claude) provider adapter.

    Imports ``langchain_anthropic`` lazily so the dependency is only paid
    when an Anthropic-backed role is actually built. Structured output uses
    Anthropic's native tool-calling, which handles GraphSpec's open
    ``params`` dict well.
    """

    name = "anthropic"

    def required_env_vars(self) -> tuple[str, ...]:
        return ("ANTHROPIC_API_KEY",)

    def build_chat(self, ref: ModelRef) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**ref.client_kwargs())

    def build_structured_output(self, ref: ModelRef, schema: type) -> Any:
        # Anthropic's default structured-output path is tool-calling, which
        # mirrors the OpenAI function_calling choice for the open params dict.
        chat = self.build_chat(ref)
        if ref.structured_method is not None:
            return chat.with_structured_output(schema, method=ref.structured_method)
        return chat.with_structured_output(schema)


class OllamaModelProvider:
    """Local Ollama provider adapter.

    Requires no API key — talks to a local Ollama server (default
    ``http://localhost:11434``, overridable via ``OLLAMA_BASE_URL`` or a
    ``base_url`` in ``ModelRef.extra_kwargs``). Structured-output reliability
    depends on the local model's tool/JSON support.
    """

    name = "ollama"

    def required_env_vars(self) -> tuple[str, ...]:
        return ()

    def _kwargs(self, ref: ModelRef) -> dict[str, Any]:
        kwargs = ref.client_kwargs()
        # Ollama has no API key; drop it if a caller set one generically.
        kwargs.pop("api_key", None)
        if "base_url" not in kwargs:
            base_url = os.environ.get("OLLAMA_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
        return kwargs

    def build_chat(self, ref: ModelRef) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(**self._kwargs(ref))

    def build_structured_output(self, ref: ModelRef, schema: type) -> Any:
        chat = self.build_chat(ref)
        if ref.structured_method is not None:
            return chat.with_structured_output(schema, method=ref.structured_method)
        return chat.with_structured_output(schema)


class ProviderRegistry:
    """Mutable provider registry used by SDK/API assembly.

    CORE can register its own provider adapter at process startup while the
    default registry keeps the existing OpenAI-backed path available.
    """

    def __init__(self, providers: Mapping[str, ModelProvider] | None = None) -> None:
        self._providers: dict[str, ModelProvider] = {}
        for provider in (providers or {}).values():
            self.register(provider)

    def register(self, provider: ModelProvider) -> None:
        name = provider.name.strip().lower()
        if not name:
            raise ValueError("ModelProvider.name must be non-empty")
        self._providers[name] = provider

    def names(self) -> tuple[str, ...]:
        """Sorted names of every registered provider (for introspection)."""
        return tuple(sorted(self._providers))

    def get(self, name: str) -> ModelProvider:
        key = name.strip().lower()
        try:
            return self._providers[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise KeyError(
                f"Unknown model provider {name!r}. Available: {available}"
            ) from exc

    def require_credentials(
        self,
        name: str,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        provider = self.get(name)
        source = os.environ if env is None else env
        missing = [key for key in provider.required_env_vars() if not source.get(key)]
        if missing:
            raise MissingModelProviderCredential(
                f"provider={provider.name!r} requires env var(s): {', '.join(missing)}"
            )


def default_model_providers() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(OpenAIModelProvider())
    registry.register(AnthropicModelProvider())
    registry.register(OllamaModelProvider())
    return registry
