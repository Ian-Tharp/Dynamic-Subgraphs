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
    """A concrete model choice for one runtime role."""

    provider: str
    model: str
    temperature: float | None = None
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

        kwargs: dict[str, Any] = {"model": ref.model, **dict(ref.extra_kwargs)}
        if ref.temperature is not None:
            kwargs["temperature"] = ref.temperature
        return ChatOpenAI(**kwargs)

    def build_structured_output(self, ref: ModelRef, schema: type) -> Any:
        # Function-calling handles GraphSpec's open ``params`` dict more
        # reliably than strict JSON-schema mode.
        return self.build_chat(ref).with_structured_output(
            schema,
            method="function_calling",
        )


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
                f"provider={provider.name!r} requires env var(s): "
                f"{', '.join(missing)}"
            )


def default_model_providers() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(OpenAIModelProvider())
    return registry
