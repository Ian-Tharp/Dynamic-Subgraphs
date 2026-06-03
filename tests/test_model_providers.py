from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from app.runtime import (
    AnthropicModelProvider,
    MissingModelProviderCredential,
    ModelRef,
    OllamaModelProvider,
    OpenAIModelProvider,
    ProviderRegistry,
    default_model_providers,
)


class _Provider:
    name = "FAKE"

    def required_env_vars(self) -> tuple[str, ...]:
        return ("FAKE_API_KEY",)

    def build_chat(self, ref: ModelRef):
        del ref
        raise NotImplementedError

    def build_structured_output(self, ref: ModelRef, schema: type):
        del ref, schema
        raise NotImplementedError


def test_model_ref_normalizes_provider() -> None:
    ref = ModelRef(provider=" Fake ", model=" model-1 ")

    assert ref.provider == "fake"
    assert ref.model == "model-1"
    assert ref.qualified_name == "fake:model-1"


def test_provider_registry_resolves_registered_provider() -> None:
    registry = ProviderRegistry()
    provider = _Provider()

    registry.register(provider)

    assert registry.get("fake") is provider


def test_provider_registry_reports_unknown_provider() -> None:
    registry = ProviderRegistry()

    with pytest.raises(KeyError, match="Unknown model provider"):
        registry.get("missing")


def test_provider_registry_requires_credentials() -> None:
    registry = ProviderRegistry()
    registry.register(_Provider())

    with pytest.raises(MissingModelProviderCredential, match="FAKE_API_KEY"):
        registry.require_credentials("fake", env={})

    registry.require_credentials("fake", env={"FAKE_API_KEY": "token"})


# ---------- built-in provider conformance ----------


class _Schema(BaseModel):
    answer: str


def test_default_registry_includes_all_builtin_providers() -> None:
    registry = default_model_providers()

    # All three resolve; lookup is case-insensitive.
    assert isinstance(registry.get("openai"), OpenAIModelProvider)
    assert isinstance(registry.get("ANTHROPIC"), AnthropicModelProvider)
    assert isinstance(registry.get("ollama"), OllamaModelProvider)


def test_builtin_provider_required_env_vars() -> None:
    assert OpenAIModelProvider().required_env_vars() == ("OPENAI_API_KEY",)
    assert AnthropicModelProvider().required_env_vars() == ("ANTHROPIC_API_KEY",)
    # Local provider needs no credentials.
    assert OllamaModelProvider().required_env_vars() == ()


def test_ollama_requires_no_credentials() -> None:
    registry = default_model_providers()
    # Empty env must still pass for the local provider.
    registry.require_credentials("ollama", env={})


def test_anthropic_requires_credentials() -> None:
    registry = default_model_providers()
    with pytest.raises(MissingModelProviderCredential, match="ANTHROPIC_API_KEY"):
        registry.require_credentials("anthropic", env={})
    registry.require_credentials("anthropic", env={"ANTHROPIC_API_KEY": "token"})


def test_openai_provider_builds_chat_and_structured_output(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIModelProvider()
    ref = ModelRef(provider="openai", model="gpt-5.4-nano", temperature=0.0)

    chat = provider.build_chat(ref)
    assert isinstance(chat, BaseChatModel)
    # Structured output is a runnable; construction must not hit the network.
    assert hasattr(provider.build_structured_output(ref, _Schema), "invoke")


def test_anthropic_provider_builds_chat_and_structured_output(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    provider = AnthropicModelProvider()
    ref = ModelRef(provider="anthropic", model="claude-haiku-4-5", temperature=0.0)

    chat = provider.build_chat(ref)
    assert isinstance(chat, BaseChatModel)
    assert hasattr(provider.build_structured_output(ref, _Schema), "invoke")


def test_ollama_provider_builds_chat_with_base_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    provider = OllamaModelProvider()
    ref = ModelRef(provider="ollama", model="llama3.1")

    chat = provider.build_chat(ref)
    assert isinstance(chat, BaseChatModel)
    assert hasattr(provider.build_structured_output(ref, _Schema), "invoke")
