from __future__ import annotations

import pytest

from app.runtime import (
    MissingModelProviderCredential,
    ModelRef,
    ProviderRegistry,
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
