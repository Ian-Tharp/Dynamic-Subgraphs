"""Bounded node-kind registry — the capability vocabulary for GraphSpec."""

from app.registry.allowlists import DEFAULT_SUBAGENTS, DEFAULT_TOOLS, FORBIDDEN_KINDS
from app.registry.definitions import NodeKindDefinition, default_kind_definitions
from app.registry.errors import RegistryValidationError, RegistryValidationIssue
from app.registry.params import PARAM_MODEL_BY_KIND
from app.registry.registry import Registry, default_registry
from app.registry.validator import validate_graph_spec

__all__ = [
    "DEFAULT_SUBAGENTS",
    "DEFAULT_TOOLS",
    "FORBIDDEN_KINDS",
    "PARAM_MODEL_BY_KIND",
    "NodeKindDefinition",
    "Registry",
    "RegistryValidationError",
    "RegistryValidationIssue",
    "default_kind_definitions",
    "default_registry",
    "validate_graph_spec",
]
