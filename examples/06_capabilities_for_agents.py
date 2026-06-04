"""Introspection for LLMs / coding agents — discover options, get JSON results.

`DynamicSubgraphs.capabilities()` enumerates every valid provider, planner,
model role, artifact, status, and structured-output method, so an agent never
has to guess an option string. `RunResult.to_dict()` is a JSON-safe view of a
run (Path -> str, plan -> dict) suitable for logging or passing to another tool.

This example needs no API keys (uses the mock planner).

Run:
    uv run python examples/06_capabilities_for_agents.py
"""

import json

from dynamic_subgraphs import DynamicSubgraphs, EngineConfig

print("=== capabilities() ===")
print(json.dumps(DynamicSubgraphs.capabilities(), indent=2))

engine = DynamicSubgraphs(EngineConfig(planner="mock"))
result = engine.run("compare two options and recommend one")

print("\n=== RunResult.to_dict() (JSON-safe) ===")
print(json.dumps(result.to_dict(), indent=2)[:800])
