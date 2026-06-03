"""Offline mock run — no API keys, no network, no tokens.

The `mock` planner uses a built-in demo GraphSpec and echo runners, so this
runs anywhere with zero configuration. Use it as a first smoke test that the
SDK imports and executes.

Run:
    uv run python examples/00_offline_mock.py
"""

from dynamic_subgraphs import DynamicSubgraphs, EngineConfig

engine = DynamicSubgraphs(EngineConfig(planner="mock"))
result = engine.run("compare two options and recommend one")

print("status:", result.status)
print("ok:", result.ok)
print("plan:", result.plan.graph_id if result.plan else None)
print("values:", result.values)
