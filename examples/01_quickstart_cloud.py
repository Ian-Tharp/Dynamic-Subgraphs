"""Cloud quickstart — OpenAI as planner + worker.

Prerequisite: OPENAI_API_KEY in the repo `.env`.

Run:
    uv run python examples/01_quickstart_cloud.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

engine = DynamicSubgraphs(EngineConfig(model=Model("openai", "gpt-5.4-nano")))
result = engine.run(
    "Compare two evidence sources on whether remote work boosts productivity, "
    "and recommend one."
)

print("ok:", result.ok, "| status:", result.status)
print("response:", result.response)
print("plan:", result.plan.graph_id if result.plan else None)
for key, value in result.values.items():
    print(f"  {key}: {str(value)[:200]}")
