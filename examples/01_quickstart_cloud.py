"""Cloud quickstart — OpenAI as planner + worker.

Prerequisite: OPENAI_API_KEY in your environment (or a local `.env`).

Run:
    uv run python examples/01_quickstart_cloud.py
"""

from dotenv import load_dotenv

# Loads a local .env if present; otherwise set your keys in the environment
# (e.g. OPENAI_API_KEY). Works the same in-repo or standalone.
load_dotenv()

from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

engine = DynamicSubgraphs(
    EngineConfig(
        model=Model("openai", "gpt-5.4-nano"),
        # Optional: enable result.cost. Prices per 1M tokens — verify against
        # your account / the provider's current pricing.
        pricing={"gpt-5.4-nano": {"input_per_1m": 0.20, "output_per_1m": 1.25}},
    )
)
result = engine.run(
    "Compare solar and wind power for a small town and recommend one, briefly."
)

print("ok:", result.ok, "| status:", result.status)
print("response:", result.response)
print("plan:", result.plan.graph_id if result.plan else None)
print("usage:", result.usage.total_tokens, "tokens | cost:", result.cost)
for key, value in result.values.items():
    print(f"  {key}: {str(value)[:200]}")
