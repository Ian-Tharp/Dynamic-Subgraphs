"""Fully-local run via LM Studio — no cloud, no API keys.

Prerequisite: LM Studio running at http://localhost:1234 with the model
loaded. Use a *capable* model for planning — small local models frequently
emit invalid plans (see docs/recipes.md "Tested models"). Here we use a larger
local model so the planner has a chance.

Run:
    uv run python examples/02_local_lmstudio.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

# Model.lmstudio() defaults base_url=http://localhost:1234/v1, api_key="lm-studio",
# and structured_method="json_schema" (local servers reject forced tool_choice).
engine = DynamicSubgraphs(EngineConfig(model=Model.lmstudio("google/gemma-3-27b")))
result = engine.run("Summarize the pros and cons of remote work in two bullet lists.")

print("ok:", result.ok, "| status:", result.status)
if not result.ok:
    print("errors:", result.errors)
    print(
        "(small/local models often fail planning — see examples/03 for the "
        "recommended hybrid pattern.)"
    )
for key, value in result.values.items():
    print(f"  {key}: {str(value)[:200]}")
