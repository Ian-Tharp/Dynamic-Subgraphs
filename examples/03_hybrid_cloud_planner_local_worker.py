"""Hybrid: cloud planner + local worker — the recommended way to use local models.

A capable cloud model plans the graph (reliable structured output); a local
model does the actual node work (private / cheap). Unset roles fall back to
the worker, so set planner_model explicitly.

Prerequisites: OPENAI_API_KEY in `.env`, and LM Studio running with the worker
model loaded.

Run:
    uv run python examples/03_hybrid_cloud_planner_local_worker.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

engine = DynamicSubgraphs(
    EngineConfig(
        planner_model=Model("openai", "gpt-5.4-nano"),  # capable planner
        worker_model=Model.lmstudio("openai/gpt-oss-20b"),  # local worker
    )
)

result = engine.run("Compare two sources on coffee's health effects and recommend one.")
print("ok:", result.ok, "| status:", result.status)
for key, value in result.values.items():
    print(f"  {key}: {str(value)[:200]}")

# Per-run override: each run can pick its own models for that call only, e.g.
#   engine.run("...", worker_model=Model.ollama("llama3.1"))
