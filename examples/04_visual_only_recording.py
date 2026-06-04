"""Record ONLY the Mermaid diagram of the generated graph.

`Recording.visual_only()` writes just `graph.mmd` under runs/<run_id>/ — handy
for embedding a picture of the plan in a PR or doc without the trace/spec noise.

Prerequisite: OPENAI_API_KEY in your environment (or a local `.env`).

Run:
    uv run python examples/04_visual_only_recording.py
"""

from dotenv import load_dotenv

# Loads a local .env if present; otherwise set your keys in the environment
# (e.g. OPENAI_API_KEY). Works the same in-repo or standalone.
load_dotenv()

from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model, Recording

engine = DynamicSubgraphs(
    EngineConfig(
        model=Model("openai", "gpt-5.4-nano"),
        recording=Recording.visual_only(),
        runs_dir="runs",
    )
)
result = engine.run(
    "Summarize the pros and cons of remote work in two short bullet lists.",
    run_id="example-visual",
)

print("ok:", result.ok, "| status:", result.status)
print("artifacts written:", sorted(result.artifacts))  # -> ['graph.mmd']
mermaid = result.artifacts.get("graph.mmd")
if mermaid:
    print("--- graph.mmd ---")
    print(mermaid.read_text(encoding="utf-8"))
elif not result.ok:
    print(
        "planner failed this run (non-deterministic) — re-run; see errors:",
        result.errors,
    )
