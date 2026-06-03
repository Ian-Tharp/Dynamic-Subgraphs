"""Debug a run with full recording, then inspect the result programmatically.

`Recording.debug()` captures every artifact (spec/trace/output/mermaid/summary/
prompt + emitted). Failures never raise — branch on `result.ok`/`result.status`
and read `result.errors` and the generated `result.plan`.

Prerequisite: OPENAI_API_KEY in `.env`.

Run:
    uv run python examples/05_debug_a_run.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model, Recording

engine = DynamicSubgraphs(
    EngineConfig(
        model=Model("openai", "gpt-5.4-nano"),
        recording=Recording.debug(),
        runs_dir="runs",
    )
)
result = engine.run(
    "Investigate the trade-offs of a 4-day work week.", run_id="example-debug"
)

print("status:", result.status, "| ok:", result.ok)
print("plan:", result.plan.graph_id if result.plan else None)
print("artifacts:", sorted(result.artifacts))
if not result.ok:
    for err in result.errors:
        print("  error:", err)
elif "summary.md" in result.artifacts:
    print("--- summary.md (first lines) ---")
    print(
        "\n".join(
            result.artifacts["summary.md"].read_text(encoding="utf-8").splitlines()[:8]
        )
    )
