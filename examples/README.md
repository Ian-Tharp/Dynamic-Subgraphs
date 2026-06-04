# Examples

Standalone, runnable SDK integration examples — one file per pattern. Each is
self-contained: **copy it into your own project and run it.** For prose
walk-throughs see [`../docs/recipes.md`](../docs/recipes.md). Prefer a notebook?
See [`quickstart.ipynb`](./quickstart.ipynb) (great in Google Colab — no install).

## Running

**Standalone** (after `pip install "dynamic-subgraphs[openai]"`): drop the file
in a folder, put your key in a `.env` beside it (or export it), and run it:

```bash
echo 'OPENAI_API_KEY=sk-...' > .env        # or: export OPENAI_API_KEY=sk-...
python 01_quickstart_cloud.py
```

**In this repo:**

```bash
uv run python examples/00_offline_mock.py
```

Each script calls `load_dotenv()`, which picks up a `.env` in the working
directory (or any parent) — so it works the same in-repo or standalone. If you'd
rather not use a file, just set the keys as environment variables.

## Prerequisites

| Need | For |
|------|-----|
| _nothing_ | `00_offline_mock.py`, `06_capabilities_for_agents.py` (mock planner) |
| `OPENAI_API_KEY` (env var or local `.env`) | `01`, `04`, `05`, and the planner in `03` |
| LM Studio running at `localhost:1234` (model loaded) | `02`, `03` |
| Your own OpenAI-compatible endpoint | `07` (template) |

## Index

| File | Shows |
|------|-------|
| `00_offline_mock.py` | Zero-config smoke test — no keys, no network |
| `01_quickstart_cloud.py` | Cloud one-liner: OpenAI planner + worker |
| `02_local_lmstudio.py` | Fully-local run via LM Studio |
| `03_hybrid_cloud_planner_local_worker.py` | **Recommended local pattern** — cloud planner + local worker |
| `04_visual_only_recording.py` | Record only `graph.mmd` (the graph diagram) |
| `05_debug_a_run.py` | Full recording + inspecting status/errors/plan/artifacts |
| `06_capabilities_for_agents.py` | `capabilities()` + `RunResult.to_dict()` for LLM/agent callers |
| `07_byo_openai_compatible_endpoint.py` | Point at any OpenAI-compatible server (template) |

## Model choice

The planner must emit a valid `GraphSpec`. Use a capable/cloud model for the
planner; small local models are fine as **workers**. See the tested-model and
latency tables in [`../docs/recipes.md`](../docs/recipes.md).
