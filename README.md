# Dynamic Subgraphs

A governed dynamic graph runtime: a stable `Supervisor` plans, validates, and runs
**transient** LangGraph workflows assembled from a bounded node registry and a
validated `GraphSpec`, recording every run under `runs/<run_id>/`. The planner emits
*plans*, never executable code; the compiler only instantiates registry-approved node
kinds; and LangGraph types stay behind the `compiler/` and `runtime/` boundaries. An
optional thin FastAPI layer exposes the supervisor over HTTP.

## Quickstart

```bash
# Install dependencies (creates .venv)
uv sync

# Run the offline mock demo (free, no tokens)
uv run python -m app.main "compare A and B"

# Run the HTTP API (boots in mock mode by default)
uv run python -m app.api
```

By default everything runs in **mock** mode — free and offline. Set `DS_PLANNER=openai`
(with `OPENAI_API_KEY`) to use the real planner and grounded tools.

## Documentation

- [`docs/api.md`](./docs/api.md) — the HTTP surface over the supervisor (endpoints, modes, auth, examples).
- [`docs/dynamic-graphs-canonical-design-v1.md`](./docs/dynamic-graphs-canonical-design-v1.md) — canonical project design and source of truth.
- [`docs/index.md`](./docs/index.md) — full documentation index.
- [`AGENTS.md`](./AGENTS.md) — agent-facing package map and MVP sequence.
