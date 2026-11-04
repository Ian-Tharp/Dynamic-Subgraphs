# Documentation Index

## Canonical

- [`dynamic-graphs-canonical-design-v1.md`](./dynamic-graphs-canonical-design-v1.md) — source of truth for the project design.

## Using the SDK

- [`recipes.md`](./recipes.md) — task-oriented recipes (policy, plan-repair, recording, per-role models) with tested-model and latency tables.
- [`governance.md`](./governance.md) — plan-level policy gate, pre-flight cost bound, approved-subgraph catalog mode, and provenance receipts.
- [`api-stability.md`](./api-stability.md) — the public-API definition and the project's change/deprecation policy.
- [`api.md`](./api.md) — the optional HTTP surface over the supervisor.
- [`mcp.md`](./mcp.md) — the optional MCP server: governed runs as tools for MCP hosts.

## Evaluation

- [`evals/model-comparison-2026-06.md`](./evals/model-comparison-2026-06.md) — a gpt-5.4-nano vs claude-haiku-4-5 comparison on the same plans.

## How to read this repo

Start with the canonical design for the model behind everything, then `recipes.md`
for how to drive the SDK and `api-stability.md` for what is and isn't public.
