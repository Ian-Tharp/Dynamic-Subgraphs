# Dynamic Subgraphs — agent map

This repo implements a **governed dynamic graph runtime**: a stable supervisor orchestrates **transient** LangGraph workflows built from a bounded node registry, validated `GraphSpec`, and full run recording.

**Not in scope:** generic chatbot, unconstrained tool access, arbitrary code execution in v1.

## Read first

| Doc | When |
|-----|------|
| `docs/dynamic-graphs-canonical-design-v1.md` | Architecture, GraphSpec, registry, phases |
| `ARCHITECTURE.md` | Package boundaries and dependency direction |
| `docs/exec-plans/active/` | Current implementation task (if present) |

## Package map

| Package | Role |
|---------|------|
| `app/models/` | GraphSpec, run state, trace types |
| `app/registry/` | Allowed node kinds and param schemas |
| `app/compiler/` | GraphSpec → LangGraph compile |
| `app/runtime/` | Execute compiled graphs; node wrappers |
| `app/supervisor/` | Durable host graph: plan → validate → run → record |
| `app/recording/` | Persist runs under `runs/<run_id>/` |
| `app/api/` | FastAPI: thin HTTP surface over the supervisor (runs/chains/registry/health, SSE, resume/replay). See `docs/api.md`. |

## Rules

1. Planner emits **plans** (`GraphSpec`), not executable code.
2. Compiler only instantiates **registry-approved** node kinds.
3. Every run produces a **trace**; failed runs are recorded too.
4. Wrap LangGraph behind interfaces in `compiler/` and `runtime/` — do not leak LangGraph types across package boundaries.
5. MCP and multi-provider models are **adapters later**; runtime API is the core.
6. Smallest testable slice first (hardcoded spec → compile → run → record).

## MVP sequence

1. Models + registry skeleton  
2. Compiler + runtime (hardcoded spec)  
3. Recording + replay  
4. Supervisor graph  
5. Planner (structured LLM output)  
6. API `POST /runs` ✅ (full supervisor surface; see `docs/api.md`)  
7. Eval / policy gates on side effects  
