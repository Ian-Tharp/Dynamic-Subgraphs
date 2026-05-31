# Architecture

## Dependency direction

```text
api → supervisor → compiler, runtime, recording
                 → registry (read-only at compile)
models ← all layers (no upward imports)
```

- **`models`**: pure data contracts (`GraphSpec`, `DynamicRunState`, traces). No LangGraph.
- **`registry`**: node-kind definitions, param schemas, tool/subagent allowlists.
- **`compiler`**: validates specs against registry; builds transient LangGraph graphs.
- **`runtime`**: executes compiled graphs; wraps nodes with stable ids, timing, errors.
- **`recording`**: writes `runs/<run_id>/` artifacts (spec, trace, output, mermaid).
- **`supervisor`**: durable host workflow — the only graph that stays fixed in v1.
- **`api`**: HTTP surface; delegates to supervisor.

## Alignment with CORE framing

This repo is the **dynamic subgraph engine** slice of a broader Agentic OS:

| CORE concept | This repo |
|--------------|-----------|
| Orchestration | `supervisor/` |
| Dynamic composition | `compiler/` + `registry/` |
| Execution | `runtime/` |
| Ledger / provenance | `recording/` |
| Evals / AEGIS | later gates on registry side effects (not v1 day-1) |

ChatGPT’s “select predefined subgraph” MVP is **Phase 1** here: hardcoded `GraphSpec` before LLM planning.

## LangGraph usage

- **Supervisor**: one static `StateGraph` (receive → plan → validate → compile_and_run → record → respond).
- **Transient graphs**: compiled per run from `GraphSpec`, discarded after completion unless paused for resume.

## State

v1 uses a single envelope (`DynamicRunState`) for all transient graphs — no per-graph `TypedDict` codegen yet.
