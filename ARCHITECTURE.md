# Architecture

## Dependency direction

```text
api → supervisor → compiler, runtime, recording
                 → registry + policy (validate against the host envelope)
eval (public) → models, registry        (scores a completed run; off by default)
models ← all layers (no upward imports)
```

- **`models`**: pure data contracts (`GraphSpec`, `DynamicRunState`, traces). No LangGraph.
- **`policy`**: host-owned execution governance (`ExecutionPolicy`) — budgets plus
  tool/subagent/node-kind allow-sets. Enforced at validation and runtime as the
  intersection `min(host, planner request)` (and against each nested child's
  *remaining* allowance), so a planner can never grant itself more than the host
  allows. The effective envelope is stamped onto the validated spec.
- **`registry`**: node-kind definitions, param schemas, tool/subagent allowlists.
- **`compiler`**: validates specs against registry + policy; builds transient LangGraph graphs.
- **`runtime`**: executes compiled graphs; wraps nodes with stable ids, timing, errors.
- **`recording`**: writes `runs/<run_id>/` artifacts (spec, trace, output, mermaid).
- **`supervisor`**: durable host workflow — the only graph that stays fixed in v1.
- **`eval`** (public, `dynamic_subgraphs.eval`): deterministic, token-free structural
  scorer (`DeterministicEvalGate`) over a completed run. Off by default; re-validates
  the plan against the registry and grades plan validity, grounding, goal completion,
  and cost into a comparable `EvalResult`.
- **`api`**: HTTP surface; delegates to supervisor.

## Alignment with CORE framing

This repo is the **dynamic subgraph engine** slice of a broader Agentic OS:

| CORE concept | This repo |
|--------------|-----------|
| Orchestration | `supervisor/` |
| Dynamic composition | `compiler/` + `registry/` |
| Execution | `runtime/` |
| Governance | `policy/` (host-owned `ExecutionPolicy`, enforced at validation + runtime) |
| Ledger / provenance | `recording/` |
| Evals / AEGIS | `dynamic_subgraphs/eval/` — deterministic *structural* scorer shipped (off by default); semantic gates on registry side effects remain future work |

ChatGPT’s “select predefined subgraph” MVP is **Phase 1** here: hardcoded `GraphSpec` before LLM planning.

## LangGraph usage

- **Supervisor**: one static `StateGraph` (receive → plan → validate → compile_and_run → record → respond).
- **Transient graphs**: compiled per run from `GraphSpec`, discarded after completion unless paused for resume.

## State

v1 uses a single envelope (`DynamicRunState`) for all transient graphs — no per-graph `TypedDict` codegen yet.
