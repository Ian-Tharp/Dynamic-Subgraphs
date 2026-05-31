# Established patterns

Patterns that worked across multiple slices. New code should follow them.

## 1. Dependency injection + lazy factory imports

When adding an external integration:

1. The class accepts the **abstract interface** in its constructor
   (`BaseChatModel`, a `Runnable`, etc.).
2. A **factory function** constructs the concrete dependency and wires it.
3. The factory's heavy import (e.g., `langchain_openai`) is **local to the
   factory body**, not at module level.

Examples: `OpenAILlmRunner`, `LLMPlanner`, `LlmReduceRunner` all take
`BaseChatModel` in constructor; `build_openai_*` factories import
`ChatOpenAI` locally.

Why: callers using mocks or alternate providers don't pay the optional-dep
cost. Tests construct fakes directly without any provider import.

## 2. Planner introspects what runtime can execute

`LLMPlanner`'s system prompt is templated from the actual runtime state:

```python
executable_kinds = default_runners().keys() | COMPILER_HANDLED_KINDS
executable_reduce_strategies = injected per main.py wiring
tools = registry.tools
subagents = registry.subagents
```

When you add a new executable kind / reduce strategy / allowlisted tool,
the planner automatically advertises it. **Do not hardcode capability
lists in the prompt** — template them.

## 3. Node kinds: runner-handled vs compiler-handled

Two paths to "executable":

| Path | Mechanism | Use when | Examples |
|---|---|---|---|
| Runner-handled | `NodeRunner = (state, params) → dict`, registered in `default_runners()`, wrapped by `make_node_wrapper` | Kind has clean `(state, params) → result` semantics | `llm_call`, `tool_call`, `reduce` |
| Compiler-handled | Kind in `COMPILER_HANDLED_KINDS`; compiler emits multiple LangGraph nodes per `NodeSpec` | Kind needs `Send` fan-out, special edge wiring, or multiple internal nodes | `parallel_map` (dispatcher/worker/join) |

Both contribute to the union the planner sees. Pick the simpler path
unless you need multi-node expansion.

## 4. State envelope: explicit reducers

`DynamicRunState` is a `TypedDict` with `Annotated` reducer channels:

| Key | Reducer | Why |
|---|---|---|
| `values` | `merge_dicts` | Top-level dict merge; last-write-wins **per key** |
| `artifacts` | `merge_dicts` | Same |
| `metadata` | `merge_dicts` | Same |
| `errors` | `operator.add` | Append-only |
| `events` | `operator.add` | Append-only (trace) |

**Implication**: multiple parallel writers to the same `values` key
overwrite each other. To collect N parallel results, use **distinct keys**
(see `parallel_map`'s `<output_key>__<idx>` slot pattern) or a different
reducer.

## 5. Supervisor status taxonomy

Every known failure mode has a status string. New failure stages:

1. Add to the taxonomy doc on `SupervisorState`.
2. Catch the **specific** exception in the relevant supervisor node.
3. Return `{"status": "<x>_failed", "errors": [structured entry]}`.
4. Route via `add_conditional_edges` to short-circuit if needed.
5. Add a branch in `respond`'s response-message switch.

**Don't crash; classify**. Tests assert on status strings, not stack traces.

## 6. Validator is the trust boundary

Everything downstream of `validate_graph_spec` assumes well-formed input.
Compiler doesn't re-validate topology. Runners don't re-validate params.
Recorder writes whatever it's given.

If something needs checking, **add it to the validator**, not downstream.

## 7. Recording: failed runs are first class

The recorder writes a full set of artifacts for *every* run including
failures. The supervisor catches recording exceptions and emits status
`record_failed` instead of crashing.

Never let recording errors kill the request.

## 8. Per-kind output mapping convention

Runners return `{"result": value}` by default; the wrapper maps it to the
node's `outputs[0]`. If a runner returns named outputs that exactly match
`outputs[*]`, they're routed directly. Mismatch → wrapper raises and is
caught by the wrapper itself as an `errors` entry.

## 9. Surgical fix loop when LLM smoke fails

When `--llm` produces a failure status, each iteration should move the
failure **further down the pipeline**, never sideways:

1. Read the **stage** (`plan_failed`, `validation_failed`, …) and the
   issue code or error message.
2. Make **one** surgical change targeting that exact failure.
3. Re-run `pytest -W error` (token-free) — must remain green.
4. Re-run `--llm` once.
5. Confirm the failure moved down the pipeline; repeat.

This works because the supervisor's status taxonomy named every failure
specifically. Don't bypass the taxonomy.
