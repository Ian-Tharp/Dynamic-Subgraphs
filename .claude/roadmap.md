# Roadmap

## Shipped

| Phase | Status |
|---|---|
| Models + registry + validator | ✓ |
| Compiler (spec → StateGraph) | ✓ |
| Runtime: executor, runners, wrappers, state | ✓ |
| Recording (full artifacts per run) | ✓ |
| Supervisor (status taxonomy, conditional routing) | ✓ |
| LLM planner (ChatOpenAI + retries on validation) | ✓ |
| LLM runner for `llm_call` | ✓ |
| LLM-backed reduce (`llm_summarize`) | ✓ |
| `parallel_map` (compiler-native, `Send`, JSON-tolerant) | ✓ |
| `branch` (compiler-native, `add_conditional_edges`, validator-checked) | ✓ |
| `wait_for_event` (compiler-native, `interrupt()`, checkpointer, supervisor.resume) | ✓ |
| `spawn_subagent` (runner-handled, role-prompt LLM subagents, echo default) | ✓ |
| `emit_artifact` (runner-handled, `ArtifactSink` Protocol, `FileArtifactSink` wired) | ✓ |
| Echo-runner audit + dedup: `render_value_for_prompt`, `build_openai_chat`, sharpened docstrings | ✓ |
| `Supervisor.replay(run_id)` — load recorded spec, re-execute under new run id | ✓ |
| Iterative supervisor scaffolding (`run_iteratively`, `IterationDecider` Protocol, `StatusIterationDecider`, `build_replan_prompt`) | ✓ |
| Real `tool_call` runners (web_search via DuckDuckGo+Bing, policy_lookup, document_extract, create_follow_up_task) | ✓ partial |
| `strict_runners` flag — refuses to fall back to default echo runners at compile time | ✓ |
| `LlmIterationDecider` + `build_openai_iteration_decider` — closes the cognitive loop with structured replan/stop/ask/fail decisions | ✓ |
| `SearchProvider` Protocol + `TavilySearchProvider` + env-aware factory — production search adapter with DDG+Bing fallback | ✓ |
| Judge truncation fix — `LlmIterationDecider` now sees up to 4000 chars per value (was 500), unblocks real-LLM evaluation | ✓ |
| Chain-level recording (`FileRecorder.record_chain`, `load_chain`, `Supervisor.run_iteratively(record_chain=True)`) | ✓ |

## Identified improvements (alignment with original intent)

Things noticed during prior slices that we could ship to better match
the canonical design's "bounded, inspectable, replayable" thesis:

### Slot-key cleanup for `parallel_map`

After parallel_map finishes, the per-worker slot keys
(`<output_key>__0`, `<output_key>__1`, …) survive in `state.values`
alongside the assembled list at `<output_key>`. The `merge_dicts`
reducer can't remove keys. Possible fixes: a dedicated "delete keys"
reducer mechanic, or a follow-up cleanup node, or accepting the leak
since the canonical output_key is still present. Low priority but
worth flagging — the recorded `output.json` is noisier than needed.

### Wall-time + depth budget enforcement

`GraphBudget` declares `max_wall_seconds` and `max_depth` but the
runtime doesn't enforce them. `max_nodes` and `max_llm_calls` are
checked at validation. Wall-time would need an executor-side timer
(maybe via LangGraph's `recursion_limit` or a wrapping timeout).
`max_depth` only matters once nested dynamic subgraphs ship.

### Validator pre-checks runner output contract

A node could declare `outputs=["foo"]` but the runner returns
`{"bar": ...}`. Currently the wrapper catches this at runtime and
attaches an error. Could be caught at compile time by introspecting
runner shape (where possible) or by a stronger contract on `NodeRunner`.

### Per-run cost / token accounting

Each LLM call costs tokens. The recorder doesn't currently capture
spend per node. Adding it would unlock the cost-aware planner the
canonical design discusses ("price LLM-heavy plans"). Especially
relevant now that the meta-loop can iterate — a budget that rolls up
across iterations would cap runaway chains.

### Honest naming for "grounded" tools (partially addressed)

`build_grounded_tools` includes:
- `web_search`: NOW genuinely production-grade when TAVILY_API_KEY is set
  (Tavily adapter); otherwise falls back to DDG+Bing. ✓
- `policy_lookup`: self-reflection of runtime allowlists. Still misnamed
  as "lookup"; could be renamed → `runtime_policy`.
- `document_extract`: still string statistics, not real PDF/HTML
  extraction. The allowlist name `mock_document_extract` is honest;
  consider also renaming the function.
- `create_follow_up_task`: still a structured no-op (`created=False`).
  Docstring is honest; name slightly oversells.

The biggest gap (real search) is closed. The other three are smaller and
have honest docstrings; renaming the functions is low-priority polish.


### Planner doesn't know it's being replanned

`build_replan_prompt` returns text passed as the next `Supervisor.run(prompt=...)`.
The planner system prompt has no "this is iteration N of a chain" signal.
Currently works because the replan text contains everything the planner
needs, but it would be cleaner for the planner to receive structured
replan context rather than reading it out of the prompt body.

### `strict_runners` mode on the executor

The default runners are echoes. Forgetting to wire production
overrides means a graph with `llm_call` nodes runs against
`run_llm_call` (echo) — quiet, no error. A `strict_runners=True`
flag on `LangGraphExecutor` would refuse to fall back to defaults
for any kind the user hasn't explicitly wired, forcing production
deployments to be explicit. Useful guard for the "we're using real
LLMs now" phase the project is in.

## Next candidate slices (ordered by leverage)

### `emit_artifact` — side-effect boundary

`EmitArtifactParams` already exists. Need:

- Design decision: what does "emit" mean? File write? Message post?
  Pluggable adapter?
- Runner reads `params.content_key` from state, calls adapter.
- Mock adapter for tests; pluggable real adapter for production.
- Test the side-effect contract via mock-capture.

This is where the project starts touching the outside world. Get the
adapter shape right before adding multiple targets.

### All 8 registry kinds are executable — runtime is phase-1 complete

The runtime now supports synthesis of every shape the canonical design
envisioned for phase 1: arbitrary topology, parallel fan-out, conditional
routing, durable pause/resume, role-delegated reasoning, and side-effect
emission. The next slices live above the runtime, not inside it.

### Real `tool_call` runners — connect to the world

Replace `DEFAULT_FAKE_TOOLS` with real integrations (web_search,
policy_lookup, …). Per-tool work is small but the **shape** of the tool
registry is load-bearing:

- Where do credentials live?
- Sandboxing story?
- Per-tool rate limiting?
- Composability (a tool that wraps an MCP server)?

### API layer (`POST /runs`)

`Supervisor.run(prompt, run_id=...)` is already shaped to be a FastAPI
handler. Need:

- Pydantic request/response models (mostly aligned).
- Routes: `POST /runs`, `GET /runs/<id>`, `GET /runs/<id>/spec`, …
- Streaming for long runs.
- Auth + rate-limiting story.

Mostly packaging work; no new runtime capability.

### Eval gates — the maturity-curve foundation

Once N runs exist in `runs/`, the gates become possible:

- Cross-run scoring rubric.
- Diff outcomes across model versions to catch planner drift.
- Nightly automation.
- Surfaced regressions.

This is the work that unlocks the "system learns over time" claim.

## Slice ordering rationale

| Slice | Effort | Leverage | When to pick |
|---|---|---|---|
| `branch` | S | M | Finish core routing primitives; smooth scaling toward more complex graphs |
| `emit_artifact` | S | M | Need the outside world to *see* the workflow's output |
| `wait_for_event` | M | **High** | Biggest architectural shift — durable workflows become possible |
| `spawn_subagent` | M | M (design-heavy) | After deciding nested-orchestration semantics |
| Real `tool_call` | M | M | When the system needs real-world data |
| API layer | M | Packaging | When you want clients to call in |
| Eval gates | L | **High (long-term)** | After enough recorded runs to learn from |

## Don't drift

Things deliberately deferred. If you find yourself considering one,
re-read the canonical design's "Decision log" section:

- `python_eval` / shell / arbitrary network — v2+, after sandboxing.
- Runtime mutation of the registry — never.
- Per-graph custom state TypedDicts — generic envelope is v1.
- Distributed workers beyond one explicit background-job integration.
- Multiple LLM providers — adapters later; runtime API is the core.
