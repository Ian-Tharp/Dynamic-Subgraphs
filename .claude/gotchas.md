# Gotchas

Things that bit us. Specific, reproducible, with the fix.

## LangGraph

### Static edges + `Command(goto=...)` don't override — they add

If a node returns `Command(goto="x")`, the **static `add_edge` outgoing
edges still fire in parallel**. Result: phantom downstream execution that
sees half-populated state.

Where this matters:
- The **supervisor** uses `add_conditional_edges` for failure routing so
  `goto="respond"` is exclusive.
- `parallel_map`'s **join** checks `state["errors"]` itself and halts
  with `Command(goto=END)` because workers' own goto=END doesn't stop the
  worker→join edge from firing across Send branches.

Rule: if you want routing to be exclusive, use `add_conditional_edges`.

### `web_search` provider is environment-dependent

`build_default_search_provider()` (used by `build_grounded_tools`)
returns:

- `TavilySearchProvider` when `TAVILY_API_KEY` is in the environment.
  This is the production path: LLM-agent-focused search, structured
  snippets with relevance scores, synthesized `answer` field. Free tier
  ~1000 searches/month at https://tavily.com.
- `DuckDuckGoSearchProvider` otherwise. DDG's instant-answer endpoint
  returns mostly definitional content; when that yields nothing, it
  falls back to scraping Bing HTML — fragile, possibly TOS-violating,
  low quality. Acceptable for development and demos.

Override explicitly with `build_default_search_provider(prefer_tavily=False)`
to force DDG even when a key is present, or pass `tavily_api_key=...`
to supply one without setting the env var.

The output shape is uniform across providers — downstream LLM nodes
consume `{tool, provider, query, answer, results: [{title, url, snippet, score?}]}`
regardless of which backend ran. Tests that pin a specific provider
must `monkeypatch.setenv` or `monkeypatch.delenv` for `TAVILY_API_KEY`
to control selection.

### Chain recording layout: chain dir is a sibling of iteration dirs

`Supervisor.run_iteratively("...", run_id="X")` produces:

    runs/
      X/                          ← chain metadata (chain.json, chain.md)
      X_iter_1/                   ← per-iteration GraphSpec/trace/output/etc.
      X_iter_2/
      ...

The chain dir and iteration dirs are siblings at the same level (flat
layout, not nested). To inspect a chain, read `runs/<chain_id>/chain.json`
or call `recorder.load_chain(chain_id)`. The per-iteration directories
are normal recorded runs and can be inspected or replayed independently.

If `chain_id` collides with an existing run_id (i.e., you ran
`sup.run(prompt, run_id="X")` and then `sup.run_iteratively(prompt, run_id="X")`),
the chain recording will overwrite the prior single-run's directory.
Pick a different `run_id` for chains or use `record_chain=False`.

### LlmIterationDecider truncation: 4000 char per value, not 500

The judge sees a "Outputs produced (state.values):" section in its eval
prompt with each value truncated at `value_render_limit` chars
(default 4000). Real LLM outputs are routinely 1-4k chars, so the
default 500 we shipped initially was way too tight — the judge would
respond "I can't verify, the output looks truncated" to every prompt
with non-trivial output. The system prompt now explicitly tells the
judge that truncation is display-only and not to penalize it.

If you see "I can't verify" / "appears truncated" in judge gaps, bump
`value_render_limit` further on the decider construction.

### LlmIterationDecider defers obvious cases to the fallback decider

The LLM judge does NOT evaluate every iteration. It defers to the
`fallback` decider (default `StatusIterationDecider`) for:

- `paused` (framework will ask the user anyway)
- `plan_failed` / `validation_failed` / `compile_failed` (no output to judge)
- `record_failed` / `resume_failed` / `replay_failed` (infrastructure issues)
- `execution_failed` (unless `judge_failed_runs=True`)

The point: don't spend tokens on decisions the framework's status
taxonomy already settled. The LLM only runs on `ok` runs (and
optionally `execution_failed`). Test invocation counts (`model.calls`)
expect zero LLM calls for paused/error cases.

### `build_replan_prompt`'s output goes to the planner as a `prompt`

The iterative supervisor calls `Supervisor.run(replan_prompt, ...)` with
the text `build_replan_prompt` produced. The planner has no separate
"replan context" channel — it sees the verbose replan text as just a
new prompt. Currently this works because the verbose text contains the
original prompt, gaps, and prior outputs, but the planner doesn't
*structurally* know it's being replanned. A future refinement: add a
dedicated `replan_context` arg to `Supervisor.run()` so the planner's
system prompt can react to it explicitly.

### Replay does NOT re-plan and does NOT inherit checkpointer state

`Supervisor.replay(run_id)` loads the validated spec the recorder
persisted on the original run, executes it under a *new* `run_id`, and
writes a new run directory. Notably:

- The **planner is not called** during replay. The point is to re-run
  the same shape, not to ask the planner what to do again.
- The **checkpointer is not seeded** with the original's state. If the
  spec contains `wait_for_event`, the replay pauses fresh from the start
  — it does NOT pick up from where the original left off.
- The original recording is **untouched**. New artifacts go to
  `runs/<new_run_id>/`. Default `new_run_id` is
  `<original>_replay_<utc_iso_timestamp>` so the original and the replay
  are colocated for easy diffing.

Use `replay()` to compare LLM output across model versions or runner
code changes. Use `resume()` (different method!) to continue a paused
run from where it stopped.

### Echo defaults vs production factories — which is active matters

`default_runners()` returns **placeholder echo runners** for every
runner-handled kind. They're pure, deterministic, no I/O — perfect for
tests but **never what you want in production**. Each kind has a real
factory you wire via `runners={}` to override:

| Kind            | Echo default        | Production factory                            |
|-----------------|---------------------|-----------------------------------------------|
| `llm_call`      | `run_llm_call`      | `build_openai_llm_runner` -> `OpenAILlmRunner`|
| `tool_call`     | `run_tool_call` (uses `DEFAULT_FAKE_TOOLS`) | **no real tool factory yet** (roadmap) |
| `reduce`        | `run_reduce` (concat/merge_dict only)       | `build_openai_reduce_runner` (adds llm_summarize) |
| `spawn_subagent`| `run_spawn_subagent`| `build_openai_spawn_subagent_runner`          |
| `emit_artifact` | `run_emit_artifact` | `make_emit_artifact_runner(FileArtifactSink(...))` |

`main.py` swaps to production factories when `--llm` is set (or for
`emit_artifact`, always — file persistence is useful even in no-LLM
demos). If you ship a new client of the supervisor, you must wire the
production factories explicitly. The compiler will happily run with
echoes — it has no way to tell.

A future hardening: a `strict_runners=True` mode on the executor that
refuses to fall back to echoes. Flagged in roadmap.md.

### Subagent registry must match the Registry's `subagents` allowlist

The `Registry.subagents` set is what the validator checks `agent_name`
against. The `runners={NodeKind.SPAWN_SUBAGENT: make_spawn_subagent_runner(...)}`
is what actually executes the call.

If those two diverge — e.g., you advertise `critic` in the Registry but
only wire `document_specialist` in the runner — the validator passes but
the runtime raises `RuntimeError("No subagent registered for 'critic'")`,
caught as `execution_failed`.

Keep them aligned at the same wiring layer (main.py / the supervisor
construction site). `DEFAULT_SUBAGENT_SYSTEM_PROMPTS` covers the default
allowlist, but if you add a new name to the allowlist you must add a
prompt and a wired subagent for it too.

### Windows console can't print Unicode by default

LLM responses regularly contain `→`, `—`, curly quotes, etc. Windows'
default `cp1252` codec crashes when `print()` hits these characters.
`main.py` reconfigures `sys.stdout`/`sys.stderr` to UTF-8 with
`errors="replace"` at startup so the demo never dies for display reasons.

If you write new scripts that print LLM output, do the same:
```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

### `interrupt()` raises on the first pass, returns on resume

`langgraph.types.interrupt(value)` does NOT return on the first execution
— it raises `GraphInterrupt` which LangGraph catches to persist state and
pause. On resume (`Command(resume=value)`), the same call returns the
resume value.

Implication for `wait_for_event`: any state-update code *before* the
interrupt call never makes it into state on the first pass. The
`make_wait_for_event_node` factory emits START + FINISH events *after*
the interrupt returns — meaning both events get their timestamps from
the resume pass, not the original pause. If you need to see "the run is
currently paused" in the trace, the supervisor or recorder must emit
that signal, not the wait node itself.

### `wait_for_event` requires a checkpointer at compile time, not run time

`LangGraphExecutor.compile()` raises `GraphCompilationError` if the spec
contains any `wait_for_event` node and the executor has no checkpointer.
This is deliberate: a missing checkpointer would otherwise silently turn
`interrupt()` into a runtime crash deep inside LangGraph internals.

For tests: `MemorySaver()` from `langgraph.checkpoint.memory`. For
production durability: `SqliteSaver(...)`.

### Resume uses `thread_id` = `run_id`

The executor passes `config={"configurable": {"thread_id": run_id}}` to
both `invoke` and `resume`. The checkpointer keys persisted state by
`thread_id`, so the supervisor's `run_id` is what ties a paused run to
its resumed continuation. Don't reuse `run_id` across logically-distinct
flows — it'll cross-contaminate checkpointer state.

### Branch routing requires character-for-character node-id match

The `branch` design uses the same name for both "the decision string" and
"the target node id". If an upstream LLM-call is told to emit "factual"
but the branch's `branches` list is `["factual_answer", "opinion_answer"]`,
the branch will halt because `"factual"` isn't in the branches set.

When prompting the planner (or hand-writing specs), the upstream node's
output must emit values that **exactly match** one of the branch names.
The planner prompt warns about this; reinforce it in node-specific
instructions when needed.

### `Send` payload IS the worker's state input

`Send(target, payload)` sets `payload` as that worker invocation's
complete state. Different Sends to the same target get isolated state.
Worker returns merge into global state via reducers.

To pass per-worker context: put it directly in the payload. Don't try to
broadcast a global "dispatch table" — that's just state pollution.

### Import cycle: `runtime` ↔ `compiler`

`app/runtime/__init__.py` exports `LangGraphExecutor`. The executor needs
`app.compiler.build`. The compiler imports `app.runtime.wrappers` and
`app.runtime.parallel_map`. → cycle.

Resolution: `LangGraphExecutor.compile()` imports `app.compiler.build`
**inside the method body**, not at module level. Don't move it back to a
top-level import.

### `StateGraph.compile()` is a runtime call, not a build step

You can call it inside a node, mid-execution. The whole project relies on
this — every supervisor run compiles a fresh transient graph from the
planner's spec.

## OpenAI structured output

### Strict json_schema mode rejects open dicts

`chat.with_structured_output(GraphSpec)` defaults to
`method="json_schema"` which requires `additionalProperties: false` on
every object schema. `GraphSpec` has `NodeSpec.params: dict[str, Any]`
which can't satisfy that — strict mode returns a 400.

Fix: `chat.with_structured_output(GraphSpec, method="function_calling")`.
Function-calling mode is more permissive and works with open dicts.

### LLM list outputs arrive as JSON-encoded strings

When `llm_call` is asked to "produce a JSON list of X", the runner
returns the *string* `'["a", "b", "c"]'`, not a Python list. Any
downstream consumer (like `parallel_map`'s `over` source) sees a string.

`parallel_map` opportunistically decodes two shapes:
1. Bare list: `"[a, b, c]"`
2. Single-key object: `"{\"<over_key>\": [a, b, c]}"` — LLMs *frequently*
   wrap their list in an object whose key matches the requested name.

If you add another upstream-list consumer, do the same opportunistic
decode.

### Planner needs literal "START" / "END" strings

The planner can produce edges that don't use the literal `"START"` /
`"END"` sentinels, leading to `no_start_to_end_path` /
`unreachable_node` validation errors. The system prompt has a worked
example with the exact JSON shape — **don't remove it**.

### Input/output keys need character-for-character matches

A downstream node's `inputs: ["sources"]` must match exactly some upstream
node's `outputs: ["sources"]`. The validator rejects "source", "Sources",
"the_sources", etc. as `missing_upstream_input`.

The planner's retry message includes the set of declared output keys
from the previous attempt so the model can rename or align.

## Pydantic

### `AIMessage.content` enforces string type

Can't construct `AIMessage(content=42)` for tests — Pydantic v2 rejects
non-string content. Use a duck-typed class with `.content` attribute
instead:

```python
class _NonStringResponse:
    content = 42
```

### `EdgeSpec` uses `from_` with alias `"from"`

`from` is a Python keyword. The model has `populate_by_name=True` so both
`from_` and `from` work as input. **Always dump with `by_alias=True`** for
external artifacts (spec.json, planner outputs, etc.) so JSON consumers
see `"from"`.

## Python 3.13

### `datetime.utcnow()` is deprecated

Use `datetime.now(UTC)` everywhere. `-W error` catches this in CI.

## Dev environment

### `python app/main.py` vs `python -m app.main`

Running the file directly doesn't add the project root to `sys.path`, so
`from app.X import Y` fails. Either use `-m app.main` OR include the
`sys.path` bootstrap that's in main.py's header (gated on
`__name__ == "__main__" and __package__ in (None, "")`).

### Secrets in `.env`

`.env` is gitignored. `python-dotenv` `load_dotenv()` runs at the top of
main.py. If a key shows up in a transcript or PR, **rotate it** — local
git history isn't the only place keys can leak.
