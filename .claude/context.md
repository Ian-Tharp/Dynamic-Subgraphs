# Project context

## What this is

**Dynamic Subgraphs**: a governed runtime where an LLM synthesizes a
transient LangGraph workflow per problem. The system validates, compiles,
executes, and records the graph, then discards the runtime object. Bounded
by a registry of node kinds — the "language" the planner composes from.

The thesis: *the registry is the language; the graph is its temporary
executable form*. Get the registry right and most other choices are
recoverable.

## What's shipped

| Layer | Status | Notes |
|---|---|---|
| Models + GraphSpec | ✓ | `app/models/` |
| Registry + validator | ✓ | `app/registry/` — trust boundary for everything downstream |
| Compiler (spec → StateGraph) | ✓ | `app/compiler/build.py` |
| Runtime: executor, runners, wrappers, state | ✓ | `app/runtime/` |
| Recording (full artifacts per run, failed runs included) | ✓ | `app/recording/`, `runs/<id>/` |
| Supervisor (plan → validate → execute → record → respond) | ✓ | `app/supervisor/`, with status taxonomy |
| LLM planner (ChatOpenAI, structured output, validation retry) | ✓ | `app/supervisor/llm_planner.py` |
| LLM runner for `llm_call` | ✓ | `app/runtime/llm_runner.py` |
| LLM-backed reduce (`llm_summarize`) | ✓ | same file |
| `parallel_map` (compiler-native + `Send` + JSON-tolerant input) | ✓ | `app/runtime/parallel_map.py` |
| `branch` (compiler-native + `add_conditional_edges`) | ✓ | `app/runtime/branch.py` |
| `wait_for_event` (compiler-native + LangGraph `interrupt()`) | ✓ | `app/runtime/wait_for_event.py` |
| Executor `checkpointer` + `paused` ExecutionResult + real `resume()` | ✓ | `app/runtime/executor.py` |
| Recorder `load_validated_spec` + per-call `overwrite` | ✓ | `app/recording/recorder.py` |
| Supervisor `resume(run_id, event)` + `paused`/`resume_failed` statuses | ✓ | `app/supervisor/supervisor.py` |
| `spawn_subagent` (echo default + OpenAI factory with role prompts) | ✓ | `app/runtime/subagents.py` |
| `emit_artifact` (echo default + `CollectingArtifactSink` / `FileArtifactSink`) | ✓ | `app/runtime/artifacts.py` |
| Shared utility: `render_value_for_prompt` (state.py) — value→prompt rendering | ✓ | dedup'd from llm_runner + subagents |
| Shared utility: `build_openai_chat` — single ChatOpenAI lazy-import seam | ✓ | `app/runtime/chat_models.py` |
| `Supervisor.replay(run_id, *, new_run_id=None)` — load recorded spec, re-execute fresh | ✓ | `app/supervisor/supervisor.py` |
| `Supervisor.run_iteratively(...)` — bounded meta-loop with `IterationDecider` Protocol | ✓ | `app/supervisor/iteration.py` |
| `LlmIterationDecider` + `build_openai_iteration_decider` — LLM judges output against criteria, emits structured replan/stop/ask/fail decisions | ✓ | `app/supervisor/iteration.py` |
| Real `tool_call` runners — `web_search` (DuckDuckGo + Bing scrape fallback), `policy_lookup`, `document_extract`, `create_follow_up_task` | ✓ (partial — see roadmap) | `app/runtime/tools.py` |
| `SearchProvider` Protocol + `TavilySearchProvider` (production) + env-aware factory (`build_default_search_provider`) — Tavily activates automatically when `TAVILY_API_KEY` is set, DDG+Bing fallback otherwise | ✓ | `app/runtime/tools.py` |
| Chain-level recording — `FileRecorder.record_chain` / `.load_chain`, `Supervisor.run_iteratively(record_chain=True)` writes `runs/<chain_id>/chain.json` + `chain.md` | ✓ | `app/recording/recorder.py` |
| Judge truncation fix — `LlmIterationDecider` value-render limit raised from 500 → 4000 chars, system prompt notes truncation is display-only | ✓ | `app/supervisor/iteration.py` |
| `strict_runners` flag on executor — refuses to fall back to default echoes | ✓ | `app/runtime/executor.py` |

## Executable node kinds

| Kind | Status | Path |
|---|---|---|
| `llm_call` | ✓ | runner |
| `tool_call` | ✓ | runner + fake-tool registry (real tools = future) |
| `reduce` | ✓ | runner; strategies: concat, merge_dict, llm_summarize |
| `parallel_map` | ✓ | compiler-handled (dispatcher/worker/join) |
| `branch` | ✓ | compiler-handled (passthrough + conditional_edges) |
| `wait_for_event` | ✓ | compiler-handled (`interrupt()` + checkpointer + resume) |
| `spawn_subagent` | ✓ | runner-handled; echo default, OpenAI-backed factory |
| `emit_artifact` | ✓ | runner-handled; echo default, `FileArtifactSink` wired in `main.py` |

**All 8 registry kinds executable.** The runtime is functionally complete for phase 1.

## Test surface

~222 tests under `tests/`, all passing with `uv run pytest -W error`.
Files mirror modules: `test_registry.py`, `test_validator.py`,
`test_wrappers.py`, `test_executor.py`, `test_parallel_map.py`,
`test_branch.py`, `test_wait_for_event.py`, `test_subagents.py`,
`test_emit_artifact.py`, `test_replay.py`, `test_iterative_supervisor.py`,
`test_tools.py`, `test_recording.py`, `test_supervisor.py`,
`test_llm_planner.py`, `test_llm_runner.py`, `test_e2e_pipeline.py`,
`test_graph_spec.py`.

## Demo entrypoint

```
uv run python -m app.main                        # token-free (StaticPlanner)
uv run python -m app.main --llm                  # real LLM planner + runner
uv run python -m app.main --llm "your prompt"
uv run python -m app.main --llm --run-id "exp-1"
```

`--llm` swaps in `LLMPlanner` + `OpenAILlmRunner` + `LlmReduceRunner` and
widens the planner's reduce-strategy set to include `llm_summarize`.
Without `--llm`, every `llm_call` is the mock and reduce is deterministic.

## Configuration

- `.env` (gitignored): `OPENAI_API_KEY`, `LANGSMITH_*`, optional `TAVILY_API_KEY`
- `python-dotenv` loaded by main.py
- Default LLM model: `gpt-5.4-nano` (override with `--model`)
- When `TAVILY_API_KEY` is set, `web_search` uses Tavily; otherwise falls back to DuckDuckGo+Bing scrape (lower quality, no key required). Free Tavily tier: https://tavily.com
