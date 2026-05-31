# Workflows

## Setup

```bash
uv sync --extra dev
# Then create .env with OPENAI_API_KEY=... (gitignored)
```

## Running the demo

```bash
uv run python -m app.main                        # token-free (StaticPlanner)
uv run python -m app.main --llm                  # real LLM planner + runner
uv run python -m app.main --llm "your prompt"
uv run python -m app.main --llm --run-id "exp-1"
uv run python -m app.main --llm --model gpt-5.4-nano
```

Recorded run lands at `runs/<run_id>/` with spec.json, trace.jsonl,
output.json, graph.mmd, summary.md, and prompt.md when provided.

## Tests

```bash
uv run pytest -W error                  # full suite, warnings = failures
uv run pytest -W error tests/test_X.py  # one file
uv run pytest -W error tests/test_X.py::test_Y -vv   # one test, verbose
```

The suite is token-free; LLM tests use fake models (`FakeChatModel`,
`FakeStructuredModel`).

## Test conventions

- **AAA structure** via blank-line separation (arrange / act / assert).
  No `# arrange` style comments.
- **Shared fixtures from `tests/conftest.py`**: `make_node`, `make_edge`,
  `minimal_spec`, `spec_factory`, `registry`. Don't hand-roll builders.
- **Parametrize** when 3+ tests differ in one input axis.
- **Fake models for LLM** — never call real APIs from unit tests.
- **Trace shape as topological invariant**: assert "for every edge,
  upstream finish < downstream start in the event stream", not absolute
  event positions.

## Adding a new node kind (recipe)

1. `app/models/node_kinds.py` → add `NodeKind.X = "x"`.
2. `app/registry/params.py` → add typed `XParams` Pydantic model.
3. `app/registry/definitions.py` → add `NodeKindDefinition(...)` to
   `default_kind_definitions()`.
4. Decide: **runner-handled** or **compiler-handled**?
   - Runner-handled (most kinds): write `run_x(state, params)` in
     `app/runtime/runners.py`, register in `default_runners()`.
   - Compiler-handled (special wiring needed, e.g., needs `Send`):
     - Add `NodeKind.X` to `COMPILER_HANDLED_KINDS` in
       `app/compiler/build.py`.
     - Implement node-emission helpers in `app/runtime/<x>.py`.
     - Handle in `build_graph` with custom emission + edge rewriting.
5. Update `LLMPlanner`'s prompt template with the kind's required params
   (the executable_kinds list updates automatically).
6. Tests: param validation, registry admission, executor end-to-end,
   error path. Follow `test_parallel_map.py` for compiler-handled kinds.
7. Optional: smoke with `--llm` to verify the planner picks it.

## Surgical fix pattern (when `--llm` fails)

When the LLM smoke fails, each iteration should move the failure
**further down the pipeline**, never sideways:

| Status | Stage | What to look at |
|---|---|---|
| `plan_failed` | Planner raised | Model output or prompt clarity |
| `validation_failed` | RegistryValidationError | Issue codes; tighten prompt or repair message |
| `compile_failed` | Unsupported kind | Planner used an unimplemented kind; constrain prompt |
| `execution_failed` | Runner exception | Runtime semantics or interop (LLM string vs list, etc.) |
| `record_failed` | Recorder raised | Disk / path / serialization |

Process:
1. Read status + issue code + error message.
2. **One** surgical change targeting that exact failure.
3. Re-run `pytest -W error` (free) to confirm no regression.
4. Re-run `--llm` once.
5. Confirm the failure moved down; repeat until `ok`.

Avoid speculative fixes for issues you haven't observed.

## When LLM behavior surprises you

Common things to check before "the model is wrong":
- Is the system prompt advertising only what the runtime can execute?
- Is the repair message specific enough (issue code + field)?
- Is there a worked example showing the exact JSON shape?
- Is the model wrapping output in `{"<key>": [...]}` instead of `[...]`?
- Is the model using Python-template syntax (`{state.values.X}`) instead
  of natural language?
