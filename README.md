# Dynamic Subgraphs

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Ian-Tharp/Dynamic-Subgraphs/actions/workflows/ci.yml/badge.svg)](https://github.com/Ian-Tharp/Dynamic-Subgraphs/actions/workflows/ci.yml)
[![Typed](https://img.shields.io/badge/typed-py.typed-blue.svg)](https://peps.python.org/pep-0561/)

A governed runtime for LLM-generated workflows. A stable `Supervisor` turns a prompt
into a **validated plan** — data, never executable code — then compiles and runs it as
a **transient** LangGraph workflow over a bounded, allowlisted vocabulary of node kinds.
Recording is **opt-in**: when enabled, each run is captured as a replayable, diffable,
cost-attributed artifact under `runs/<run_id>/` — the public SDK defaults to in-memory
execution and writes no files.

What that buys you over a free-form agent loop: the model proposes the workflow but
never executes arbitrary code; the compiler only instantiates registry-approved node
kinds; recursion is depth- and budget-capped; and any run — success or failure — can be
recorded for inspection and replay. LangGraph types stay behind the `compiler/` and
`runtime/` boundaries. An optional thin FastAPI layer exposes the supervisor over HTTP.

## What can it build?

From one prompt, the planner assembles a **transient** graph out of a small set
of governed node kinds — `llm_call`, `tool_call`, `branch`, `reduce`,
`parallel_map`, `spawn_subagent`, `spawn_subgraph`, `wait_for_event`,
`emit_artifact`. A few of the shapes it produces (these render in the same Mermaid
the engine writes to `graph.mmd` for every run):

**Parallel research → recommend** — fan out to independent workers, then reduce:

```mermaid
graph TD
    START([START])
    END([END])
    extract_a["extract_a<br/>tool_call"]
    extract_b["extract_b<br/>tool_call"]
    summarize_a["summarize_a<br/>llm_call"]
    summarize_b["summarize_b<br/>llm_call"]
    compare["compare_and_recommend<br/>reduce"]
    START --> extract_a
    START --> extract_b
    extract_a --> summarize_a
    extract_b --> summarize_b
    summarize_a --> compare
    summarize_b --> compare
    compare --> END
```

**Tool-grounded answer** — search the web, answer from it, write a report:

```mermaid
graph TD
    START([START])
    END([END])
    search["web_search<br/>tool_call"]
    answer["answer<br/>llm_call"]
    report["report<br/>emit_artifact"]
    START --> search
    search --> answer
    answer --> report
    report --> END
```

**Dynamic routing** — classify the request, then take only the path it warrants:

```mermaid
graph TD
    START([START])
    END([END])
    classify["classify_intent<br/>llm_call"]
    route{"route<br/>branch"}
    answer["answer<br/>llm_call"]
    search["web_search<br/>tool_call"]
    investigate["investigate<br/>spawn_subgraph"]
    START --> classify
    classify --> route
    route -->|simple| answer
    route -->|needs data| search
    route -->|complex| investigate
    search --> answer
    investigate --> answer
    answer --> END
```

**Human-in-the-loop** — pause for an external event, then resume:

```mermaid
graph TD
    START([START])
    END([END])
    draft["draft_proposal<br/>llm_call"]
    review["await_approval<br/>wait_for_event"]
    finalize["finalize<br/>llm_call"]
    START --> draft
    draft --> review
    review --> finalize
    finalize --> END
```

**Nested composition** — a node plans and runs a child graph on a fresh, isolated
state envelope under enforced **depth and spend ceilings** (recursion that can't run
away):

```mermaid
graph TD
    START([START])
    END([END])
    plan["plan<br/>llm_call"]
    investigate["investigate<br/>spawn_subgraph"]
    synthesize["synthesize<br/>reduce"]
    START --> plan
    plan --> investigate
    investigate --> synthesize
    synthesize --> END
```

### How it works

Those graphs are *transient* — planned, validated, run, and recorded per
request. The only thing that stays fixed is the **Supervisor**, a stable host
graph that governs every run:

```mermaid
graph LR
    START([prompt]) --> plan
    plan["plan<br/>(GraphSpec)"] --> validate
    validate["validate<br/>(registry + budgets)"] --> run
    run["compile & run<br/>(transient graph)"] --> record
    record["record<br/>(spec·trace·mermaid·cost)"] --> replay
    replay["replay / diff / audit"] --> respond
    respond([result]) --> END([END])
```

The planner emits a **plan, never code**; the compiler only instantiates
registry-approved node kinds; and every run — success or failure — is recorded as a
replayable, diffable, cost-attributed artifact. That audit trail is the point: it's
what a free-form agent loop can't give you.

### When (not) to reach for it

Dynamic Subgraphs earns its keep when **the shape of the work varies per input *and*
you need the run governed and auditable**. Use it when:

- The workflow can't be enumerated ahead of time — the right nodes/edges depend on
  the request (heterogeneous intake, branching investigations, data-dependent
  recursion whose depth isn't known until runtime).
- You need an audit trail: a validated plan, a per-node trace, deterministic replay,
  and cost attribution — for compliance, debugging, or reproducibility.
- The model should *propose* the workflow but must not execute arbitrary code, and
  its tool/capability surface must stay allowlisted and budget-capped.

**Reach for something simpler when:**

- **The shape is known.** If you can draw the DAG ahead of time, hand-author a fixed
  [LangGraph](https://github.com/langchain-ai/langgraph) graph — it's cheaper and more
  predictable. (If you can write the orchestration as a script, you don't need a
  planner generating it.)
- **The task is small.** A frontier model in a plain tool loop already decomposes a
  one-to-three-step task in-context; a planning round-trip is pure overhead there.
- **Your hard problem is global consistency, not orchestration.** Isolated child
  envelopes are great for blast-radius but work *against* shared canonical state —
  pair DS with a retrieval/consistency layer rather than expecting it to enforce one.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add dynamic-subgraphs                 # slim core (engine only)
uv add "dynamic-subgraphs[openai]"       # + OpenAI provider
uv add "dynamic-subgraphs[anthropic]"    # + Anthropic provider
uv add "dynamic-subgraphs[ollama]"       # + local Ollama provider
uv add "dynamic-subgraphs[api]"          # + FastAPI HTTP surface
uv add "dynamic-subgraphs[cost]"         # + automatic result.cost (LiteLLM prices)
uv add "dynamic-subgraphs[all]"          # everything
```

Or with pip:

```bash
pip install dynamic-subgraphs
pip install "dynamic-subgraphs[openai]"  # same extras: anthropic, ollama, api, cost, all
```

The core install is intentionally light (`langgraph`, `langchain-core`,
`pydantic`, `python-dotenv`); provider SDKs and the API server are optional
extras so you only pull what you use.

## Quickstart (development)

```bash
# Set up the dev environment (all extras + dev tooling)
uv sync --all-extras

# Run the offline mock demo (free, no tokens)
uv run python -m app.main "compare A and B"

# Run the HTTP API (boots in mock mode by default; needs the `api` extra)
uv run python -m app.api
```

By default everything runs in **mock** mode — free and offline. Set
`DS_PLANNER=llm` and `DS_PROVIDER=<provider>` to use the real planner and
grounded tools. The legacy `DS_PLANNER=openai` value still maps to
`planner=llm` with `provider=openai`.

Built-in providers (`default_model_providers()`):

| `DS_PROVIDER` | Package | Credentials |
|---------------|---------|-------------|
| `openai` | `langchain-openai` | `OPENAI_API_KEY` |
| `anthropic` | `langchain-anthropic` | `ANTHROPIC_API_KEY` |
| `ollama` | `langchain-ollama` | none (local server; `OLLAMA_BASE_URL` optional) |

Each role (planner, worker, reducer, subagent, judge) can target a different
provider/model through `RunConfig`'s role-specific `ModelRef` fields; unset
roles fall back to the worker model, then to the base `provider`+`model`.

## SDK usage

The `dynamic_subgraphs` package is the importable facade — build an
`EngineConfig`, hand it to the engine, then call `run()`:

```python
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

# Cloud (key from env)
engine = DynamicSubgraphs(EngineConfig(model=Model("openai", "gpt-5.4-nano")))

# ...or a local LM Studio / Ollama server (bring your own URL/key/model)
engine = DynamicSubgraphs(EngineConfig(model=Model.lmstudio("google/gemma-3-27b")))
engine = DynamicSubgraphs(EngineConfig(model=Model.ollama("llama3.1")))

result = engine.run("Compare two sources on X and recommend one.")
result.response      # synthesized answer text
result.values        # {output_key: value, ...}
result.plan          # the generated GraphSpec
result.artifacts     # {filename: Path} (populated only when recording is on)
result.usage         # exact TokenUsage: input/output/total + per-model breakdown
result.cost          # USD (None unless a pricing book is configured — see below)
result.effective_budget  # the host-granted budget (planner request capped by policy)
result.plan_attempts     # planner attempts (>1 if a rejected plan was repaired)
```

### Token usage & cost

`result.usage` is **always** populated with the providers' own reported token
counts (via LangChain's usage callback — exact, all providers, no estimation).

**Cost** works automatically with the `cost` extra (it uses
[LiteLLM](https://github.com/BerriAI/litellm)'s maintained price map — you don't
specify prices, and we don't ship a table that goes stale):

```bash
pip install "dynamic-subgraphs[openai,cost]"
```

```python
engine = DynamicSubgraphs(EngineConfig(model=Model("openai", "gpt-5.4-nano")))
r = engine.run("...")
r.usage.total_tokens   # e.g. 3233   (exact, free, always)
r.cost                 # e.g. 0.0021 (USD — auto-computed)
```

Without the extra, `result.cost` is `None` (tokens are still exact). You can
also pass a manual `pricing` book on `EngineConfig` to **override** prices or to
cover local / custom-endpoint models LiteLLM doesn't know:

```python
EngineConfig(model=..., pricing={"my-model": {"input_per_1m": 0.5, "output_per_1m": 1.0}})
```

(If you use LangSmith, it computes cost server-side as well.)

All engine configuration lives on `EngineConfig`: the per-role models
(`model`, `planner_model`, `worker_model`, `reducer_model`, `subagent_model`,
`judge_model`), the `recording` policy, `planner` mode, `runs_dir`,
`providers`, `checkpointer`, the host-owned `policy` (`ExecutionPolicy`), and
`max_plan_attempts` (the plan-repair loop).

### Governance: host-owned limits & plan repair

The planner proposes a workflow; the **host** owns the limits. An
`ExecutionPolicy` is the contract — and it's **enforced**, not advisory:

```python
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, ExecutionPolicy, Model

engine = DynamicSubgraphs(EngineConfig(
    model=Model("openai", "gpt-5.4-nano"),
    policy=ExecutionPolicy(
        max_nodes=6, max_llm_calls=4, max_depth=2, max_fanout=16,
        allowed_tools=frozenset({"web_search"}),   # host ∩ registry
    ),
))
r = engine.run("...")
r.effective_budget   # the granted budget = min(host, planner request)
```

Enforced at validation (root **and** every nested `spawn_subgraph` child):

- **Budgets** — node/LLM-call counts capped at `min(host, planner request)`; a
  plan can't grant itself a larger budget.
- **Allow-sets** — tool / subagent / node-kind use is the host ∩ registry
  intersection; a plan naming a disallowed capability is rejected.
- **Fan-out** — a `parallel_map` over more items than `max_fanout` halts before
  any work fires.
- **Nesting** — a child's budget is the parent's *remaining* allowance, so a nest
  can't outspend the root; depth is capped at the tighter of the host and the rail.
- **Wall-clock** — a run that outruns `max_wall_seconds` is abandoned (a hung
  runner can't block forever).

When a plan is rejected for a *recoverable* reason, the supervisor feeds the
issues + host limits back into a re-plan, up to `max_plan_attempts` (**default
2** — repair once; set `1` for strict block-and-report). So a too-ambitious plan
is re-planned *within* the limits instead of just failing. Defaults are
permissive-but-bounded, so a typical plan is unaffected.

> ⚠️ **Use a capable model for the planner.** The planner must emit a valid
> `GraphSpec`; small local models (7B-class, and in practice anything below
> ~20–30B) frequently produce invalid plans and fail. Run small/local models
> as the `worker_model` with a stronger `planner_model`.

### Recording (opt-in)

By default the engine writes **no files** — embedding it never clutters your
working tree. **Suggestion:** set a `recording` policy while developing or
debugging to capture the trace under `runs/<run_id>/`, then leave it at the
default in production / library use:

```python
from dynamic_subgraphs import Recording, Artifact

engine = DynamicSubgraphs(EngineConfig(
    model=Model("openai", "gpt-5.4-nano"),
    recording=Recording.debug(),      # capture everything
    runs_dir="runs",
))
```

Recording is **granular** — choose exactly which artifacts to write with the
`Artifact` enum (its values are the filenames) and the `Recording` policy:

```python
recording=Recording.visual_only()           # just graph.mmd (the diagram)
recording=Recording.all() - {Artifact.SPEC}  # everything except spec.json
recording={Artifact.MERMAID, Artifact.TRACE}  # a raw set works too
```

| Preset | Writes | Use for |
|--------|--------|---------|
| `Recording.none()` (default) | nothing | embedding / production |
| `Recording.all()` | every artifact | full capture |
| `Recording.debug()` | every artifact | debugging a run |
| `Recording.visual_only()` | `graph.mmd` | a picture of the graph |
| `Recording.replayable()` | `spec.json` + `output.json` | enabling `resume`/`replay` |

Coding agents can enumerate every valid option via
`DynamicSubgraphs.capabilities()`. See [`docs/recipes.md`](./docs/recipes.md)
for copy-pasteable patterns.

Engine model defaults can be overridden per `run()` call, so each run picks the
models for its own node calls (e.g. a cheap cloud planner with local workers):

```python
result = engine.run(
    "Investigate this task.",
    planner_model=Model("openai", "gpt-5.4-nano"),
    worker_model=Model.lmstudio("openai/gpt-oss-20b"),
)
```

## Documentation

- [`examples/`](./examples/) — runnable, standalone SDK integration examples (one file per pattern).
- [`docs/recipes.md`](./docs/recipes.md) — copy-pasteable SDK patterns (local models, hybrid, recording presets, debugging) + tested-model and latency tables.
- [`docs/api-stability.md`](./docs/api-stability.md) — API stability & change policy: what counts as public, SemVer, deprecation, and how we keep changes non-breaking.
- [`docs/evals/`](./docs/evals/) — eval reports (e.g. the gpt-5.4-nano vs claude-haiku-4-5 e2e comparison: latency / tokens / cost / quality, traced via LangSmith).
- [`docs/api.md`](./docs/api.md) — the HTTP surface over the supervisor (endpoints, modes, auth, examples).
- [`docs/dynamic-graphs-canonical-design-v1.md`](./docs/dynamic-graphs-canonical-design-v1.md) — canonical project design and source of truth.
- [`docs/index.md`](./docs/index.md) — full documentation index.
- [`AGENTS.md`](./AGENTS.md) — agent-facing package map and MVP sequence.

## Contributing & support

Contributions are welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the
dev setup, test, and formatting workflow, and
[`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md). Found a bug or have a request?
[Open an issue](https://github.com/Ian-Tharp/Dynamic-Subgraphs/issues). For
security reports, see [`SECURITY.md`](./SECURITY.md).

## Status

Pre-1.0 (`0.x`) — the public SDK surface is usable and tested, but the API may
change between minor versions until 1.0. See [`CHANGELOG.md`](./CHANGELOG.md).

## License

Licensed under the [Apache License 2.0](./LICENSE). See [`NOTICE`](./NOTICE) for
attribution.
