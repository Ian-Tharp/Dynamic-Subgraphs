# Dynamic Subgraphs

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Ian-Tharp/Dynamic-Subgraphs/actions/workflows/ci.yml/badge.svg)](https://github.com/Ian-Tharp/Dynamic-Subgraphs/actions/workflows/ci.yml)
[![Typed](https://img.shields.io/badge/typed-py.typed-blue.svg)](https://peps.python.org/pep-0561/)

A governed dynamic graph runtime: a stable `Supervisor` plans, validates, and runs
**transient** LangGraph workflows assembled from a bounded node registry and a
validated `GraphSpec`, recording every run under `runs/<run_id>/`. The planner emits
*plans*, never executable code; the compiler only instantiates registry-approved node
kinds; and LangGraph types stay behind the `compiler/` and `runtime/` boundaries. An
optional thin FastAPI layer exposes the supervisor over HTTP.

## Install

```bash
pip install dynamic-subgraphs                 # slim core (engine only)
pip install "dynamic-subgraphs[openai]"       # + OpenAI provider
pip install "dynamic-subgraphs[anthropic]"    # + Anthropic provider
pip install "dynamic-subgraphs[ollama]"       # + local Ollama provider
pip install "dynamic-subgraphs[api]"          # + FastAPI HTTP surface
pip install "dynamic-subgraphs[all]"          # everything
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
```

All engine configuration lives on `EngineConfig`: the per-role models
(`model`, `planner_model`, `worker_model`, `reducer_model`, `subagent_model`,
`judge_model`), the `recording` policy, `planner` mode, `runs_dir`,
`providers`, and `checkpointer`.

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
