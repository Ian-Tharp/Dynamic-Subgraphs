# SDK Next Steps

This note captures the work needed to turn Dynamic Subgraphs from a local
runtime/API project into an importable SDK that CORE can use directly.

## Current State

- A public `dynamic_subgraphs` package now exists and exposes the SDK facade:
  `DynamicSubgraphs`, `Model` (alias of `ModelRef`), `ModelSelection`, and
  `RunResult`. The engine implementation still lives under `app` and is
  re-exported (migration layer). One-line usage:
  ```python
  from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model
  engine = DynamicSubgraphs(EngineConfig(model=Model.lmstudio("google/gemma-3-27b")))
  result = engine.run("Compare two sources and recommend one.")
  result.response; result.values; result.plan; result.artifacts
  ```
  Configuration is a single immutable `EngineConfig` object passed to the
  engine (opinionated; no loose-kwargs constructor). It holds the per-role
  models, `recording` policy, `planner`, `runs_dir`, `providers`, and
  `checkpointer`. **Planner capability caveat:** planner reliability depends on
  the model *and* graph complexity. Tested locally: `gpt-oss-20b` (20B) emitted
  invalid `GraphSpec`s; `gemma-3-27b` worked but was flaky; `gemma-4-e4b` (~4B)
  planned a simple single-node graph yet failed a multi-source graph 3/3. Cloud
  `gpt-5.4-nano` and `claude-haiku-4-5` planned reliably. All models tested
  worked fine as *workers*. Run small/local models as `worker_model` behind a
  capable `planner_model`. Full table in `docs/recipes.md`.
  Engine-level model defaults can be overridden per `run()` call (so each run
  picks its own node models). `Model` carries first-class `base_url`,
  `api_key`, and `structured_method`, with `Model.lmstudio(...)`,
  `Model.openai_compatible(...)`, and `Model.ollama(...)` helpers.
  **Verified live:** fully-local LM Studio (planner+worker via `json_schema`)
  and hybrid cloud-planner + local-worker both return `status="ok"`.
  Recording is **opt-in** (`record=False` default): the engine writes no files
  unless `record=True`, so embedding it never clutters the caller's tree. Off
  uses a `NullRecorder` + in-memory `CollectingArtifactSink`; on uses
  `FileRecorder` + `FileArtifactSink` under `runs/<run_id>/`.
- The runtime implementation still lives under the Python package name `app`.
- `RunConfig` now separates execution mode from provider choice:
  - `planner="mock"` is the token-free offline path.
  - `planner="llm"` uses the selected model provider.
  - legacy `planner="openai"` is normalized to `planner="llm"` with
    `provider="openai"`.
- `ProviderRegistry`, `ModelProvider`, and `ModelRef` provide the
  provider-neutral model-selection seam.
- Three built-in provider adapters ship by default (`default_model_providers()`):
  `OpenAIModelProvider`, `AnthropicModelProvider`, and `OllamaModelProvider`
  (local, no credentials). Provider lazy-imports keep optional SDKs off the
  hot path. Conformance tests cover lookup, credential checks, chat
  construction, and structured-output construction for all three.
- `RunConfig` supports **role-specific** model selection: `planner_model`,
  `worker_model`, `reducer_model`, `subagent_model`, and `judge_model` as
  optional `ModelRef`s. Unset roles fall back to the worker model, which falls
  back to the base `provider`+`model`. `assembly.py` routes each role to its
  provider (reusing one client when refs match) so a single run can mix
  providers (e.g. OpenAI planner + Anthropic workers). `RunConfig.providers_in_use()`
  drives multi-provider credential checks in the API.
- FastAPI accepts and persists `planner`, `provider`, and `model` so
  resume/replay keep the same provider selection as the original run.
  **Not yet:** the API request/response and run-config store do not yet carry
  the per-role `ModelRef`s — that plumbing lands with the SDK facade below.

## Target SDK Shape

The SDK should expose a small public facade that does not require callers to
import FastAPI internals or know about assembly details:

```python
from dynamic_subgraphs import DynamicSubgraphs, DynamicSubgraphsConfig, ModelRef

engine = DynamicSubgraphs.from_config(
    DynamicSubgraphsConfig(
        planner="llm",
        planner_model=ModelRef(provider="openai", model="gpt-5.4-nano"),
        worker_model=ModelRef(provider="anthropic", model="claude-example"),
        runs_dir="core_runs/dynamic_subgraphs",
    )
)

result = engine.run("Investigate this CORE task.", run_id="core-001")
```

The existing `Supervisor` remains the core engine. The SDK facade should wrap
configuration, provider registration, recorder setup, and checkpointer setup.

## Next Implementation Slices

1. Package namespace
   - ✅ Created the `dynamic_subgraphs` import namespace with public exports in
     `dynamic_subgraphs/__init__.py`.
   - ✅ `app` is kept as the implementation/migration layer (re-exported).
   - ⏳ Eventually move implementation modules under `dynamic_subgraphs/` (or
     make `app` a thin alias) — needed before publishing a clean wheel.

2. SDK facade
   - ✅ `DynamicSubgraphs(model=..., planner_model=..., ...)` engine with
     per-`run()` model overrides; returns a `RunResult` exposing
     `response`/`values`/`plan`/`artifacts`/`status`/`errors`. Unit + live
     verified.
   - ⏳ Expose `run_chain`, `resume`, and `replay` on the facade (the
     `Supervisor` already supports them; just need facade methods + result
     types).
   - ⏳ Keep FastAPI as an optional transport wrapper over the same facade.

3. Role-specific model selection
   - ✅ Split one `model` into role-specific refs (`planner_model`,
     `worker_model`, `judge_model`, `reducer_model`, `subagent_model`) on
     `RunConfig`, with unset roles defaulting to the worker model.
   - ✅ `assembly.py` routes each role to its provider via the registry.
   - ⏳ Persist all role refs for resume/replay (API store + serialize) — the
     run-config store currently persists only `planner`/`provider`/`model`.
   - ⏳ Expose role refs on the FastAPI request bodies and the SDK facade.

4. Provider adapters
   - ✅ OpenAI is the first built-in provider.
   - ✅ Added Anthropic (`langchain-anthropic`, tool-calling structured output)
     and Ollama (`langchain-ollama`, local, no credentials) adapters, both
     registered in `default_model_providers()`.
   - ✅ Conformance tests cover provider lookup, credential checks, chat
     construction, and structured-output construction.
   - ⏳ Live structured-output verification for Anthropic is blocked only on
     account credits (the adapter reaches the API and authenticates; a real
     run returned a billing 400, not a wiring error). Ollama live verification
     needs a running local server.

5. Packaging metadata
   - Decide whether the package should require Python 3.13 or lower that floor
     for CORE compatibility.
   - Split optional dependencies:
     - core runtime
     - `api`
     - `openai`
     - future provider extras
     - `dev`
   - Add wheel build verification with `uv build`.

6. CORE integration smoke
   - Install the built wheel into a clean temporary environment.
   - Run an import smoke test.
   - Run a mock SDK execution.
   - Run an LLM execution only behind an explicit env gate.
   - Verify recorded runs land under CORE's configured run directory.

7. Notion/spec consolidation
   - Pull the prior Notion notes into a source-controlled integration brief.
   - Capture CORE-specific requirements that are not represented in code yet.
   - Link the brief from this document or replace this note with a fuller
     implementation plan.

## Verification Checklist

Run these before handing an SDK packaging PR to CORE:

```powershell
.venv\Scripts\python.exe -m pytest -q
uv build
```

Then install the wheel in a clean environment and run:

```powershell
python -c "from dynamic_subgraphs import DynamicSubgraphs"
```

For real provider verification, preload credentials before collection:

```powershell
$env:DS_RUN_INTEGRATION = "1"
.venv\Scripts\python.exe -m pytest tests/test_integration_api.py -v
```

## Open Decisions

- Whether CORE wants in-process SDK usage, sidecar HTTP usage, or both.
- Which non-OpenAI provider should be the first production adapter.
- Whether provider credentials are read only from environment variables or from
  CORE's secrets/config system.
- Whether run recording stays filesystem-backed for CORE or moves behind a
  storage adapter.
