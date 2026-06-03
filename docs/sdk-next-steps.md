# SDK Next Steps

This note captures the work needed to turn Dynamic Subgraphs from a local
runtime/API project into an importable SDK that CORE can use directly.

## Current State

- The runtime still lives under the Python package name `app`.
- `RunConfig` now separates execution mode from provider choice:
  - `planner="mock"` is the token-free offline path.
  - `planner="llm"` uses the selected model provider.
  - legacy `planner="openai"` is normalized to `planner="llm"` with
    `provider="openai"`.
- `ProviderRegistry`, `ModelProvider`, and `ModelRef` provide the first
  provider-neutral model-selection seam.
- OpenAI remains the only built-in production provider adapter.
- FastAPI now accepts and persists `planner`, `provider`, and `model` so
  resume/replay keep the same provider selection as the original run.

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
   - Create a `dynamic_subgraphs` import namespace.
   - Keep `app` temporarily as a compatibility alias or migration layer.
   - Move public exports into `dynamic_subgraphs/__init__.py`.

2. SDK facade
   - Add `DynamicSubgraphsConfig`.
   - Add `DynamicSubgraphs.from_config(...)`.
   - Expose `run`, `run_chain`, `resume`, and `replay`.
   - Keep FastAPI as an optional transport wrapper over the same facade.

3. Role-specific model selection
   - Split one `model` into role-specific refs:
     - `planner_model`
     - `worker_model`
     - `judge_model`
     - `reducer_model`
     - `subagent_model`
   - Default unset roles to the worker model.
   - Persist all role refs for resume/replay.

4. Provider adapters
   - Keep OpenAI as the first built-in provider.
   - Add at least one second provider adapter to prove the contract.
   - Prefer adapters that can support structured output for `GraphSpec` and
     iteration decisions.
   - Add conformance tests for provider lookup, credential checks, chat calls,
     and structured-output calls.

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
