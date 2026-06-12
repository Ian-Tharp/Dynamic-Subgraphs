# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While
pre-1.0 (`0.x`), the public API may change between minor versions.

## [Unreleased]

## [0.5.0] — 2026-06-11

This release completes the **eval/value layer** (Slice 7) and ships the **3-arm
benchmark harness** that measures the dynamic-topology thesis: scored, persisted,
comparable runs — and the machinery to test LLM-invented topologies against
hand-authored baselines under a pre-registered decision rule. Everything is
**off by default**: nothing scores a run until an `EvalGate` is configured.

### Upgrade notes (behavior changes since 0.4.0)
- **`Recording.all()` (and `record=True`) now selects the new `eval.json`
  artifact.** Harmless without a gate (nothing is written); with
  `EngineConfig(eval_gate=...)` set, scored runs persist `runs/<id>/eval.json`.
- **`capabilities()`** now reports `eval_gates` and the new `"evaluated"`
  recording preset.

### Added
- **Eval persistence:** `Artifact.EVAL` + `Recording.evaluated()` preset;
  `FileRecorder.record_eval()` / `load_eval()`; `list_runs()` summaries carry
  `quality` / `value_per_ktok` / `origin` / `deterministic` /
  `quality_floor_met` when a run was scored.
- **Engine wiring:** `EngineConfig(eval_gate=..., eval_tags=...)`;
  `run(..., task_id=, origin=, reference=)` benchmark tags;
  `RunResult.eval` (and in `to_dict()`). A failing gate never fails the run
  (breadcrumb-logged, mirrors best-effort recording).
- **Corpus reader:** `EvalCorpus` (load / group by task / group by topology)
  and `OriginComparison` — per-run paired Pareto comparison with quality-floor
  gating; refuses mismatched `rubric_id` / `applicable_dimensions` pairs.
- **Benchmark fixed-arm seam:** `RunConfig.fixed_spec` and
  `run(..., fixed_spec=...)` — run a hand-authored, registry-validated
  `GraphSpec` on the real LLM-worker path (validation/governance still
  enforced; an invalid fixed spec is rejected).
- **Benchmark harness:** `dynamic_subgraphs.bench` — `BenchTask`,
  `RouterLibrary`, `fill_spec` (literal `{prompt}` substitution into authored
  specs' instructions and tool args), `run_benchmark` (task × arm × repeat,
  one pinned worker model), `ArmRunRow` / `BenchReport`, pilot CV statistics,
  and `BenchIntegrityError` — a successful run that yields no `EvalResult`
  aborts the benchmark rather than booking corrupt quality-0 evidence;
  bench ids are single-use.
- **Paired-mode scoring:** `DeterministicEvalGate(grounding_applicability=
  "reference_only")` — grounding applicability becomes a task property so
  `applicable_dimensions` are identical across benchmark arms.
- **Plans-as-data baselines:** reviewed router library
  (`docs/evals/router-library-v1/`) + single-fixed sanity graph, all
  registry-validated and offline-executable; pilot task pack
  (`docs/evals/bench-tasks-pilot-v1.jsonl`) with de-echoed, reference-anchored
  checklists.
- **Pilot results + pre-registered rule:**
  `docs/evals/bench-dynamic-vs-fixed-2026-06.md` — completed 90-run pilot and
  the frozen main-run decision rule (registered before main-run spend).

### Fixed
- `DeterministicEvalGate` web-search detection now recognizes the contract
  `tool_name` param key (legacy `tool` / `name` keys remain as fallbacks).

## [0.4.0] — 2026-06-05

This release closes the concurrent-sibling `spawn_subgraph` budget TOCTOU and lands a
broad **maintainability & professionalism pass**: documentation reconciliation, a single
source of truth for the run-status vocabulary, a common error root, governance test
hardening, and HTTP-edge hardening. The deterministic eval scorer is now reachable from
the top-level facade.

### Upgrade notes (behavior changes since 0.3.0)
- **`DeterministicEvalGate` is now importable from the top level** —
  `from dynamic_subgraphs import DeterministicEvalGate, build_deterministic_eval_gate`
  (previously only via `dynamic_subgraphs.eval`). Still off by default.
- **The optional HTTP API (`api` extra) is stricter.** A request `prompt` is now bounded
  (`max_length`; a `422` over the limit); a malformed `run_id` returns **400** (was an
  unhandled `500`); run ids that are bare dot segments (`.` / `..`) are rejected; and the
  served OpenAPI `version` tracks the installed package version (was hardcoded `1.0.0`).
  Typical clients are unaffected.

### Added
- `DeterministicEvalGate` and `build_deterministic_eval_gate` are re-exported from the
  package facade (`dynamic_subgraphs.__all__`), so the whole eval surface is reachable
  from one import.

### Fixed
- Closed the concurrent-sibling `spawn_subgraph` budget TOCTOU (the deferred 0.2.0
  known limitation): two spawns scheduled in one superstep no longer each read the
  same pre-merge counters and over-allocate their children. A shared per-graph
  `BudgetLedger` reserves the remaining node/LLM allowance under a lock (refunding
  the unused portion), so concurrent siblings' grants always sum to within the host
  budget — fail-closed, never overspending. Allocation is first-come-greedy under
  contention (the loser fails closed); sequential spawns share the budget across
  supersteps.
- API: a malformed `run_id` returns `400` instead of an unhandled `500`; the run-id
  guard refuses path-traversal dot segments; the SSE trace stream no longer crashes on a
  quiet job (`queue.Empty`) or leaks a subscriber queue when a client disconnects.
- The release workflow import-smokes the built wheel in a clean venv before publishing.
- Governance test coverage: the wall-clock timeout and the concurrent-spawn budget guard
  now have behavioral tests that pin the real contract (the prior assertions were too
  weak to catch their own regression).

### Changed
- The API derives its OpenAPI `version` from package metadata and uses a `lifespan`
  handler instead of the deprecated `on_event` shutdown hook.
- Internal hardening (no public-API change): the run-status vocabulary is a single
  `RunStatus` enum; the engine's errors share a `DynamicSubgraphsError` root (stdlib
  bases preserved); `app.policy` is type-checked strictly via one pyproject-owned mypy
  scope; the shared response rendering, `parallel_map` output key, and recorder decision
  serializer are de-duplicated; best-effort recorder swallows now log a breadcrumb.

### Documentation
- Reconciled `ARCHITECTURE.md` (now documents the governance + eval layers), the docs
  index (lists only shipped docs), `CONTRIBUTING.md` (branch off `develop`), and several
  stale module headers and docstrings that predated their current implementations.

## [0.3.0] — 2026-06-05

The headline of this release is the **start of the eval/value layer**: a
deterministic, token-free run scorer (`DeterministicEvalGate`) that turns a
completed run into a comparable, queryable `EvalResult`. It also adds **safe
planner steering** (`EngineConfig.planner_guidance`) and tightens **fan-out
budget enforcement** so a `parallel_map` can no longer overspend the host's
LLM-call budget.

### Upgrade notes (behavior changes since 0.2.0)
- **`parallel_map` fan-out now respects `max_llm_calls`.** A fan-out of `llm_call`
  workers that would push the run past its granted LLM-call budget now halts
  fail-closed (`LlmCallBudgetExceeded`) at dispatch — previously only the fan-out
  *width* (`max_fanout`) was bounded, so a wide fan-out could overspend. Typical
  plans are unaffected; raise `EngineConfig(policy=ExecutionPolicy(max_llm_calls=...))`
  if you intend large `llm_call` fan-outs.

### Added
- `DeterministicEvalGate` — the structural eval scorer (Slice 7). Grades a
  completed run into a typed `EvalResult` across four deterministic, token-free
  dimensions: plan validity (registry re-validation + budget adherence),
  grounding (did a `web_search` actually produce results when the task called
  for it; inapplicable runs drop out rather than dilute the score),
  reference-anchored goal completion (checklist coverage, with a labelled
  low-confidence keyword heuristic when no `EvalReference` is supplied), and a
  token-parsimony cost proxy. Emits a shape-aware `topology_signature` (a branch
  DAG never collides with a linear chain of the same kinds) plus a separate
  `instruction_sha256`. Scoring is byte-stable across re-runs. Off by default and
  not yet wired into the engine — recorder persistence (`eval.json`) and
  `EngineConfig.eval_gate` land in the following slices.
- `EngineConfig(planner_guidance=...)` — optional domain steering appended to the
  planner's system prompt, with the GraphSpec contract preserved. A safe way to
  bias planning (e.g. "prefer `parallel_map` over deep recursion") without owning
  the whole prompt. Full-prompt replacement remains on the internal
  `LLMPlanner(system_prompt=...)` hook; a unified `PromptOverrides` surface (incl.
  the eval rubric) is a planned follow-up.

### Fixed
- `parallel_map` fan-out is now debited against the host `max_llm_calls` budget at
  dispatch and halts fail-closed (`LlmCallBudgetExceeded`) when a within-width-cap
  fan-out would overrun the granted LLM-call budget. Previously the fan-out *width*
  was capped (`max_fanout`) but per-worker LLM spend was only counted after the
  fact, so a wide fan-out of `llm_call` workers could exceed `max_llm_calls`.
  `tool_call` fan-outs are unaffected (they spend no LLM calls).

## [0.2.0] — 2026-06-04

The headline of this release is **governance that's actually enforced**: the
host owns the limits, and a planner can no longer grant itself more than the
host allows. It also adds a plan-repair loop, exact token usage + auto cost, the
model-agnostic SDK facade, opt-in recording, and the typed foundation of an
eval/value layer.

### Upgrade notes (behavior changes since 0.1.0)
- **Host budgets are enforced by default.** A plan that self-grants beyond the
  default ceilings (`max_nodes=12`, `max_llm_calls=8`, `max_depth=2`) — or names
  a disallowed tool/subagent/node kind under a restrictive `ExecutionPolicy` —
  is now capped or rejected. Typical plans are unaffected; set a stricter (or
  looser) `EngineConfig(policy=ExecutionPolicy(...))` to tune the envelope.
- **Plan-repair is on by default** (`max_plan_attempts=2`): a recoverable
  validation failure triggers one re-plan with the issues + host limits fed
  back (an extra planner call). Set `max_plan_attempts=1` for the old
  strict block-and-report behavior.
- **Validation is stricter:** a node input must be produced by an actual
  *ancestor* (a sibling-branch value is no longer accepted by ordering luck),
  and reserved/collision-prone node ids (`START`/`END`, the `__pm_` marker,
  non-identifier strings) are rejected. Previously-accepted-but-invalid specs
  now fail with a clear issue.

### Added
- **Host-owned governance** — `EngineConfig(policy=ExecutionPolicy(...))`,
  enforced end-to-end (root run and every nested `spawn_subgraph` child):
  budgets capped at `min(host, planner request)`; tool/subagent/node-kind
  allow-sets as the host ∩ registry intersection; a `parallel_map` fan-out cap
  (`GraphBudget.max_fanout`, default 64); nested budgets composed against the
  parent's *remaining* allowance; and a wall-clock bound (`max_wall_seconds`)
  that abandons a run/hung runner. `RunResult.effective_budget` surfaces the
  granted-vs-requested budget. `ExecutionPolicy` is exported from the facade.
- **Plan-repair loop** — `EngineConfig(max_plan_attempts=...)`;
  `RunResult.plan_attempts` reports how many planner attempts a run took.
- **Eval/value layer foundation** (types only, off until an `EvalGate` is
  configured) — `dynamic_subgraphs.eval` with `EvalGate`, the persisted
  `EvalResult` (+ `ScoreComponent`, `RunFingerprint`, `EvalReference`,
  `EvalTags`, `EvalContext`) and the `value_per_ktok`/`value_per_usd` axes.
- **Exact token usage + automatic cost** — `RunResult.usage` (provider-reported
  counts, always populated) and `RunResult.cost` (auto-computed with the `cost`
  extra via LiteLLM's maintained price map; `EngineConfig(pricing=...)`
  overrides). Both surfaced in `to_dict()`.
- **Model-agnostic SDK facade** — `DynamicSubgraphs`, `EngineConfig`, `Model`,
  `RunResult`, `Recording`/`Artifact`, `capabilities()`; OpenAI, Anthropic, and
  local Ollama / LM Studio (any OpenAI-compatible endpoint) with per-role model
  selection.
- **Granular, opt-in run recording** (`Recording` presets + per-artifact
  selection); the default writes no files.
- `DynamicSubgraphs.capabilities()` lists `node_kinds`; `NODE_KINDS` is a runtime
  tuple sourced from the `NodeKind` enum, with guard tests so the agent-facing
  surface can't drift from the compiler.
- `py.typed` (PEP 561) and optional extras (`api`, `openai`, `anthropic`,
  `ollama`, `cost`, `all`).

### Changed
- Removed `mock_document_extract` from the default tool allowlist (it echoed
  empty content and dead-ended retrieval/compare plans); retrieval routes to
  `web_search`.
- `replay()` re-validates the recorded spec against the **current** policy
  before re-executing — a spec recorded under looser limits no longer replays
  under a tightened policy.

### Fixed
- Validator input-provenance now follows real ancestry; reserved/malformed node
  ids are rejected (see *Upgrade notes*).
- `app/policy.py` is type-checked in CI (mypy) alongside the public package.

### Documentation
- README reframed around the value proposition (governed, auditable runtime),
  with the full node-kind vocabulary, a "When (not) to reach for it" section,
  and a governance + plan-repair guide; recipes for policy and repair.
- `docs/api-stability.md` — the project's opinionated API-stability & change
  policy (public-API definition, SemVer / 0.x rule, deprecation, CI guards).
- `docs/recipes.md` (tested-model + latency tables), `docs/evals/` (a
  gpt-5.4-nano vs claude-haiku-4-5 comparison), and a project wiki.

### Known limitations
- Concurrent sibling `spawn_subgraph` nodes can momentarily read the same
  pre-merge budget counter (a TOCTOU bounded by the depth ceiling); a
  reserve-and-refund fix is planned.
- The eval layer ships types only — no scorer is wired yet.

## [0.1.0]

- Initial governed dynamic-graph runtime: Supervisor, registry, compiler,
  runtime, recording, and a thin FastAPI surface.
