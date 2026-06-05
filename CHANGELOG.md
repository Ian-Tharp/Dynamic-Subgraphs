# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While
pre-1.0 (`0.x`), the public API may change between minor versions.

## [Unreleased]

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
- `py.typed` (PEP 561), optional extras (`api`, `openai`, `anthropic`, `ollama`,
  `cost`, `all`), and `docs/specs/` design docs for the eval + policy work.

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
