# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While
pre-1.0 (`0.x`), the public API may change between minor versions.

## [Unreleased]

### Changed
- **Runtime governance guards — the host policy is now enforced end-to-end, not
  just on numeric budgets.** At validation (root run *and* every nested
  `spawn_subgraph` child): **allow-sets** (tools / subagents / node kinds) are
  the host ∩ registry intersection — a plan naming a disallowed capability is
  rejected (`tool_not_allowlisted` / `node_kind_not_allowed`); **nested budgets
  compose** — a child's node/LLM/depth budget is the parent's *remaining*
  allowance, so a nest can't outspend the root, and nesting depth is capped at
  the tighter of the host `max_depth` and the hard rail. At runtime:
  `parallel_map` **fan-out** over more than the granted `max_fanout` halts
  fail-closed before any work fires; a run that outruns `max_wall_seconds` is
  **abandoned** (daemon invoke thread + bounded join, so a hung runner can't
  block forever). Verified with offline tests and a live e2e.

### Added
- `GraphBudget.max_fanout` (default 64) — the planner may request a fan-out
  limit; the effective cap is `min(host, request)`, stamped and enforced.

### Fixed
- `replay()` re-validates the recorded spec against the **current** host policy
  before re-executing, so a spec recorded under looser limits no longer replays
  if the host has since tightened them (was trusted verbatim).
- `app/policy.py` is now type-checked in CI (mypy) alongside the public package;
  removed stale `type: ignore`s. (Broader `app/*` typing is a follow-up.)

### Documentation
- README reframed around the actual value proposition — a *governed, auditable*
  runtime for LLM-generated workflows (validated plans, allowlisted vocabulary,
  budget-capped recursion, replayable runs), rather than leading with "the model
  invents the topology".
- Added the missing `branch` node kind to the README's vocabulary list (the
  registry ships nine kinds; the README listed eight) and a dynamic-routing diagram
  that shows it.
- Added a "When (not) to reach for it" section — when a fixed graph or a plain
  tool loop is the better tool, stated plainly.
- Clarified in the README lede that recording is **opt-in** — the public SDK
  defaults to in-memory execution and writes no files (removed wording that
  implied every run is persisted).
- Pointed the PyPI `Documentation` URL at the docs on the active branch.
- Added `docs/api-stability.md` — the project's opinionated API stability &
  change policy (public-API definition, SemVer / 0.x rule, additive-by-default
  discipline, deprecation policy, and the CI guards that enforce it). Linked
  from the README and CONTRIBUTING.

### Changed
- **Plan-repair loop (default ON).** When a plan is rejected by the validator
  for a *recoverable* reason (budget overrun, dangling edge, missing input,
  bad branch, reserved id…), the supervisor now feeds the issues + the host
  limits back into a re-plan, up to `EngineConfig(max_plan_attempts=...)`
  planner attempts. **Default is 2** (repair once) — a behavior change that
  makes plans "just work" more often; set `max_plan_attempts=1` for strict
  block-and-report on the first failure. Each attempt is another planner call;
  un-recoverable issues never retry. `RunResult.plan_attempts` reports how many
  it took. Verified live (a real over-budget plan recovered on the repair pass).
- **Host-owned budget enforcement.** Budget validation now enforces the
  host-owned `ExecutionPolicy` (`EngineConfig(policy=...)`), not the planner's
  self-declared `GraphBudget`: the effective limit per field is
  `min(host ceiling, planner request)`, and the granted budget is stamped onto
  the validated spec (so the recursion rail, nested-subgraph clamp, recording,
  and API all read the host-enforced limits) — at the root **and** every nested
  `spawn_subgraph` child. With no policy set, default ceilings apply
  (`max_nodes=12`, `max_llm_calls=8`, `max_depth=2`), matching the historical
  `GraphBudget` defaults. *Behavior change:* a plan that self-grants more than
  the host allows is now capped or rejected (`budget_exceeded`) — a planner can
  no longer grant itself a larger budget. Verified against real planner output.
  Allowlist intersection (tools/subagents/kinds) and runtime guards
  (spend-ledger, fan-out, wall-clock, reserve-and-refund nesting) land next.
- Removed `mock_document_extract` from the default tool allowlist (it echoed
  empty content, so a planner that picked it for a retrieval/compare task
  produced a dead-end run); retrieval now routes to `web_search`. Surfaced by
  the model-comparison eval — see `docs/evals/model-comparison-2026-06.md`.

### Fixed
- Validator input-provenance: a node input is satisfiable only when produced by
  an actual **ancestor** of the consuming node, not by any earlier-visited node.
  A value produced solely by a sibling branch (with no edge ordering it first)
  is now correctly rejected instead of accepted by topological luck.
- Validator now rejects reserved / collision-prone node ids — `START`/`END` and
  ids containing the `__pm_` marker the compiler derives parallel_map internal
  node names from — plus malformed ids (must match `[A-Za-z0-9_-]+`). Prevents a
  planner-chosen id from shadowing a generated node.

### Added
- `EngineConfig(max_plan_attempts=...)` — bounds the plan-repair loop (see
  *Changed*); default 2, set 1 for strict block-and-report.
- `RunResult.plan_attempts` — how many planner attempts a run took (1 unless the
  repair loop re-planned), included in `to_dict()`.
- `EngineConfig(policy=ExecutionPolicy(...))` — host-owned budget governance,
  now **wired and enforced** (see *Changed*). `ExecutionPolicy` is exported from
  the public `dynamic_subgraphs` facade.
- `RunResult.effective_budget` — the host-*granted* budget for a run (the
  planner's request capped by the policy; equals `plan.budget`), included in
  `to_dict()`. Lets a caller see granted-vs-requested.
- ExecutionPolicy foundation (host-owned governance, PR1 — types + resolver):
  `app.policy` with `ExecutionPolicy`, `EffectiveBudget`, `RemainingBudget`, and
  the pure `resolve_effective_budget` (effective budget =
  `min(planner, host, parent-remaining)` per field; vocabulary = host ∩
  registry; a child resolution refuses to fall back to a full budget).
  `MAX_DEPTH_CEILING` now lives in `app.policy` (re-exported from the validator).
  Design: `docs/specs/2026-06-04-execution-policy-design.md`.
- Eval/value layer foundation (Slice 7, PR1 — types only, OFF by default):
  `dynamic_subgraphs.eval` with the `EvalGate` protocol, the persisted
  `EvalResult` (+ `ScoreComponent`, `RunFingerprint`, `EvalReference`,
  `EvalTags`, `EvalContext`) and the `value_per_ktok`/`value_per_usd` axes.
  Nothing scores a run until an `EvalGate` is configured. Design:
  `docs/specs/2026-06-03-eval-value-layer-design.md`.
- `DynamicSubgraphs.capabilities()` now lists `node_kinds` (the registry's full
  node vocabulary), and `dynamic_subgraphs.types.NODE_KINDS` exposes it as a
  runtime tuple — both sourced from the `NodeKind` enum so the agent-facing
  surface can't drift from what the compiler accepts. Guard tests assert the
  capabilities map and the README each enumerate every node kind (this drift
  shipped once — the README listed eight of nine kinds).
- `RunResult.usage` (exact `TokenUsage` — input/output/total + per-model
  breakdown, from the providers' own counts via LangChain's usage callback).
- `RunResult.cost` — computed automatically with the `cost` extra (LiteLLM's
  maintained price map; no prices to specify, no table we keep current).
  `EngineConfig(pricing=...)` still overrides per model / prices local models.
- `docs/evals/` — eval reports. First entry: gpt-5.4-nano vs claude-haiku-4-5
  e2e comparison (latency / tokens / cost / quality), traced via LangSmith.
- Public `dynamic_subgraphs` SDK facade: `DynamicSubgraphs`, `EngineConfig`,
  `Model`, `Recording`/`Artifact`, `RunResult`, `capabilities()`.
- Model-agnostic providers: OpenAI, Anthropic, and local Ollama / LM Studio
  (any OpenAI-compatible endpoint) with per-role model selection.
- Granular, opt-in run recording (`Recording` presets + per-artifact selection).
- `py.typed` marker (PEP 561) — the SDK ships inline types.
- Packaging: Apache-2.0 license, optional extras (`api`, `openai`, `anthropic`,
  `ollama`, `all`), slim core dependencies.
- Tooling: ruff (lint + format), mypy (strict on the public package), coverage,
  pre-commit, and GitHub Actions CI.
- Docs: `examples/` cookbook, `docs/recipes.md` (tested-model + latency tables).

## [0.1.0]

- Initial governed dynamic-graph runtime: Supervisor, registry, compiler,
  runtime, recording, and a thin FastAPI surface.
