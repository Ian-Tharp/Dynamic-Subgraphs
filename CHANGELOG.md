# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While
pre-1.0 (`0.x`), the public API may change between minor versions.

## [Unreleased]

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
