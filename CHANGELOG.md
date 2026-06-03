# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While
pre-1.0 (`0.x`), the public API may change between minor versions.

## [Unreleased]

### Changed
- Removed `mock_document_extract` from the default tool allowlist (it echoed
  empty content, so a planner that picked it for a retrieval/compare task
  produced a dead-end run); retrieval now routes to `web_search`. Surfaced by
  the model-comparison eval — see `docs/evals/model-comparison-2026-06.md`.

### Added
- `RunResult.usage` (exact `TokenUsage` — input/output/total + per-model
  breakdown, from the providers' own counts via LangChain's usage callback)
  and opt-in `RunResult.cost` via an `EngineConfig(pricing=...)` book (keyed by
  model alias; matches dated snapshots by prefix). No price table is shipped.
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
