# Contributing to Dynamic Subgraphs

Thanks for your interest in contributing! This project is a governed
dynamic-graph runtime / SDK. Contributions of all kinds are welcome — bug
reports, docs, tests, and features.

> **Before changing public API or behavior, read
> [`docs/api-stability.md`](./docs/api-stability.md)** — the project's standard
> for what counts as public, SemVer, deprecation, and keeping changes
> non-breaking (additive, opt-in, defaulted). It's prescriptive on purpose.

## Dev setup

We use [`uv`](https://docs.astral.sh/uv/). The dev environment installs all
optional extras (provider SDKs + the API surface) plus the dev tooling:

```bash
uv sync --all-extras
```

> If your checkout lives on a cloud-synced folder (OneDrive/Dropbox), set
> `UV_LINK_MODE=copy` to avoid hardlink errors. This is local-only — never set
> it in CI.

## The checks (must pass before a PR merges)

```bash
uv run ruff format .            # format
uv run ruff check .             # lint
uv run mypy dynamic_subgraphs   # type-check the public package (strict)
uv run pytest                   # tests (the 6 integration tests auto-skip)
```

CI runs the same on every PR (Linux + Windows, Python 3.11/3.12/3.13).
Install the git hooks to run format/lint/type-check automatically:

```bash
uv run pre-commit install
```

## Tests

- The default `pytest` run is **offline and deterministic** — no API keys or
  network. The 6 integration tests in `tests/test_integration_api.py` make real
  LLM/web calls and are skipped unless you opt in:
  ```bash
  # PowerShell
  $env:DS_RUN_INTEGRATION="1"; uv run pytest tests/test_integration_api.py
  ```
  They're also tagged `@pytest.mark.integration`, so `pytest -m "not integration"`
  excludes them explicitly.
- Add tests for any new behavior. Keep non-integration tests free of real
  network/LLM calls (use mocks / `monkeypatch`, as existing tests do).

## Conventions

- **Branches:** branch off `develop` (the integration branch) and open your PR
  against `develop`. `develop` and `main` are protected — both require a green CI
  run and a maintainer merge, so never push to either directly. `main` tracks the
  last release; maintainers fast-forward `develop` into it at release time.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- **Architecture:** respect the layered boundaries in
  [`ARCHITECTURE.md`](./ARCHITECTURE.md) — LangGraph types stay behind
  `compiler/`/`runtime/`; the planner emits *plans*, never code.
- **Public API:** the importable surface is `dynamic_subgraphs`; keep it typed
  (it ships `py.typed` and is mypy-strict).

## Releasing (maintainers)

Update [`CHANGELOG.md`](./CHANGELOG.md), bump the version in `pyproject.toml`,
tag `vX.Y.Z`, and let the release workflow publish to PyPI.
