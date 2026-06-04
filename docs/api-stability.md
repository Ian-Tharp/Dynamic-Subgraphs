# API Stability & Change Policy

This is the project's **opinionated standard** for evolving Dynamic Subgraphs
without breaking the people who depend on it. It is deliberately prescriptive —
when a question comes up ("is this a breaking change? where does it go in the
changelog? can I rename this?"), the answer should be here.

It is grounded in current Python community practice:
[Semantic Versioning 2.0.0](https://semver.org/),
the [Python Packaging versioning guide](https://packaging.python.org/en/latest/discussions/versioning/),
[PEP 8](https://peps.python.org/pep-0008/) (public interfaces / `__all__`),
[Real Python — public API surface](https://realpython.com/ref/best-practices/public-api-surface/),
[PEP 702](https://peps.python.org/pep-0702/) (`@deprecated`),
[PEP 561](https://peps.python.org/pep-0561/) (`py.typed`),
and [Keep a Changelog](https://keepachangelog.com/).

---

## 1. What counts as the public API

SemVer only means something if "the public API" is **explicit**. For this
project the supported surface is, and only is:

- **Everything exported from `dynamic_subgraphs/__init__.py`'s `__all__`** — the
  facade: `DynamicSubgraphs`, `EngineConfig`, `Model`, `RunResult`, `Recording`/
  `Artifact`, the eval types, `capabilities()`, etc.
- The **documented behavior** of those symbols (arguments, defaults, return
  shapes, `RunResult` fields, `capabilities()` keys).
- The **inline types** of that surface — we ship [`py.typed`](https://peps.python.org/pep-0561/),
  so type signatures are part of the contract.

**Everything else is internal and carries no stability guarantee** — it may
change in any release:

- the entire `app.*` package (the engine implementation),
- any name prefixed with a single underscore,
- anything not listed in a module's `__all__`.

Rules that follow from this:

- **Keep the public surface small.** Per [PEP 8](https://peps.python.org/pep-0008/)
  and [Real Python](https://realpython.com/ref/best-practices/public-api-surface/),
  a small, well-documented API is easier to keep stable than a large, leaky one.
  Adding a name to a public `__all__` is a deliberate act (it now has to be
  supported), reviewed like any other API decision.
- **Internal interfaces stay underscore-prefixed even when `__all__` is set** —
  belt and suspenders.
- If an adopter is importing from `app.*`, that is a signal we are missing a
  facade seam — fix it by exposing a supported entry point, not by freezing
  `app.*`.

## 2. Versioning (SemVer, with the 0.x rule)

We follow [SemVer 2.0.0](https://semver.org/): `MAJOR.MINOR.PATCH` — MAJOR for
incompatible public-API changes, MINOR for backward-compatible additions, PATCH
for backward-compatible fixes.

**While we are `0.x` (pre-1.0):**

- A **breaking** change to the public API bumps the **minor** (`0.2.0 → 0.3.0`).
- An **additive** feature or a **fix** bumps the **patch** (`0.2.0 → 0.2.1`).
- This is the convention the
  [packaging guide](https://packaging.python.org/en/latest/discussions/versioning/)
  describes for 0.x, and it is stated in the `CHANGELOG` header.

**At 1.0** we commit to the public API: breaking changes require a MAJOR bump and
a deprecation cycle (§4). Do not reach 1.0 until the surface in §1 is one we are
willing to support under that promise.

The version lives in `pyproject.toml` and **must match the release tag** — a
release is `git tag vX.Y.Z && git push origin vX.Y.Z` (that tag, not a merge to
`main`, triggers the PyPI publish).

## 3. Change discipline — additive by default

This is the core of "fewer breaking changes." Most evolution can be done without
breaking anyone if you follow these:

- **New configuration is an optional field with a safe default that preserves
  prior behavior.** Precedent in this repo: `EngineConfig(recording=...)`
  defaults to off, `eval_gate=None`, and the planned `ExecutionPolicy()` default
  is permissive-but-bounded so a no-policy run behaves exactly as before.
- **New behavior is opt-in (off by default).** Recording, eval scoring, and
  policy enforcement must leave an unconfigured run unchanged.
- **Prefer keyword-only parameters for new arguments** (`def run(self, prompt, *,
  new_thing=None)`), so existing positional call sites never break.
- **Widen, don't narrow.** Accept a superset of inputs; return a compatible
  superset of outputs. Adding a field to a result object is safe; removing,
  renaming, or retyping one is breaking.
- **Version your serialized formats.** On-disk/JSON artifacts carry a
  `schema_version` (`GraphSpec`, `EvalResult`); readers tolerate older versions,
  and the version bumps only on an incompatible change. A new artifact must land
  *with* its producer and docs (enforced by the `Artifact ⇔ producer ⇔ recipes`
  guard tests).
- **Don't change the meaning of an existing default.** Changing what a default
  value *does* is as breaking as changing a signature.

### Validation tightening is a fix, not a silent break

Making the validator reject input that was *previously accepted but was never
actually valid* (e.g. the ancestor-provenance fix, or reserved node-id
rejection) is a **bug fix**, recorded under `### Fixed`/`### Changed` — not an
additive feature. It can change behavior for callers who relied on the bug, so:
call it out explicitly in the changelog, and verify against **real planner
output**, not just hand-written test specs, before merging (see §6).

## 4. Deprecation policy (when a break is truly necessary)

When you cannot make a change additively, you deprecate — you do not just delete.

1. **Keep the old path working** alongside the new one.
2. **Mark it** with `typing_extensions.deprecated("use X instead")`
   ([PEP 702](https://peps.python.org/pep-0702/)) — this gives both a
   static-checker warning *and* a runtime `DeprecationWarning`. Use a correct
   `stacklevel` so the warning points at the caller. (Add `typing_extensions` as
   a dependency the first time this is needed; it covers Python 3.11–3.12 where
   `warnings.deprecated` isn't yet available.)
3. **Document it** under `### Deprecated` in the changelog, with the replacement
   and the planned removal version.
4. **Removal window:** not before the next **minor** while `0.x` (at least one
   release of overlap), and only at a **MAJOR** once we are `1.0+`.

> ⚠️ A runtime `DeprecationWarning` is **filtered out by default** in Python, so
> many users never see it
> ([Seth Larson](https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries)).
> Treat the **changelog + release notes + docs as the primary channel** for
> announcing a deprecation; the warning is a secondary nudge, not the
> announcement.

## 5. Changelog discipline

We follow [Keep a Changelog](https://keepachangelog.com/). Every user-visible
change gets an entry under `[Unreleased]` in the right section:

`Added` · `Changed` · `Deprecated` · `Removed` · `Fixed` · `Security`
(plus a project-local `Documentation` section for docs-only changes).

- A **breaking** change must be unmistakable: put it under `Changed`/`Removed`
  with a **`BREAKING:`** prefix, and state the affected symbol, what changed,
  why, and the migration. (At 1.0+ we will keep a dedicated `Breaking` section.)
- Reference the design/spec doc when one exists.
- One entry per change; write it for a *reader deciding whether an upgrade is
  safe*, not for yourself.

## 6. The standard is enforced, not just written

Prose policies rot; these are wired into CI so they can't:

- **ruff** (lint + format) and **mypy** (strict on the public package) gate every
  PR; the public surface stays typed.
- **pytest** matrix across OSes and Python 3.11–3.13; a public-API addition
  ships with a test and a changelog entry.
- **Drift / round-trip guards already in the suite** are part of this policy and
  should grow with it: the `node_kinds`/README enumeration guards, the
  `Artifact ⇔ producer ⇔ recipes` guard, the `TokenUsage` re-export test,
  old-artifact round-trip tests, and the `MAX_DEPTH_CEILING` re-export test.
- **Behavior changes are verified against real model output** (a gated live
  run), not only mocks, before merge — mocks can't tell you a real planner now
  trips a tightened validator.

When you add to the public API, add a guard that fails if it silently changes.

## 7. Commits & PRs

- **Conventional-commit prefixes:** `feat`, `fix`, `docs`, `refactor`, `test`,
  `chore`, `ci`, `build` — with an optional scope, e.g. `feat(policy): ...`.
- **Small, single-purpose PRs**, ordered so no intermediate merge leaves the repo
  in a worse state (e.g. governance enforcement lands with its wiring, never
  half-wired).
- The PR description states **whether there is a behavior change** and, if so,
  the migration. PRs into `develop`/`main` go through review + CI (branch
  protection); only maintainers merge.

---

### TL;DR

Public = the facade's `__all__` + its types; everything else is internal.
Add optional, defaulted, opt-in — never remove, rename, or re-mean without a
deprecation cycle. SemVer (minor = break while 0.x). Every change gets a
correctly-sectioned changelog entry, and the rules are enforced by CI guards and
a real-model check, not by good intentions.
