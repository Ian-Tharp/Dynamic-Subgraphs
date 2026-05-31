# FastAPI Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing `Supervisor` over HTTP — create/run/inspect/list/resume/replay runs, run iterative chains, browse the registry — without leaking LangGraph types or changing runtime semantics.

**Architecture:** A thin `app/api/` package translates HTTP ↔ `Supervisor`. Every run executes in an in-process `JobStore` (thread pool); the request handler decides how long to wait (sync/async/auto). A shared `MemorySaver` checkpointer makes `resume` work across requests. Reads (`GET` endpoints) go through new `FileRecorder` helpers. SSE streams job lifecycle now, with a bus seam for per-node events later.

**Tech Stack:** FastAPI, uvicorn, Pydantic v2 (already used), `concurrent.futures.ThreadPoolExecutor`, LangGraph `MemorySaver` (already a dependency via langgraph), `httpx` + `TestClient` for tests.

**Spec:** `docs/superpowers/specs/2026-05-30-fastapi-layer-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `app/assembly.py` (new) | `RunConfig` + `build_supervisor()` — shared CLI/API supervisor wiring |
| `app/main.py` (modify) | call `build_supervisor()` instead of private helpers |
| `app/recording/recorder.py` (modify) | add read helpers: `list_runs`, `load_output`, `run_dir`, `artifact_path`, `exists` |
| `app/api/settings.py` (new) | `ApiSettings` from env + allowlists |
| `app/api/jobs.py` (new) | `JobStore`, `Job`, `JobState`, background execution + subscribe bus |
| `app/api/schemas.py` (new) | request/response Pydantic contract models |
| `app/api/errors.py` (new) | API exceptions + handlers + error envelope |
| `app/api/serialize.py` (new) | `SupervisorResult`/`IterativeSupervisorResult` → response dicts |
| `app/api/deps.py` (new) | DI: settings, recorder, job store, checkpointer, auth |
| `app/api/routers/health.py` (new) | `/healthz` |
| `app/api/routers/registry.py` (new) | `/registry` |
| `app/api/routers/runs.py` (new) | `/runs ...` (create, list, status, files, resume, replay, SSE) |
| `app/api/routers/chains.py` (new) | `/chains ...` |
| `app/api/app.py` (new) | `create_app()` factory wiring |
| `app/api/__init__.py` (modify) | export `create_app` |
| `app/api/__main__.py` (new) | `python -m app.api` → uvicorn |
| `pyproject.toml` (modify) | add fastapi, uvicorn, httpx(dev) |
| `tests/test_api_*.py` (new) | endpoint + JobStore + assembly tests |

**Naming locked across tasks** (use these exact names everywhere):
- `JobState` values: `queued`, `running`, `ok`, `failed`, `paused`. (No separate `recording` state — the supervisor records internally and we cannot observe that sub-step without instrumenting it; the bus seam allows adding it later.)
- `RunConfig(planner, model, strict_runners)`; `build_supervisor(config, *, recorder, checkpointer=None)`.
- `JobStore.create(run_id, kind)`, `.submit(job, fn)`, `.get(run_id)`, `.shutdown()`.
- `Job.set_state(state)`, `.complete(result, state)`, `.fail(message)`, `.wait(timeout)`, `.subscribe()`.
- Settings env prefix `DS_`.

---

## Task 1: Baseline commit on a feature branch

The repo currently has **zero commits** and ~40 untracked files. We need a baseline so per-task feature commits are diffable, and we must ensure secrets/artifacts are not committed.

**Files:**
- Modify/verify: `.gitignore`

- [ ] **Step 1: Inspect current ignore rules**

Run: `git status --short` and open `.gitignore`.

- [ ] **Step 2: Ensure secrets and artifacts are ignored**

Make sure `.gitignore` contains at least these lines (add any missing — do NOT remove existing entries):

```gitignore
.venv/
__pycache__/
*.pyc
.env
.pytest_cache/
runs/
```

Rationale: `.env` holds a real API key and must never be committed; `runs/` and `.venv/` are regenerable artifacts.

- [ ] **Step 3: Confirm .env is not staged**

Run: `git status --short` and verify `.env` does NOT appear (it should be ignored). If it appears, fix `.gitignore` before continuing.

- [ ] **Step 4: Create the feature branch**

Run:
```bash
git checkout -b feat/api-layer
```

- [ ] **Step 5: Baseline commit of the existing project**

```bash
git add -A
git status   # sanity check: no .env, no .venv/, no runs/
git commit -m "chore: baseline commit of dynamic-subgraphs before API layer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Expected: a first commit containing app/, tests/, docs/, pyproject.toml, etc. — but not `.env`, `.venv/`, or `runs/`.

---

## Task 2: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add runtime + dev dependencies via uv**

Run:
```bash
uv add fastapi "uvicorn[standard]"
uv add --dev httpx
```

Expected: `pyproject.toml` gains `fastapi` and `uvicorn[standard]` under `dependencies`, `httpx` under `[project.optional-dependencies] dev`; `uv.lock` updates; packages install into `.venv`.

- [ ] **Step 2: Verify imports resolve**

Run:
```bash
.venv/Scripts/python.exe -c "import fastapi, uvicorn, httpx; print(fastapi.__version__)"
```
Expected: prints a version, no ImportError.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add fastapi, uvicorn, httpx deps

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Shared supervisor builder (`app/assembly.py`)

Lift the runner-wiring out of `main.py` into one place both CLI and API use.

**Files:**
- Create: `app/assembly.py`
- Test: `tests/test_assembly.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_assembly.py
from __future__ import annotations

from pathlib import Path

from app.assembly import RunConfig, build_supervisor
from app.recording import FileRecorder
from app.supervisor import StaticPlanner, Supervisor


def test_build_supervisor_mock_returns_supervisor(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)

    supervisor = build_supervisor(config, recorder=recorder)

    assert isinstance(supervisor, Supervisor)


def test_build_supervisor_mock_runs_end_to_end(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)
    supervisor = build_supervisor(config, recorder=recorder)

    result = supervisor.run("compare two things", run_id="assembly-001")

    assert result.status == "ok"
    assert (tmp_path / "assembly-001" / "spec.json").exists()


def test_build_supervisor_mock_uses_static_planner(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    config = RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False)

    supervisor = build_supervisor(config, recorder=recorder)

    assert isinstance(supervisor._planner, StaticPlanner)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_assembly.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.assembly'`.

- [ ] **Step 3: Implement `app/assembly.py`**

```python
# app/assembly.py
"""Shared builder: turn a RunConfig into a wired Supervisor.

Used by both the CLI (`app/main.py`) and the HTTP API so they construct
identical supervisors. The mock path is token-free (StaticPlanner + mock/echo
runners); the openai path uses the real planner + grounded tools in strict mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from app.models import DynamicRunState, NodeKind
from app.recording import Recorder
from app.runtime import (
    FileArtifactSink,
    LangGraphExecutor,
    NodeRunner,
    build_grounded_tool_runner,
    build_openai_llm_runner,
    build_openai_reduce_runner,
    build_openai_spawn_subagent_runner,
    make_emit_artifact_runner,
)
from app.supervisor import (
    Planner,
    StaticPlanner,
    Supervisor,
    build_openai_planner,
)

_LLM_REDUCE_STRATEGIES = {"concat", "merge_dict", "llm_summarize"}


@dataclass(frozen=True)
class RunConfig:
    """Resolved per-run configuration shared by CLI and API."""

    planner: Literal["mock", "openai"]
    model: str
    strict_runners: bool


def mock_llm_runner(
    state: DynamicRunState,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Token-free stand-in: echoes instruction and visible upstream keys."""

    instruction = str(params["instruction"])
    upstream_keys = sorted(state.get("values", {}).keys())
    suffix = f" [seen: {', '.join(upstream_keys)}]" if upstream_keys else ""
    return {"result": f"<mock-llm>{instruction}{suffix}</mock-llm>"}


def _build_planner(config: RunConfig) -> Planner:
    if config.planner == "openai":
        return build_openai_planner(
            model=config.model,
            executable_reduce_strategies=_LLM_REDUCE_STRATEGIES,
        )
    from app.main import build_demo_spec

    return StaticPlanner(build_demo_spec())


def _build_runners(config: RunConfig, *, runs_dir: str) -> dict[NodeKind, NodeRunner]:
    emit_runner = make_emit_artifact_runner(FileArtifactSink(root_dir=runs_dir))
    runners: dict[NodeKind, NodeRunner] = {NodeKind.EMIT_ARTIFACT: emit_runner}

    if config.planner == "openai":
        runners[NodeKind.LLM_CALL] = build_openai_llm_runner(model=config.model)
        runners[NodeKind.REDUCE] = build_openai_reduce_runner(model=config.model)
        runners[NodeKind.SPAWN_SUBAGENT] = build_openai_spawn_subagent_runner(
            model=config.model
        )
        runners[NodeKind.TOOL_CALL] = build_grounded_tool_runner()
    else:
        runners[NodeKind.LLM_CALL] = mock_llm_runner

    return runners


def build_supervisor(
    config: RunConfig,
    *,
    recorder: Recorder,
    checkpointer: Any | None = None,
) -> Supervisor:
    """Construct a Supervisor wired for `config`.

    `recorder` is the persistence target. `checkpointer` (e.g. a shared
    MemorySaver) enables resume across calls for graphs with wait_for_event.
    """

    runs_dir = str(getattr(recorder, "root_dir", "runs"))
    planner = _build_planner(config)
    runners = _build_runners(config, runs_dir=runs_dir)
    executor = LangGraphExecutor(
        runners=runners,
        checkpointer=checkpointer,
        strict_runners=config.strict_runners,
    )
    return Supervisor(planner=planner, executor=executor, recorder=recorder)
```

Note: `_build_planner` imports `build_demo_spec` from `app/main.py` (kept there in Task 4). This avoids moving the demo spec; if you prefer, move `build_demo_spec` into `assembly.py` and import it from `main.py` instead — but keep ONE definition.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_assembly.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/assembly.py tests/test_assembly.py
git commit -m "feat: shared build_supervisor() for CLI and API

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Refactor `main.py` to use the shared builder

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Replace the planner/runner wiring in `main()` with `build_supervisor`**

In `app/main.py`, keep `build_demo_spec`, `render_supervisor_result`, `_parse_args`, and the UTF-8 stdout setup. Remove the private `_build_planner`, `_build_llm_call_runner`, `_build_reduce_runner`, `_build_spawn_subagent_runner`, `_build_tool_call_runner`, `_build_emit_artifact_runner`, and the `mock_llm_runner` (now in `assembly.py`). Rewrite `main()` body as:

```python
def main() -> None:
    load_dotenv()
    args = _parse_args()

    runs_dir = Path(__file__).resolve().parents[1] / "runs"

    config = RunConfig(
        planner="openai" if args.llm else "mock",
        model=args.model,
        strict_runners=args.llm,
    )
    recorder = FileRecorder(root_dir=runs_dir, overwrite=True)
    supervisor = build_supervisor(config, recorder=recorder)

    planner_label = f"LLMPlanner({args.model})" if args.llm else "StaticPlanner"
    runner_label = f"OpenAILlmRunner({args.model})" if args.llm else "mock_llm_runner"
    print(f"[demo] planner = {planner_label}")
    print(f"[demo] runner  = {runner_label}")
    print(f"[demo] runs    = {runs_dir}")
    print(f"[demo] prompt  = {args.prompt!r}")

    result = supervisor.run(args.prompt, run_id=args.run_id)
    render_supervisor_result(result)
```

Update the imports at the top of `main.py`: remove the now-unused runtime/supervisor runner imports; add:

```python
from app.assembly import RunConfig, build_supervisor
from app.recording import FileRecorder
```

Keep imports still used by `build_demo_spec` (`GraphSpec`, `NodeSpec`, `NodeKind`, `EdgeSpec`, `GraphBudget`) and `render_supervisor_result` (`SupervisorResult`).

- [ ] **Step 2: Verify the mock demo still runs**

Run: `.venv/Scripts/python.exe -m app.main "compare A and B"`
Expected: prints the demo banner and a `Status: ok` block; writes `runs/demo-run-001/`.

- [ ] **Step 3: Verify the full suite still passes**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all previously-passing tests still pass (222+ passed).

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "refactor: main.py uses shared build_supervisor()

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Recorder read helpers

**Files:**
- Modify: `app/recording/recorder.py`
- Test: `tests/test_recorder_reads.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recorder_reads.py
from __future__ import annotations

from pathlib import Path

import pytest

from app.assembly import RunConfig, build_supervisor
from app.recording import FileRecorder


def _seed_run(tmp_path: Path, run_id: str) -> FileRecorder:
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    supervisor = build_supervisor(
        RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False),
        recorder=recorder,
    )
    supervisor.run("seed", run_id=run_id)
    return recorder


def test_exists_true_after_run(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    assert recorder.exists("rec-001") is True


def test_exists_false_for_unknown(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path)
    assert recorder.exists("nope") is False


def test_list_runs_returns_seeded_ids(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    _seed_run(tmp_path, "rec-002")

    summaries = recorder.list_runs()

    ids = {s["run_id"] for s in summaries}
    assert {"rec-001", "rec-002"} <= ids
    sample = next(s for s in summaries if s["run_id"] == "rec-001")
    assert sample["status"] in {"ok", "failed", "paused"}
    assert isinstance(sample["nodes"], int)


def test_load_output_has_values(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    output = recorder.load_output("rec-001")
    assert "values" in output and "ok" in output


def test_load_output_missing_raises(tmp_path: Path) -> None:
    recorder = FileRecorder(root_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        recorder.load_output("ghost")


def test_artifact_path_rejects_traversal(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    with pytest.raises(ValueError):
        recorder.artifact_path("rec-001", "../escape.txt")


def test_run_dir_points_at_run(tmp_path: Path) -> None:
    recorder = _seed_run(tmp_path, "rec-001")
    assert recorder.run_dir("rec-001") == tmp_path / "rec-001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_recorder_reads.py -v`
Expected: FAIL with `AttributeError: 'FileRecorder' object has no attribute 'exists'`.

- [ ] **Step 3: Add read helpers to `FileRecorder`**

In `app/recording/recorder.py`, add these methods to the `FileRecorder` class (after `load_chain`). Reuse the existing module-level `_validate_run_id`.

```python
    def exists(self, run_id: str) -> bool:
        _validate_run_id(run_id)
        return (self._root / run_id).is_dir()

    def run_dir(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self._root / run_id

    def load_output(self, run_id: str) -> dict[str, Any]:
        _validate_run_id(run_id)
        output_path = self._root / run_id / "output.json"
        if not output_path.exists():
            raise FileNotFoundError(
                f"No output.json for run_id {run_id!r} at {output_path}"
            )
        return json.loads(output_path.read_text(encoding="utf-8"))

    def artifact_path(self, run_id: str, name: str) -> Path:
        _validate_run_id(run_id)
        if not name or not all(ch in _SAFE_RUN_ID_CHARS for ch in name):
            raise ValueError(
                f"artifact name contains unsafe characters: {name!r}"
            )
        return self._root / run_id / "artifacts" / name

    def list_runs(self) -> list[dict[str, Any]]:
        """Summaries of every recorded run directory (id, status, node count).

        Skips chain directories (those with chain.json but no spec.json).
        """
        if not self._root.exists():
            return []
        summaries: list[dict[str, Any]] = []
        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            spec_path = child / "spec.json"
            if not spec_path.exists():
                continue
            status = "unknown"
            output_path = child / "output.json"
            if output_path.exists():
                try:
                    out = json.loads(output_path.read_text(encoding="utf-8"))
                    if out.get("ok"):
                        status = "ok"
                    elif out.get("error"):
                        status = "failed"
                    else:
                        status = "failed" if out.get("errors") else "ok"
                except (json.JSONDecodeError, OSError):
                    status = "unknown"
            node_count = 0
            try:
                spec_raw = json.loads(spec_path.read_text(encoding="utf-8"))
                node_count = len(spec_raw.get("nodes", []))
            except (json.JSONDecodeError, OSError):
                node_count = 0
            summaries.append(
                {"run_id": child.name, "status": status, "nodes": node_count}
            )
        return summaries
```

`Any` and `json` are already imported at the top of the file; `_SAFE_RUN_ID_CHARS` already exists module-level.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_recorder_reads.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add app/recording/recorder.py tests/test_recorder_reads.py
git commit -m "feat: FileRecorder read helpers (list/load/exists/paths)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `ApiSettings`

**Files:**
- Create: `app/api/settings.py`
- Test: `tests/test_api_settings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_settings.py
from __future__ import annotations

from app.api.settings import ApiSettings


def test_defaults_are_mock_and_free() -> None:
    settings = ApiSettings.from_env({})
    assert settings.planner == "mock"
    assert settings.model == "gpt-5.4-nano"
    assert "gpt-5.4-nano" in settings.model_allowlist
    assert settings.api_key is None
    assert settings.auto_sync_seconds == 25
    assert settings.max_sync_seconds == 120


def test_env_overrides_are_parsed() -> None:
    env = {
        "DS_PLANNER": "openai",
        "DS_MODEL": "gpt-5.4-nano",
        "DS_MODEL_ALLOWLIST": "gpt-5.4-nano, gpt-5.4-mini",
        "DS_RUNS_DIR": "/tmp/runs",
        "DS_API_KEY": "secret",
        "DS_AUTO_SYNC_SECONDS": "5",
        "DS_MAX_SYNC_SECONDS": "30",
    }
    settings = ApiSettings.from_env(env)
    assert settings.planner == "openai"
    assert settings.model_allowlist == ("gpt-5.4-nano", "gpt-5.4-mini")
    assert settings.runs_dir == "/tmp/runs"
    assert settings.api_key == "secret"
    assert settings.auto_sync_seconds == 5
    assert settings.max_sync_seconds == 30


def test_model_allowed_check() -> None:
    settings = ApiSettings.from_env({"DS_MODEL_ALLOWLIST": "a,b"})
    assert settings.is_model_allowed("a") is True
    assert settings.is_model_allowed("zzz") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.settings'`.

- [ ] **Step 3: Implement `app/api/settings.py`**

```python
# app/api/settings.py
"""Environment-driven API settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

PlannerMode = Literal["mock", "openai"]


@dataclass(frozen=True)
class ApiSettings:
    planner: PlannerMode = "mock"
    model: str = "gpt-5.4-nano"
    model_allowlist: tuple[str, ...] = ("gpt-5.4-nano",)
    runs_dir: str = "runs"
    api_key: str | None = None
    auto_sync_seconds: int = 25
    max_sync_seconds: int = 120

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ApiSettings":
        env = os.environ if env is None else env
        planner = env.get("DS_PLANNER", "mock")
        if planner not in ("mock", "openai"):
            planner = "mock"
        allowlist_raw = env.get("DS_MODEL_ALLOWLIST", "gpt-5.4-nano")
        allowlist = tuple(
            item.strip() for item in allowlist_raw.split(",") if item.strip()
        )
        return cls(
            planner=planner,  # type: ignore[arg-type]
            model=env.get("DS_MODEL", "gpt-5.4-nano"),
            model_allowlist=allowlist or ("gpt-5.4-nano",),
            runs_dir=env.get("DS_RUNS_DIR", "runs"),
            api_key=env.get("DS_API_KEY") or None,
            auto_sync_seconds=int(env.get("DS_AUTO_SYNC_SECONDS", "25")),
            max_sync_seconds=int(env.get("DS_MAX_SYNC_SECONDS", "120")),
        )

    def is_model_allowed(self, model: str) -> bool:
        return model in self.model_allowlist
```

- [ ] **Step 4: Create `app/api/routers/__init__.py` (empty package marker)**

```python
# app/api/routers/__init__.py
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_settings.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add app/api/settings.py app/api/routers/__init__.py tests/test_api_settings.py
git commit -m "feat: ApiSettings env config

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `JobStore` + `Job`

**Files:**
- Create: `app/api/jobs.py`
- Test: `tests/test_api_jobs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_jobs.py
from __future__ import annotations

import time

import pytest

from app.api.jobs import Job, JobExistsError, JobState, JobStore


def test_create_then_get() -> None:
    store = JobStore(max_workers=2)
    job = store.create("j1", kind="run")
    assert store.get("j1") is job
    assert job.state == JobState.QUEUED
    store.shutdown()


def test_duplicate_create_raises() -> None:
    store = JobStore(max_workers=1)
    store.create("dup", kind="run")
    with pytest.raises(JobExistsError):
        store.create("dup", kind="run")
    store.shutdown()


def test_submit_runs_to_completion() -> None:
    store = JobStore(max_workers=2)
    job = store.create("j2", kind="run")

    def work(j: Job) -> None:
        j.set_state(JobState.RUNNING)
        j.complete(result={"value": 42}, state=JobState.OK)

    store.submit(job, work)
    assert job.wait(timeout=2.0) is True
    assert job.state == JobState.OK
    assert job.result == {"value": 42}
    store.shutdown()


def test_exception_in_work_marks_failed() -> None:
    store = JobStore(max_workers=1)
    job = store.create("j3", kind="run")

    def boom(j: Job) -> None:
        raise RuntimeError("kaboom")

    store.submit(job, boom)
    assert job.wait(timeout=2.0) is True
    assert job.state == JobState.FAILED
    assert "kaboom" in (job.error or "")
    store.shutdown()


def test_wait_times_out_while_running() -> None:
    store = JobStore(max_workers=1)
    job = store.create("j4", kind="run")

    def slow(j: Job) -> None:
        j.set_state(JobState.RUNNING)
        time.sleep(0.5)
        j.complete(result=None, state=JobState.OK)

    store.submit(job, slow)
    assert job.wait(timeout=0.05) is False
    assert job.wait(timeout=2.0) is True
    store.shutdown()


def test_subscribe_yields_terminal_status() -> None:
    store = JobStore(max_workers=1)
    job = store.create("j5", kind="run")

    def work(j: Job) -> None:
        j.set_state(JobState.RUNNING)
        j.complete(result=None, state=JobState.OK)

    store.submit(job, work)
    job.wait(timeout=2.0)

    queue = job.subscribe()
    seen = []
    while True:
        msg = queue.get(timeout=2.0)
        if msg["type"] == "__end__":
            break
        seen.append(msg)
    states = [m["state"] for m in seen if m["type"] == "status"]
    assert "ok" in states
    store.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.jobs'`.

- [ ] **Step 3: Implement `app/api/jobs.py`**

```python
# app/api/jobs.py
"""In-process job store: background execution + subscribe bus.

Every run/chain becomes a Job executed on a thread pool. The request handler
decides how long to wait (sync/async/auto). The subscribe bus feeds SSE today;
it is the seam where per-node events will publish later.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from queue import Queue
from typing import Any

_TERMINAL: frozenset[str] = frozenset({"ok", "failed", "paused"})


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    PAUSED = "paused"


class JobExistsError(Exception):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Job already exists: {run_id!r}")
        self.run_id = run_id


def _now() -> datetime:
    return datetime.now(UTC)


class Job:
    """Mutable, thread-safe handle for one background run/chain."""

    def __init__(self, run_id: str, kind: str) -> None:
        self.run_id = run_id
        self.kind = kind
        self.state: JobState = JobState.QUEUED
        self.submitted_at: datetime = _now()
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.result: Any = None
        self.error: str | None = None
        self.budget_wall_seconds: int | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._subscribers: list[Queue[dict[str, Any]]] = []

    def _publish(self, msg: dict[str, Any]) -> None:
        for q in self._subscribers:
            q.put(msg)

    def set_state(self, state: JobState) -> None:
        with self._lock:
            self.state = state
            if state == JobState.RUNNING and self.started_at is None:
                self.started_at = _now()
            self._publish({"type": "status", "state": state.value})

    def complete(self, *, result: Any, state: JobState) -> None:
        with self._lock:
            self.result = result
            self.state = state
            self.finished_at = _now()
            self._publish({"type": "status", "state": state.value})
            self._publish({"type": "__end__"})
            self._subscribers.clear()
            self._done.set()

    def fail(self, message: str) -> None:
        with self._lock:
            self.error = message
            self.state = JobState.FAILED
            self.finished_at = _now()
            self._publish({"type": "status", "state": JobState.FAILED.value})
            self._publish({"type": "__end__"})
            self._subscribers.clear()
            self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout=timeout)

    def is_terminal(self) -> bool:
        return self.state.value in _TERMINAL

    def subscribe(self) -> "Queue[dict[str, Any]]":
        q: Queue[dict[str, Any]] = Queue()
        with self._lock:
            q.put({"type": "status", "state": self.state.value})
            if self.is_terminal():
                q.put({"type": "__end__"})
            else:
                self._subscribers.append(q)
        return q


class JobStore:
    def __init__(self, max_workers: int = 4) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ds-job"
        )

    def create(self, run_id: str, *, kind: str) -> Job:
        with self._lock:
            if run_id in self._jobs:
                raise JobExistsError(run_id)
            job = Job(run_id=run_id, kind=kind)
            self._jobs[run_id] = job
            return job

    def submit(self, job: Job, fn: Callable[[Job], None]) -> None:
        self._executor.submit(self._wrap, job, fn)

    @staticmethod
    def _wrap(job: Job, fn: Callable[[Job], None]) -> None:
        try:
            fn(job)
        except Exception as exc:  # noqa: BLE001 - background boundary must fail closed
            job.fail(f"{type(exc).__name__}: {exc}")

    def get(self, run_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(run_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_jobs.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add app/api/jobs.py tests/test_api_jobs.py
git commit -m "feat: in-process JobStore with subscribe bus

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Request/response schemas

**Files:**
- Create: `app/api/schemas.py`
- Test: `tests/test_api_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_schemas.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import ChainRequest, RunRequest


def test_run_request_defaults_mode_auto() -> None:
    req = RunRequest(prompt="hi")
    assert req.mode == "auto"
    assert req.run_id is None
    assert req.planner is None
    assert req.model is None


def test_run_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        RunRequest(prompt="")


def test_run_request_rejects_bad_mode() -> None:
    with pytest.raises(ValidationError):
        RunRequest(prompt="hi", mode="turbo")


def test_chain_request_defaults() -> None:
    req = ChainRequest(prompt="hi")
    assert req.max_iterations == 3
    assert req.mode == "auto"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.schemas'`.

- [ ] **Step 3: Implement `app/api/schemas.py`**

```python
# app/api/schemas.py
"""HTTP request/response contract models. Distinct from internal app/models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ExecutionMode = Literal["sync", "async", "auto"]
PlannerChoice = Literal["mock", "openai"]


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1)
    run_id: str | None = None
    mode: ExecutionMode = "auto"
    planner: PlannerChoice | None = None
    model: str | None = None


class ChainRequest(BaseModel):
    prompt: str = Field(min_length=1)
    run_id: str | None = None
    mode: ExecutionMode = "auto"
    planner: PlannerChoice | None = None
    model: str | None = None
    max_iterations: int = Field(default=3, ge=1, le=10)


class ResumeRequest(BaseModel):
    event: Any


class ReplayRequest(BaseModel):
    new_run_id: str | None = None


class RunAck(BaseModel):
    run_id: str
    status: str
    state: str
    links: dict[str, str]


class RunResult(BaseModel):
    run_id: str
    status: str
    response: str
    spec: dict[str, Any] | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    record_dir: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    links: dict[str, str] = Field(default_factory=dict)


class RunStatus(BaseModel):
    run_id: str
    state: str
    status: str | None = None
    response: str | None = None
    budget_wall_seconds: int | None = None
    on_disk: bool = False
    links: dict[str, str] = Field(default_factory=dict)


class RunListItem(BaseModel):
    run_id: str
    status: str
    nodes: int


class RunList(BaseModel):
    runs: list[RunListItem]
    count: int


class ErrorEnvelope(BaseModel):
    error: dict[str, Any]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_schemas.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/api/schemas.py tests/test_api_schemas.py
git commit -m "feat: API request/response schemas

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Errors + serialization helpers

**Files:**
- Create: `app/api/errors.py`
- Create: `app/api/serialize.py`
- Test: `tests/test_api_serialize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_serialize.py
from __future__ import annotations

from pathlib import Path

from app.api.serialize import run_result_payload, run_status_payload
from app.assembly import RunConfig, build_supervisor
from app.recording import FileRecorder


def _run(tmp_path: Path, run_id: str):
    recorder = FileRecorder(root_dir=tmp_path, overwrite=True)
    supervisor = build_supervisor(
        RunConfig(planner="mock", model="gpt-5.4-nano", strict_runners=False),
        recorder=recorder,
    )
    return supervisor.run("compare", run_id=run_id)


def test_run_result_payload_shape(tmp_path: Path) -> None:
    result = _run(tmp_path, "ser-001")
    payload = run_result_payload(result)
    assert payload["run_id"] == "ser-001"
    assert payload["status"] == "ok"
    assert "values" in payload
    assert payload["spec"] is not None
    assert payload["spec"]["graph_id"]


def test_run_status_payload_shape(tmp_path: Path) -> None:
    result = _run(tmp_path, "ser-002")
    payload = run_status_payload(result)
    assert payload["run_id"] == "ser-002"
    assert payload["state"] in {"ok", "failed", "paused"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_serialize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.serialize'`.

- [ ] **Step 3: Implement `app/api/errors.py`**

```python
# app/api/errors.py
"""API exceptions and JSON error-envelope handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    status_code = 500
    error_type = "ApiError"

    def __init__(self, message: str, *, detail: object = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class NotFound(ApiError):
    status_code = 404
    error_type = "NotFound"


class Conflict(ApiError):
    status_code = 409
    error_type = "Conflict"


class Unauthorized(ApiError):
    status_code = 401
    error_type = "Unauthorized"


class BadRequest(ApiError):
    status_code = 400
    error_type = "BadRequest"


class ServiceUnavailable(ApiError):
    status_code = 503
    error_type = "ServiceUnavailable"


def _envelope(error_type: str, message: str, detail: object = None) -> dict:
    return {"error": {"type": error_type, "message": message, "detail": detail}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.error_type, exc.message, exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_envelope("ValidationError", "Request validation failed", exc.errors()),
        )
```

- [ ] **Step 4: Implement `app/api/serialize.py`**

```python
# app/api/serialize.py
"""Turn internal result objects into JSON-ready response dicts."""

from __future__ import annotations

from typing import Any

from app.supervisor import SupervisorResult


def _spec_dict(result: SupervisorResult) -> dict[str, Any] | None:
    if result.validated_spec is None:
        return None
    return result.validated_spec.model_dump(mode="json", by_alias=True)


def _budget_wall_seconds(result: SupervisorResult) -> int | None:
    spec = result.validated_spec
    if spec is None or spec.budget is None:
        return None
    return getattr(spec.budget, "max_wall_seconds", None)


def run_links(run_id: str) -> dict[str, str]:
    base = f"/runs/{run_id}"
    return {
        "self": base,
        "spec": f"{base}/spec",
        "trace": f"{base}/trace",
        "stream": f"{base}/trace/stream",
        "output": f"{base}/output",
        "graph": f"{base}/graph",
        "summary": f"{base}/summary",
        "artifacts": f"{base}/artifacts",
    }


def run_result_payload(result: SupervisorResult) -> dict[str, Any]:
    values: dict[str, Any] = {}
    artifacts: list[str] = []
    if result.result is not None:
        state = result.result.state
        values = dict(state.get("values", {}))
        artifacts = sorted(state.get("artifacts", {}).keys())
    record_dir = str(result.record.directory) if result.record is not None else None
    return {
        "run_id": result.run_id,
        "status": result.status,
        "response": result.response,
        "spec": _spec_dict(result),
        "values": values,
        "errors": list(result.errors),
        "record_dir": record_dir,
        "artifacts": artifacts,
        "links": run_links(result.run_id),
    }


def _state_from_status(status: str) -> str:
    if status == "ok":
        return "ok"
    if status == "paused":
        return "paused"
    return "failed"


def run_status_payload(result: SupervisorResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "state": _state_from_status(result.status),
        "status": result.status,
        "response": result.response,
        "budget_wall_seconds": _budget_wall_seconds(result),
        "on_disk": result.record is not None,
        "links": run_links(result.run_id),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_serialize.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add app/api/errors.py app/api/serialize.py tests/test_api_serialize.py
git commit -m "feat: API error envelope + result serializers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Dependencies wiring (`deps.py`)

Holds app-wide singletons (settings, recorder, job store, shared checkpointer) and the per-request config resolver + auth guard. Stored on `app.state` so tests can override.

**Files:**
- Create: `app/api/deps.py`
- Test: `tests/test_api_deps.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_deps.py
from __future__ import annotations

import pytest

from app.api.deps import AppContext, resolve_run_config
from app.api.errors import BadRequest, ServiceUnavailable
from app.api.settings import ApiSettings


def _ctx(**env) -> AppContext:
    return AppContext.build(ApiSettings.from_env(env))


def test_resolve_defaults_to_settings() -> None:
    ctx = _ctx()
    config = resolve_run_config(ctx, planner=None, model=None)
    assert config.planner == "mock"
    assert config.model == "gpt-5.4-nano"
    assert config.strict_runners is False


def test_resolve_rejects_model_outside_allowlist() -> None:
    ctx = _ctx(DS_MODEL_ALLOWLIST="gpt-5.4-nano")
    with pytest.raises(BadRequest):
        resolve_run_config(ctx, planner=None, model="evil-model")


def test_resolve_openai_without_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ctx = _ctx()
    with pytest.raises(ServiceUnavailable):
        resolve_run_config(ctx, planner="openai", model=None)


def test_resolve_openai_with_key_ok(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    ctx = _ctx()
    config = resolve_run_config(ctx, planner="openai", model=None)
    assert config.planner == "openai"
    assert config.strict_runners is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_deps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.deps'`.

- [ ] **Step 3: Implement `app/api/deps.py`**

```python
# app/api/deps.py
"""App-wide context + per-request config resolution + auth."""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import Request

from app.api.errors import BadRequest, ServiceUnavailable, Unauthorized
from app.api.jobs import JobStore
from app.api.settings import ApiSettings
from app.assembly import RunConfig, build_supervisor
from app.recording import FileRecorder
from app.supervisor import Supervisor


@dataclass
class AppContext:
    settings: ApiSettings
    recorder: FileRecorder
    jobs: JobStore
    checkpointer: object

    @classmethod
    def build(cls, settings: ApiSettings) -> "AppContext":
        from langgraph.checkpoint.memory import MemorySaver

        return cls(
            settings=settings,
            recorder=FileRecorder(root_dir=settings.runs_dir, overwrite=True),
            jobs=JobStore(max_workers=4),
            checkpointer=MemorySaver(),
        )

    def supervisor_for(self, config: RunConfig) -> Supervisor:
        return build_supervisor(
            config, recorder=self.recorder, checkpointer=self.checkpointer
        )


def get_context(request: Request) -> AppContext:
    return request.app.state.context


def resolve_run_config(
    ctx: AppContext,
    *,
    planner: str | None,
    model: str | None,
) -> RunConfig:
    chosen_planner = planner or ctx.settings.planner
    chosen_model = model or ctx.settings.model

    if not ctx.settings.is_model_allowed(chosen_model):
        raise BadRequest(
            f"Model {chosen_model!r} is not in the allowlist "
            f"{list(ctx.settings.model_allowlist)}"
        )
    if chosen_planner == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise ServiceUnavailable(
            "planner=openai requested but OPENAI_API_KEY is not set"
        )
    return RunConfig(
        planner=chosen_planner,  # type: ignore[arg-type]
        model=chosen_model,
        strict_runners=chosen_planner == "openai",
    )


def require_auth(request: Request) -> None:
    """Guard for POST endpoints. No-op when DS_API_KEY is unset."""
    ctx: AppContext = request.app.state.context
    expected = ctx.settings.api_key
    if not expected:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {expected}":
        raise Unauthorized("Missing or invalid bearer token")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_deps.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/api/deps.py tests/test_api_deps.py
git commit -m "feat: API app context, config resolver, auth guard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Health + registry routers, and `create_app`

This task makes a runnable app so subsequent routers can be tested with `TestClient`.

**Files:**
- Create: `app/api/routers/health.py`
- Create: `app/api/routers/registry.py`
- Create: `app/api/app.py`
- Modify: `app/api/__init__.py`
- Create: `app/api/__main__.py`
- Test: `tests/test_api_health_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_health_registry.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings


def _client(tmp_path) -> TestClient:
    settings = ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)})
    return TestClient(create_app(settings))


def test_healthz_ok(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_registry_lists_kinds_and_allowlists(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.get("/registry")
    assert resp.status_code == 200
    body = resp.json()
    kinds = {k["kind"] for k in body["node_kinds"]}
    assert "llm_call" in kinds and "wait_for_event" in kinds
    assert "web_search" in body["tools"]
    assert "critic" in body["subagents"]
    assert "python_eval" in body["forbidden_kinds"]
    sample = next(k for k in body["node_kinds"] if k["kind"] == "llm_call")
    assert "param_schema" in sample
    assert sample["param_schema"]["type"] == "object"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_health_registry.py -v`
Expected: FAIL with `ImportError: cannot import name 'create_app'`.

- [ ] **Step 3: Implement `app/api/routers/health.py`**

```python
# app/api/routers/health.py
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["meta"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 4: Implement `app/api/routers/registry.py`**

```python
# app/api/routers/registry.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.registry import (
    DEFAULT_SUBAGENTS,
    DEFAULT_TOOLS,
    FORBIDDEN_KINDS,
    default_kind_definitions,
)

router = APIRouter(tags=["registry"])


@router.get("/registry")
def get_registry() -> dict[str, Any]:
    kinds: list[dict[str, Any]] = []
    for kind, definition in default_kind_definitions().items():
        kinds.append(
            {
                "kind": kind.value,
                "description": definition.description,
                "counts_as_llm_call": definition.counts_as_llm_call,
                "has_side_effects": definition.has_side_effects,
                "requires_tool_allowlist": definition.requires_tool_allowlist,
                "requires_subagent_allowlist": definition.requires_subagent_allowlist,
                "param_schema": definition.param_model.model_json_schema(),
            }
        )
    return {
        "node_kinds": kinds,
        "tools": sorted(DEFAULT_TOOLS),
        "subagents": sorted(DEFAULT_SUBAGENTS),
        "forbidden_kinds": sorted(FORBIDDEN_KINDS),
    }
```

- [ ] **Step 5: Implement `app/api/app.py`**

```python
# app/api/app.py
"""create_app() — wire settings, context, routers, error handlers."""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.deps import AppContext
from app.api.errors import install_error_handlers
from app.api.routers import chains, health, registry, runs
from app.api.settings import ApiSettings


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    load_dotenv()
    settings = settings or ApiSettings.from_env()

    app = FastAPI(title="Dynamic Subgraphs API", version="1.0.0")
    app.state.context = AppContext.build(settings)

    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(registry.router)
    app.include_router(runs.router)
    app.include_router(chains.router)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        app.state.context.jobs.shutdown()

    return app
```

Note: `runs` and `chains` routers are created in Tasks 12–14. To keep this task runnable before they exist, create **stub** router modules now and flesh them out later:

```python
# app/api/routers/runs.py   (stub — replaced in Tasks 12-13)
from fastapi import APIRouter

router = APIRouter(tags=["runs"])
```

```python
# app/api/routers/chains.py   (stub — replaced in Task 14)
from fastapi import APIRouter

router = APIRouter(tags=["chains"])
```

- [ ] **Step 6: Implement `app/api/__init__.py`**

```python
# app/api/__init__.py
"""HTTP API (FastAPI) — thin layer over the supervisor."""

from app.api.app import create_app

__all__ = ["create_app"]
```

- [ ] **Step 7: Implement `app/api/__main__.py`**

```python
# app/api/__main__.py
"""`python -m app.api` -> run uvicorn."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("DS_HOST", "127.0.0.1")
    port = int(os.environ.get("DS_PORT", "8000"))
    uvicorn.run("app.api:create_app", host=host, port=port, factory=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_health_registry.py -v`
Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
git add app/api/routers/health.py app/api/routers/registry.py app/api/routers/runs.py app/api/routers/chains.py app/api/app.py app/api/__init__.py app/api/__main__.py tests/test_api_health_registry.py
git commit -m "feat: create_app + health and registry routers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Runs router — create, list, status, file sub-resources

**Files:**
- Modify (replace stub): `app/api/routers/runs.py`
- Test: `tests/test_api_runs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_runs.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings


def _client(tmp_path, **env) -> TestClient:
    env = {"DS_RUNS_DIR": str(tmp_path), **env}
    return TestClient(create_app(ApiSettings.from_env(env)))


def test_create_run_sync_returns_full_result(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post("/runs", json={"prompt": "compare A and B", "mode": "sync"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["spec"]["graph_id"]
    assert body["values"]
    assert body["links"]["self"].startswith("/runs/")


def test_create_run_async_returns_202(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/runs", json={"prompt": "x", "mode": "async", "run_id": "async-1"}
    )
    assert resp.status_code == 202
    assert resp.json()["run_id"] == "async-1"


def test_async_run_becomes_visible(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "mode": "async", "run_id": "async-2"})
    # mock runs finish near-instantly; poll once.
    status = client.get("/runs/async-2")
    assert status.status_code == 200
    assert status.json()["run_id"] == "async-2"


def test_auto_run_fast_returns_200(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post("/runs", json={"prompt": "x", "run_id": "auto-1"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_duplicate_run_id_conflicts(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "run_id": "dup", "mode": "sync"})
    resp = client.post("/runs", json={"prompt": "y", "run_id": "dup", "mode": "sync"})
    assert resp.status_code == 409


def test_get_run_404(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.get("/runs/ghost").status_code == 404


def test_list_runs(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "run_id": "list-1", "mode": "sync"})
    resp = client.get("/runs")
    assert resp.status_code == 200
    ids = {r["run_id"] for r in resp.json()["runs"]}
    assert "list-1" in ids


def test_file_subresources(tmp_path) -> None:
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "x", "run_id": "files-1", "mode": "sync"})
    assert client.get("/runs/files-1/spec").status_code == 200
    assert client.get("/runs/files-1/output").status_code == 200
    assert client.get("/runs/files-1/trace").status_code == 200
    graph = client.get("/runs/files-1/graph")
    assert graph.status_code == 200
    assert "graph" in graph.text.lower() or "-->" in graph.text
    assert client.get("/runs/files-1/summary").status_code == 200


def test_model_outside_allowlist_rejected(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post("/runs", json={"prompt": "x", "model": "evil", "mode": "sync"})
    assert resp.status_code == 400


def test_auth_required_when_key_set(tmp_path) -> None:
    client = _client(tmp_path, DS_API_KEY="secret")
    resp = client.post("/runs", json={"prompt": "x", "mode": "sync"})
    assert resp.status_code == 401
    ok = client.post(
        "/runs",
        json={"prompt": "x", "mode": "sync", "run_id": "authed"},
        headers={"Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_runs.py -v`
Expected: FAIL (404s / missing routes — the stub router has no endpoints).

- [ ] **Step 3: Implement `app/api/routers/runs.py`**

```python
# app/api/routers/runs.py
from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from app.api.deps import (
    AppContext,
    get_context,
    require_auth,
    resolve_run_config,
)
from app.api.errors import Conflict, NotFound
from app.api.jobs import Job, JobState
from app.api.schemas import RunRequest
from app.api.serialize import run_links, run_result_payload, run_status_payload
from app.recording.recorder import _validate_run_id

router = APIRouter(tags=["runs"])


def _new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def _ensure_unique(ctx: AppContext, run_id: str) -> None:
    if ctx.jobs.get(run_id) is not None or ctx.recorder.exists(run_id):
        raise Conflict(f"run_id already exists: {run_id!r}")


def _make_run_worker(ctx: AppContext, config, prompt: str, run_id: str):
    supervisor = ctx.supervisor_for(config)

    def work(job: Job) -> None:
        job.set_state(JobState.RUNNING)
        result = supervisor.run(prompt, run_id=run_id)
        spec = result.validated_spec
        if spec is not None and spec.budget is not None:
            job.budget_wall_seconds = getattr(spec.budget, "max_wall_seconds", None)
        if result.status == "ok":
            job.complete(result=result, state=JobState.OK)
        elif result.status == "paused":
            job.complete(result=result, state=JobState.PAUSED)
        else:
            job.complete(result=result, state=JobState.FAILED)

    return work


@router.post("/runs")
def create_run(
    request: Request,
    body: RunRequest,
    _: None = Depends(require_auth),
) -> Response:
    ctx = get_context(request)
    config = resolve_run_config(ctx, planner=body.planner, model=body.model)

    run_id = body.run_id or _new_run_id()
    _validate_run_id(run_id)
    _ensure_unique(ctx, run_id)

    job = ctx.jobs.create(run_id, kind="run")
    ctx.jobs.submit(job, _make_run_worker(ctx, config, body.prompt, run_id))

    if body.mode == "async":
        return JSONResponse(status_code=202, content=_ack(job))

    timeout = (
        ctx.settings.max_sync_seconds
        if body.mode == "sync"
        else ctx.settings.auto_sync_seconds
    )
    finished = job.wait(timeout=timeout)
    if finished and job.result is not None:
        return JSONResponse(status_code=200, content=run_result_payload(job.result))
    return JSONResponse(status_code=202, content=_ack(job))


def _ack(job: Job) -> dict[str, Any]:
    return {
        "run_id": job.run_id,
        "status": job.state.value,
        "state": job.state.value,
        "links": run_links(job.run_id),
    }


@router.get("/runs")
def list_runs(request: Request) -> dict[str, Any]:
    ctx = get_context(request)
    runs = ctx.recorder.list_runs()
    return {"runs": runs, "count": len(runs)}


@router.get("/runs/{run_id}")
def get_run(request: Request, run_id: str) -> dict[str, Any]:
    ctx = get_context(request)
    _validate_run_id(run_id)
    job = ctx.jobs.get(run_id)
    if job is not None and job.result is not None:
        return run_status_payload(job.result)
    if job is not None and not job.is_terminal():
        return {
            "run_id": run_id,
            "state": job.state.value,
            "status": None,
            "response": None,
            "budget_wall_seconds": job.budget_wall_seconds,
            "on_disk": ctx.recorder.exists(run_id),
            "links": run_links(run_id),
        }
    if ctx.recorder.exists(run_id):
        output = ctx.recorder.load_output(run_id)
        state = "ok" if output.get("ok") else "failed"
        return {
            "run_id": run_id,
            "state": state,
            "status": state,
            "response": None,
            "budget_wall_seconds": None,
            "on_disk": True,
            "links": run_links(run_id),
        }
    raise NotFound(f"No run {run_id!r}")


def _run_file(ctx: AppContext, run_id: str, name: str):
    _validate_run_id(run_id)
    if not ctx.recorder.exists(run_id):
        raise NotFound(f"No run {run_id!r}")
    path = ctx.recorder.run_dir(run_id) / name
    if not path.exists():
        raise NotFound(f"No {name} for run {run_id!r}")
    return path


@router.get("/runs/{run_id}/spec")
def get_spec(request: Request, run_id: str) -> Response:
    path = _run_file(get_context(request), run_id, "spec.json")
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


@router.get("/runs/{run_id}/output")
def get_output(request: Request, run_id: str) -> Response:
    path = _run_file(get_context(request), run_id, "output.json")
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))


@router.get("/runs/{run_id}/trace")
def get_trace(request: Request, run_id: str) -> Response:
    path = _run_file(get_context(request), run_id, "trace.jsonl")
    return PlainTextResponse(
        content=path.read_text(encoding="utf-8"),
        media_type="application/x-ndjson",
    )


@router.get("/runs/{run_id}/graph")
def get_graph(request: Request, run_id: str) -> Response:
    path = _run_file(get_context(request), run_id, "graph.mmd")
    return PlainTextResponse(content=path.read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/summary")
def get_summary(request: Request, run_id: str) -> Response:
    path = _run_file(get_context(request), run_id, "summary.md")
    return PlainTextResponse(content=path.read_text(encoding="utf-8"), media_type="text/markdown")


@router.get("/runs/{run_id}/artifacts")
def list_artifacts(request: Request, run_id: str) -> dict[str, Any]:
    ctx = get_context(request)
    _validate_run_id(run_id)
    if not ctx.recorder.exists(run_id):
        raise NotFound(f"No run {run_id!r}")
    art_dir = ctx.recorder.run_dir(run_id) / "artifacts"
    names = sorted(p.name for p in art_dir.iterdir()) if art_dir.is_dir() else []
    return {"run_id": run_id, "artifacts": names}


@router.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(request: Request, run_id: str, name: str) -> Response:
    ctx = get_context(request)
    if not ctx.recorder.exists(run_id):
        raise NotFound(f"No run {run_id!r}")
    path = ctx.recorder.artifact_path(run_id, name)  # validates name
    if not path.exists():
        raise NotFound(f"No artifact {name!r} for run {run_id!r}")
    return FileResponse(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_runs.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add app/api/routers/runs.py tests/test_api_runs.py
git commit -m "feat: runs router (create/list/status/files)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Runs router — resume, replay, SSE stream

**Files:**
- Modify: `app/api/routers/runs.py`
- Test: `tests/test_api_resume_replay.py`, `tests/test_api_stream.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_resume_replay.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings
from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec, GraphBudget
from app.supervisor import StaticPlanner


def _wait_spec() -> GraphSpec:
    return GraphSpec(
        graph_id="wait-graph",
        goal="pause then finish",
        budget=GraphBudget(max_nodes=8),
        nodes=[
            NodeSpec(
                id="hold",
                kind=NodeKind.WAIT_FOR_EVENT,
                outputs=["signal"],
                params={"event_type": "human_input", "output_key": "signal"},
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "hold"}),
            EdgeSpec.model_validate({"from": "hold", "to": "END"}),
        ],
    )


def _client(tmp_path) -> TestClient:
    app = create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)}))
    # Force the wait-graph planner so a run pauses.
    app.state.context.recorder  # keep recorder
    from app.assembly import build_supervisor

    original = app.state.context.supervisor_for

    def patched(config):
        sup = original(config)
        sup._planner = StaticPlanner(_wait_spec())
        return sup

    app.state.context.supervisor_for = patched  # type: ignore[assignment]
    return TestClient(app)


def test_resume_completes_paused_run(tmp_path) -> None:
    client = _client(tmp_path)
    created = client.post(
        "/runs", json={"prompt": "go", "run_id": "wait-1", "mode": "sync"}
    )
    assert created.status_code == 200
    assert created.json()["status"] == "paused"

    resumed = client.post(
        "/runs/wait-1/resume", json={"event": {"event_type": "human_input", "value": "hello"}}
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "ok"


def test_replay_creates_new_run(tmp_path) -> None:
    # Replay needs a completed run; use the default mock planner instead.
    app = create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)}))
    client = TestClient(app)
    client.post("/runs", json={"prompt": "x", "run_id": "rp-1", "mode": "sync"})
    resp = client.post("/runs/rp-1/replay", json={"new_run_id": "rp-1-replay"})
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "rp-1-replay"
```

```python
# tests/test_api_stream.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings


def test_stream_emits_status_and_done(tmp_path) -> None:
    app = create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)}))
    client = TestClient(app)
    client.post("/runs", json={"prompt": "x", "run_id": "stream-1", "mode": "async"})

    with client.stream("GET", "/runs/stream-1/trace/stream") as resp:
        assert resp.status_code == 200
        text = "".join(chunk for chunk in resp.iter_text())

    assert "event: status" in text
    assert "event: done" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_resume_replay.py tests/test_api_stream.py -v`
Expected: FAIL (routes 404 / not implemented).

- [ ] **Step 3: Add resume, replay, and SSE endpoints to `app/api/routers/runs.py`**

Add these imports at the top of `runs.py`:

```python
import time
from collections.abc import Iterator

from fastapi.responses import StreamingResponse

from app.api.schemas import ResumeRequest, ReplayRequest
```

Append these endpoints to `runs.py`:

```python
@router.post("/runs/{run_id}/resume")
def resume_run(
    request: Request,
    run_id: str,
    body: ResumeRequest,
    _: None = Depends(require_auth),
) -> Response:
    ctx = get_context(request)
    _validate_run_id(run_id)
    if not ctx.recorder.exists(run_id):
        raise NotFound(f"No run {run_id!r} to resume")
    config = resolve_run_config(ctx, planner=None, model=None)
    supervisor = ctx.supervisor_for(config)
    result = supervisor.resume(run_id, event=body.event)
    status_code = 200 if result.status in {"ok", "paused"} else 409
    return JSONResponse(status_code=status_code, content=run_result_payload(result))


@router.post("/runs/{run_id}/replay")
def replay_run(
    request: Request,
    run_id: str,
    body: ReplayRequest,
    _: None = Depends(require_auth),
) -> Response:
    ctx = get_context(request)
    _validate_run_id(run_id)
    if not ctx.recorder.exists(run_id):
        raise NotFound(f"No run {run_id!r} to replay")
    config = resolve_run_config(ctx, planner=None, model=None)
    supervisor = ctx.supervisor_for(config)
    result = supervisor.replay(run_id, new_run_id=body.new_run_id)
    status_code = 200 if result.status in {"ok", "paused"} else 409
    return JSONResponse(status_code=status_code, content=run_result_payload(result))


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/runs/{run_id}/trace/stream")
def stream_run(request: Request, run_id: str) -> StreamingResponse:
    ctx = get_context(request)
    _validate_run_id(run_id)
    job = ctx.jobs.get(run_id)
    if job is None and not ctx.recorder.exists(run_id):
        raise NotFound(f"No run {run_id!r}")

    def gen() -> Iterator[str]:
        if job is not None:
            queue = job.subscribe()
            while True:
                msg = queue.get(timeout=ctx.settings.max_sync_seconds)
                if msg["type"] == "__end__":
                    break
                yield _sse("status", {"state": msg["state"]})
        # final recorded trace, if present
        trace_path = ctx.recorder.run_dir(run_id) / "trace.jsonl"
        if trace_path.exists():
            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            yield _sse("trace", {"events": events})
        final_state = job.state.value if job is not None else "ok"
        yield _sse("done", {"status": final_state, "run_id": run_id})

    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_resume_replay.py tests/test_api_stream.py -v`
Expected: all passed.

If `test_resume_completes_paused_run` fails because the run did not pause, confirm the executor received the shared checkpointer (it does via `AppContext.build` → `supervisor_for` → `build_supervisor(checkpointer=...)`). The paused run's checkpoint lives in the shared `MemorySaver`, so resume in a later request finds it.

- [ ] **Step 5: Commit**

```bash
git add app/api/routers/runs.py tests/test_api_resume_replay.py tests/test_api_stream.py
git commit -m "feat: resume, replay, and SSE stream endpoints

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Chains router

**Files:**
- Modify (replace stub): `app/api/routers/chains.py`
- Test: `tests/test_api_chains.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_chains.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)})))


def test_create_chain_sync(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/chains",
        json={"prompt": "investigate", "run_id": "chain-1", "mode": "sync", "max_iterations": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chain_id"] == "chain-1"
    assert body["status"] in {"ok", "stopped", "max_iterations"}
    assert isinstance(body["steps"], list)


def test_get_chain(tmp_path) -> None:
    client = _client(tmp_path)
    client.post(
        "/chains",
        json={"prompt": "investigate", "run_id": "chain-2", "mode": "sync", "max_iterations": 1},
    )
    resp = client.get("/chains/chain-2")
    assert resp.status_code == 200
    assert resp.json()["chain_id"] == "chain-2"


def test_get_chain_404(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.get("/chains/ghost").status_code == 404


def test_create_chain_async_202(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/chains",
        json={"prompt": "x", "run_id": "chain-3", "mode": "async", "max_iterations": 1},
    )
    assert resp.status_code == 202
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_chains.py -v`
Expected: FAIL (routes 404 — stub).

- [ ] **Step 3: Implement `app/api/routers/chains.py`**

```python
# app/api/routers/chains.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from app.api.deps import (
    AppContext,
    get_context,
    require_auth,
    resolve_run_config,
)
from app.api.errors import Conflict, NotFound
from app.api.jobs import Job, JobState
from app.api.schemas import ChainRequest
from app.recording.recorder import _validate_run_id

router = APIRouter(tags=["chains"])


def _chain_payload(result: Any) -> dict[str, Any]:
    return {
        "chain_id": result.chain_id,
        "status": result.status,
        "response": result.response,
        "steps": [
            {
                "iteration": s.iteration,
                "run_id": s.run_id,
                "status": s.result.status,
                "decision": s.decision.action,
                "reason": s.decision.reason,
            }
            for s in result.steps
        ],
    }


def _make_chain_worker(ctx: AppContext, config, prompt: str, run_id: str, max_iter: int):
    supervisor = ctx.supervisor_for(config)

    def work(job: Job) -> None:
        job.set_state(JobState.RUNNING)
        result = supervisor.run_iteratively(
            prompt, run_id=run_id, max_iterations=max_iter
        )
        state = JobState.OK if result.status in {"ok", "stopped", "max_iterations"} else JobState.FAILED
        job.complete(result=result, state=state)

    return work


@router.post("/chains")
def create_chain(
    request: Request,
    body: ChainRequest,
    _: None = Depends(require_auth),
) -> Response:
    ctx = get_context(request)
    config = resolve_run_config(ctx, planner=body.planner, model=body.model)
    run_id = body.run_id or f"chain-{__import__('uuid').uuid4().hex[:12]}"
    _validate_run_id(run_id)
    if ctx.jobs.get(run_id) is not None or ctx.recorder.exists(run_id):
        raise Conflict(f"chain id already exists: {run_id!r}")

    job = ctx.jobs.create(run_id, kind="chain")
    ctx.jobs.submit(
        job, _make_chain_worker(ctx, config, body.prompt, run_id, body.max_iterations)
    )

    if body.mode == "async":
        return JSONResponse(
            status_code=202,
            content={"chain_id": run_id, "status": job.state.value, "links": {"self": f"/chains/{run_id}"}},
        )

    timeout = (
        ctx.settings.max_sync_seconds
        if body.mode == "sync"
        else ctx.settings.auto_sync_seconds
    )
    finished = job.wait(timeout=timeout)
    if finished and job.result is not None:
        return JSONResponse(status_code=200, content=_chain_payload(job.result))
    return JSONResponse(
        status_code=202,
        content={"chain_id": run_id, "status": job.state.value, "links": {"self": f"/chains/{run_id}"}},
    )


@router.get("/chains/{chain_id}")
def get_chain(request: Request, chain_id: str) -> dict[str, Any]:
    ctx = get_context(request)
    _validate_run_id(chain_id)
    job = ctx.jobs.get(chain_id)
    if job is not None and job.result is not None:
        return _chain_payload(job.result)
    try:
        return ctx.recorder.load_chain(chain_id)
    except FileNotFoundError as exc:
        raise NotFound(f"No chain {chain_id!r}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_chains.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/api/routers/chains.py tests/test_api_chains.py
git commit -m "feat: chains router (run_iteratively over HTTP)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Full-suite verification + smoke run

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass (the original 222 plus the new API tests), zero failures.

- [ ] **Step 2: Boot the server and hit it live**

In one terminal:
```bash
.venv/Scripts/python.exe -m app.api
```
In another (or via the user's `!` prefix):
```bash
curl -s http://127.0.0.1:8000/healthz
curl -s -X POST http://127.0.0.1:8000/runs -H "content-type: application/json" -d "{\"prompt\":\"compare A and B\",\"mode\":\"sync\"}"
curl -s http://127.0.0.1:8000/registry
```
Expected: `{"status":"ok"}`; a full run result JSON with `status: "ok"` and a `spec`; the registry listing. Stop the server with Ctrl-C.

- [ ] **Step 3: Commit nothing (verification only). If you made fixes, commit them**

```bash
git add -A
git commit -m "test: verify full API suite + live smoke

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/api.md`
- Modify: `docs/index.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Write `docs/api.md`**

Document: how to run (`python -m app.api`, env vars from §3 of the spec), the endpoint table (from spec §4), the sync/async/auto modes, auth, and a `curl` example for each verb. Include the note that SSE is progress-level now with per-node planned.

- [ ] **Step 2: Expand `README.md`**

Replace the one-line README with: project one-paragraph summary, quickstart (`uv sync`, run the demo, run the API), and a pointer to `docs/api.md` and the canonical design.

- [ ] **Step 3: Add API row to `docs/index.md`**

Under "Implementation notes", add: `- [api.md](./api.md) — HTTP surface over the supervisor.`

- [ ] **Step 4: Update `AGENTS.md` MVP sequence**

Mark step 6 (`API POST /runs`) done; update the package map row for `app/api/` from "(future)" to a short description of the live surface.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/api.md docs/index.md AGENTS.md
git commit -m "docs: API guide, README quickstart, MVP step 6 done

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Finish the branch

- [ ] **Step 1: Final full-suite run**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all green.

- [ ] **Step 2: Use the finishing-a-development-branch skill**

Invoke `superpowers:finishing-a-development-branch` to choose how to integrate `feat/api-layer` (merge to `main`, open a PR, or keep the branch). Do not merge without the user's choice.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §2 architecture / dependency direction → Tasks 3, 11 (api → supervisor; LangGraph stays behind assembly/runtime). ✓
- §2 `app/assembly.py` refactor → Tasks 3, 4. ✓
- §3 ApiSettings + env + allowlists → Task 6; per-request override + allowlist enforcement → Task 10. ✓
- §4 every endpoint → health/registry (11), runs CRUD+files (12), resume/replay/SSE (13), chains (14). ✓
- §5 sync/async/auto via JobStore grace-window → Tasks 7, 12, 14. ✓
- §5 run-failure = 200 with status; 202 async → Task 12 tests. ✓
- §6 progress SSE + bus seam → Tasks 7 (`subscribe`), 13. ✓
- §7 shared MemorySaver checkpointer for resume → Task 10 (`AppContext.build`), 13 (resume test). ✓
- §8 error envelope, auth, schemas → Tasks 8, 9, 10. ✓
- §9 recorder read helpers → Task 5. ✓
- §10 deps + run command → Tasks 2, 11. ✓
- §11 offline token-free testing → all tests use `planner=mock`; openai path tested via 503 + allowlist, no network. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases" left; every code step contains complete code. The only deliberately-deferred description is Task 16 Step 1 (doc prose), which is documentation content, not code. ✓

**Type consistency:** `JobState` values, `RunConfig`/`build_supervisor` signature, `JobStore.create(run_id, *, kind=)`, `Job.complete(*, result=, state=)`, `run_result_payload`/`run_status_payload`, and `_validate_run_id` import are used identically across Tasks 3–14. SSE keys (`type`, `state`, `__end__`) match between `jobs.py` (Task 7) and the stream generator (Task 13). ✓

**Known deviation from spec (documented):** the spec's SSE example showed a `recording` state; the plan omits it because the supervisor records internally and we cannot observe that sub-step without instrumenting execution. The bus seam allows adding it later. Behavior is otherwise as specified.
