# FastAPI Layer — Design Spec

**Date:** 2026-05-30
**Status:** approved design, pre-implementation
**Scope:** MVP sequence step 6 (`API POST /runs`) expanded to the full supervisor surface.
**Supersedes:** the one-line docstring in `app/api/__init__.py`.

---

## 1. Goal

Expose the existing `Supervisor` over HTTP without leaking LangGraph types or
changing runtime semantics. The API is a thin translation layer: HTTP request →
`Supervisor` call → JSON. Everything the supervisor and recorder already do
(`run`, `run_iteratively`, `resume`, `replay`, recording, registry) becomes
reachable over HTTP, plus listing/inspection of recorded runs.

Non-goals for this slice:

- True per-node live streaming (executor instrumentation) — deferred, with a
  seam left in place (see §6).
- Cross-process / cross-restart resume durability (SqliteSaver) — deferred (§7).
- Per-graph custom state, auth beyond an optional bearer token, multi-tenant
  isolation, rate limiting.

---

## 2. Architecture & dependency direction

Honors the existing rule from `ARCHITECTURE.md`:

```text
api → supervisor → {compiler, runtime, recording}
                 → registry (read-only)
models ← all layers
```

LangGraph types never cross the API boundary. Every value returned to a client
is a Pydantic model or a recorded file's contents.

### Package layout

```text
app/api/
  __init__.py        # exports create_app
  __main__.py        # `python -m app.api` -> uvicorn
  settings.py        # ApiSettings (env-driven) + allowlists + optional bearer key
  schemas.py         # request/response Pydantic models — the API contract
  jobs.py            # in-process JobStore + ThreadPoolExecutor background execution
  errors.py          # exception handlers -> JSON error envelope
  deps.py            # DI: settings, recorder, job store, shared checkpointer, auth
  routers/
    __init__.py
    runs.py          # /runs ...
    chains.py        # /chains ...
    registry.py      # /registry
    health.py        # /healthz
app/assembly.py      # NEW shared builder: RunConfig -> Supervisor (used by api AND main.py)
```

### Targeted refactor: `app/assembly.py`

`main.py` currently holds private `_build_planner`, `_build_*_runner` helpers
that wire a `Supervisor` from a mock/llm flag. The API needs the same wiring.
Lift this into a single shared builder so CLI and API construct **identical**
supervisors:

```python
@dataclass(frozen=True)
class RunConfig:
    planner: Literal["mock", "openai"]
    model: str
    strict_runners: bool          # True for openai, False for mock

def build_supervisor(
    config: RunConfig,
    *,
    recorder: Recorder,
    checkpointer: Any | None = None,
) -> Supervisor: ...
```

`main.py` is rewritten to call `build_supervisor(...)`, shrinking it and
removing the duplicate wiring. This is the only change to existing runtime code.

---

## 3. Configuration (`ApiSettings`, env-driven)

Read from environment / `.env` at app construction:

| Env var | Default | Meaning |
|---|---|---|
| `DS_PLANNER` | `mock` | Default planner mode (`mock` boots free + offline). |
| `DS_MODEL` | `gpt-5.4-nano` | Default model when planner is `openai`. |
| `DS_MODEL_ALLOWLIST` | `gpt-5.4-nano` | Comma list; per-request `model` must be in here. |
| `DS_RUNS_DIR` | `runs` | Recorder root directory. |
| `DS_API_KEY` | *(unset)* | If set, `POST` endpoints require `Authorization: Bearer <key>`. |
| `DS_AUTO_SYNC_SECONDS` | `25` | `auto` mode grace window before falling back to 202. |
| `DS_MAX_SYNC_SECONDS` | `120` | Hard cap a `sync` request blocks before falling back to 202. |

`create_app(settings: ApiSettings | None = None) -> FastAPI` — settings injected
for tests, otherwise loaded from env. Per-request `planner`/`model` overrides are
validated against the allowlist; `planner=openai` without `OPENAI_API_KEY` → `503`.

---

## 4. Endpoints

| Method | Path | Action |
|---|---|---|
| `POST` | `/runs` | plan → validate → execute → record (mode: sync/async/auto) |
| `GET` | `/runs` | list recorded runs (paged) |
| `GET` | `/runs/{id}` | merged status: live job state (if any) + recorded artifacts |
| `GET` | `/runs/{id}/spec` | `spec.json` |
| `GET` | `/runs/{id}/trace` | recorded `trace.jsonl` (ndjson) |
| `GET` | `/runs/{id}/trace/stream` | **SSE** progress stream (§6) |
| `GET` | `/runs/{id}/output` | `output.json` |
| `GET` | `/runs/{id}/graph` | `graph.mmd` (mermaid) |
| `GET` | `/runs/{id}/summary` | `summary.md` |
| `GET` | `/runs/{id}/artifacts` | list emitted artifact names |
| `GET` | `/runs/{id}/artifacts/{name}` | one emitted file (content-type guessed) |
| `POST` | `/runs/{id}/resume` | feed `wait_for_event` payload |
| `POST` | `/runs/{id}/replay` | re-execute recorded spec under a new id |
| `POST` | `/chains` | `run_iteratively` (sync/async/auto) |
| `GET` | `/chains/{id}` | `chain.json` + steps |
| `GET` | `/registry` | node kinds, param JSON-schemas, tool/subagent allowlists, forbidden kinds |
| `GET` | `/healthz` | liveness |

SSE lives on its own path (`/trace/stream`) rather than content-negotiating
`/trace`, for cleaner testing and docs.

---

## 5. Execution model — sync / async / auto

**One execution path.** Every run and chain is submitted to an in-process
`JobStore` backed by a `ThreadPoolExecutor`. The supervisor call runs in a
worker thread; the request handler only decides how long to wait.

- `mode=async` → submit, return **202** `{run_id, status: "queued", links}`
  immediately. Client polls `GET /runs/{id}`.
- `mode=sync` → submit, block-wait up to `DS_MAX_SYNC_SECONDS`, return **200**
  full result. Exceeds cap → **202** + run_id (the run keeps going; only the
  wait stops).
- `mode=auto` (default) → submit, block-wait up to `DS_AUTO_SYNC_SECONDS`.
  Completes in window → **200** full result; else → **202** to poll.

### Why `auto` uses a grace window, not the planner's predicted budget

The planner's `spec.budget.max_wall_seconds` is a *prediction* and is unknown
until planning completes. Keying `auto` off **actual completion within a grace
window** is strictly more reliable than trusting an estimate, and avoids
splitting the supervisor's single plan→execute graph for this slice. The
planner's `budget` still governs the hard limits *inside* the run, and the job's
status surfaces `spec.budget.max_wall_seconds` once planning completes so a
client can still see the estimate.

### Run-level failure vs HTTP error

Run-level failures (invalid plan, compile error, execution error) are **still
recorded** and return **200** with the failed `status` in the body. HTTP error
codes are reserved for request-level problems (bad input, auth, not found). The
only non-200 success code is **202** (async accepted).

### JobStore

```text
Job:
  run_id
  kind: "run" | "chain"
  state: queued | running | recording | ok | failed | paused
  submitted_at / started_at / finished_at
  result: SupervisorResult | IterativeSupervisorResult | None
  error: str | None
  budget_wall_seconds: int | None   # filled once planning completes
  subscribe() -> event stream       # bus seam for SSE + future per-node events
```

Thread-safe (lock around state transitions). Holds both live and recently
finished jobs in memory; `GET /runs/{id}` falls back to disk (recorder) for jobs
not in memory (e.g., pre-existing runs, or after the in-memory job is evicted).

---

## 6. SSE streaming — progress now, per-node later

`GET /runs/{id}/trace/stream` emits **lifecycle** events by subscribing to the
JobStore bus:

```text
event: status  data: {"state": "running"}
event: status  data: {"state": "recording"}
event: trace   data: {"events": [ ...full recorded trace... ]}
event: done    data: {"status": "ok", "run_id": "..."}
```

No executor changes. Hand-rolled with `StreamingResponse` (no `sse-starlette`
dependency).

**Extension point for true per-node live (future slice):** the JobStore already
exposes a subscribe/event-bus seam. Deepening to per-node only requires adding a
callback hook in the runtime node wrappers that publishes node start/finish/status
onto that same bus. The HTTP/SSE layer does not change when that lands.

---

## 7. Resume durability

A single shared `MemorySaver` checkpointer lives in app state (`deps.py`) and is
injected into **every** supervisor the assembly builds for this app instance.
This makes pause-in-job-A / resume-in-request-B work **across requests within one
process**.

- Cross-process / cross-restart durability (SqliteSaver) is a noted future config
  flag, **not** in this slice (needs `langgraph-checkpoint-sqlite`).
- `resume` on a run whose spec has no `wait_for_event`, or after a restart wiped
  memory, returns a clean `409`/`404` with an explanatory message rather than a
  stack trace.

---

## 8. Errors, auth, schemas

### Error envelope

All errors serialize as:

```json
{ "error": { "type": "RunNotFound", "message": "...", "detail": null } }
```

via FastAPI exception handlers.

| Code | When |
|---|---|
| `400` / `422` | malformed request / Pydantic validation |
| `401` | missing/bad bearer when `DS_API_KEY` is set |
| `404` | run / chain / artifact not found |
| `409` | run_id already exists; run not resumable |
| `503` | `planner=openai` requested but no `OPENAI_API_KEY` |

### Auth

Optional bearer token on **POST** endpoints only (GETs open). Disabled entirely
when `DS_API_KEY` is unset (localhost dev default).

### Schemas (`schemas.py`)

Request: `RunRequest` (prompt, run_id?, mode?, planner?, model?),
`ChainRequest` (… + max_iterations), `ResumeRequest` (event), `ReplayRequest`
(new_run_id?).

Response: `RunAck` (202), `RunResult` (200 sync), `RunStatus` (GET /runs/{id}),
`RunList`, `ChainResult`, `RegistryInfo`, `ErrorEnvelope`.

Contract models live in `schemas.py`, kept distinct from internal `app/models`
types so the wire format can evolve independently.

---

## 9. Recorder read-helpers (targeted addition)

`GET /runs` and the file sub-resource endpoints need reads the recorder does not
expose yet. Added to `FileRecorder` (where directory-layout knowledge belongs),
all pure reads:

- `list_runs() -> list[RunSummary]` — scan `runs/`, derive id + status + nodes.
- `load_output(run_id) -> dict`
- `run_dir(run_id) -> Path`
- `artifact_path(run_id, name) -> Path` (validated, no path traversal)
- `exists(run_id) -> bool`

`run_id` validation reuses the existing `_validate_run_id` safe-char check;
artifact names are validated the same way to prevent traversal.

---

## 10. Dependencies & running

Add to `pyproject.toml`:

- runtime: `fastapi`, `uvicorn[standard]`
- dev: `httpx` (for `TestClient`)

SSE is hand-rolled — no `sse-starlette`. Run via:

```bash
python -m app.api                       # uvicorn, reads env/.env
uvicorn "app.api:create_app" --factory  # explicit
```

---

## 11. Testing (all offline, zero tokens)

`fastapi.testclient.TestClient` against `create_app(<mock settings>)`.

Coverage:

- Each endpoint happy path.
- `mode` matrix: `sync` returns full 200, `async` returns 202 + pollable,
  `auto` returns 200 when fast.
- A `wait_for_event` spec driven through `POST /runs/{id}/resume` (exercises the
  shared checkpointer).
- `replay` of a recorded run.
- A chain via `POST /chains` using a static `IterationDecider`.
- `/registry` shape; `/healthz`.
- Auth: 401 without bearer when key set; open when unset.
- Allowlist: per-request `model` outside allowlist → 422; `planner=openai`
  without key → 503.
- Error paths: 404 unknown run/chain/artifact, 409 duplicate run_id.
- JobStore unit tests: state transitions, thread-safety, grace-window timeout
  semantics.
- SSE: `/trace/stream` yields ordered `status` → `trace` → `done` events for an
  async run.

The `openai` path is tested by injecting a fake planner via FastAPI dependency
override (no real network calls), plus asserting `503` when the key is absent.

---

## 12. Open items / future slices (explicitly out of scope here)

1. True per-node live SSE (instrument runtime node wrappers onto the JobStore bus).
2. SqliteSaver checkpointer for cross-restart resume.
3. Eval / policy gates (MVP step 7) — separate spec.
4. Persistent (non-in-memory) JobStore if multi-process serving is ever needed.
