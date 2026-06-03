# HTTP API

A thin FastAPI layer over the existing `Supervisor`. Every HTTP request becomes a
`Supervisor` call and returns JSON or a recorded file's contents — LangGraph types
never cross the boundary.

Canonical design: [`superpowers/specs/2026-05-30-fastapi-layer-design.md`](./superpowers/specs/2026-05-30-fastapi-layer-design.md).

---

## Running

```bash
python -m app.api                        # uvicorn, reads env/.env
uvicorn "app.api:create_app" --factory   # explicit factory form
```

`create_app(settings: ApiSettings | None = None)` loads settings from the
environment (and `.env`) when none are injected. By default the API boots in
**mock** mode: free, offline, zero tokens.

---

## Configuration

Read from the environment / `.env` at app construction (env prefix `DS_`):

| Env var | Default | Meaning |
|---|---|---|
| `DS_PLANNER` | `mock` | Default planner mode (`mock` or `llm`; legacy `openai` maps to `llm` + `provider=openai`). |
| `DS_PROVIDER` | `openai` | Default model provider for `planner=llm` and `decider=llm`. |
| `DS_MODEL` | `gpt-5.4-nano` | Default model for the selected provider. |
| `DS_MODEL_ALLOWLIST` | `gpt-5.4-nano` | Comma list; accepts plain model ids or provider-qualified ids such as `openai:gpt-5.4-nano`. |
| `DS_RUNS_DIR` | `runs` | Recorder root directory. |
| `DS_API_KEY` | *(unset)* | If set, `POST` endpoints require `Authorization: Bearer <key>`. |
| `DS_AUTO_SYNC_SECONDS` | `25` | `auto` mode grace window before falling back to 202. |
| `DS_MAX_SYNC_SECONDS` | `120` | Hard cap a `sync` request blocks before falling back to 202. |

Per-request `planner`/`provider`/`model` overrides are validated against the
allowlist. Requesting `planner=llm` with a provider whose required credentials
are missing returns `503`.

---

## Endpoints

| Method | Path | Action |
|---|---|---|
| `POST` | `/runs` | plan → validate → execute → record (mode: sync/async/auto) |
| `GET` | `/runs` | list recorded runs |
| `GET` | `/runs/{id}` | merged status: live job state (if any) + recorded artifacts |
| `GET` | `/runs/{id}/spec` | `spec.json` |
| `GET` | `/runs/{id}/trace` | recorded `trace.jsonl` (ndjson) |
| `GET` | `/runs/{id}/trace/stream` | **SSE** progress stream (see below) |
| `GET` | `/runs/{id}/output` | `output.json` |
| `GET` | `/runs/{id}/graph` | `graph.mmd` (mermaid) |
| `GET` | `/runs/{id}/summary` | `summary.md` |
| `GET` | `/runs/{id}/artifacts` | list emitted artifact names |
| `GET` | `/runs/{id}/artifacts/{name}` | one emitted file (content-type guessed) |
| `POST` | `/runs/{id}/resume` | feed a `wait_for_event` payload |
| `POST` | `/runs/{id}/replay` | re-execute a recorded spec under a new id |
| `POST` | `/chains` | `run_iteratively` (sync/async/auto) |
| `GET` | `/chains/{id}` | `chain.json` + steps |
| `GET` | `/registry` | node kinds, param JSON-schemas, tool/subagent allowlists, forbidden kinds |
| `GET` | `/healthz` | liveness |

`POST /chains` defaults to `decider="status"`: token-free, stop on `ok`, ask
on `paused`, fail closed otherwise. Set `decider="llm"` to use the structured
iteration judge after each completed iteration; it can `stop`, `replan`,
`ask_user`, or `fail`, and accepts optional `success_criteria` plus
`judge_failed_runs`.

---

## Execution modes — sync / async / auto

Every run and chain is submitted to an in-process `JobStore` backed by a
`ThreadPoolExecutor`. The supervisor call runs in a worker thread; the request
handler only decides how long to wait.

- `mode=async` → submit, return **202** `{run_id, status: "queued", links}`
  immediately. Client polls `GET /runs/{id}`.
- `mode=sync` → submit, block-wait up to `DS_MAX_SYNC_SECONDS`, return **200**
  with the full result. Exceeds the cap → **202** + run_id (the run keeps going;
  only the wait stops).
- `mode=auto` (default) → submit, block-wait up to `DS_AUTO_SYNC_SECONDS`.
  Completes in window → **200** full result; else → **202** to poll.

**Run-level failure vs HTTP error.** Run-level failures (invalid plan, compile
error, execution error) are still recorded and return **200** with the failed
`status` in the body. HTTP error codes are reserved for request-level problems
(bad input, auth, not found). The only non-200 success code is **202**.

---

## Authentication

Optional bearer token on **POST** endpoints only (GET endpoints are always open).
Disabled entirely when `DS_API_KEY` is unset (the localhost dev default). When
set, POST requests must carry `Authorization: Bearer <DS_API_KEY>` or receive a
`401`.

---

## Errors

All errors serialize as a single envelope via FastAPI exception handlers:

```json
{ "error": { "type": "NotFound", "message": "...", "detail": null } }
```

| Code | When |
|---|---|
| `400` / `422` | malformed request / Pydantic validation |
| `401` | missing/bad bearer when `DS_API_KEY` is set |
| `404` | run / chain / artifact not found |
| `409` | run_id already exists; run not resumable |
| `503` | `planner=llm` or `decider=llm` requested but the selected provider is missing required credentials |

---

## SSE streaming

`GET /runs/{id}/trace/stream` emits **lifecycle** events by subscribing to the
JobStore bus (hand-rolled `StreamingResponse`, no `sse-starlette` dependency):

```text
event: status  data: {"state": "running"}
event: status  data: {"state": "ok"}
event: done    data: {"status": "ok", "run_id": "..."}
```

This is **progress-level** streaming now. True per-node live streaming is planned:
the JobStore already exposes a subscribe/event-bus seam, so deepening to per-node
events only requires publishing node start/finish onto that same bus from the
runtime node wrappers — the HTTP/SSE layer does not change when that lands.

---

## Examples

Create a run (default `auto` mode) and inspect it:

```bash
# Create a run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "compare A and B"}'

# Async create (returns 202 + run_id immediately)
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt": "compare A and B", "mode": "async", "run_id": "demo-001"}'

# List recorded runs
curl http://localhost:8000/runs

# Merged status (live + on-disk)
curl http://localhost:8000/runs/demo-001

# Recorded sub-resources
curl http://localhost:8000/runs/demo-001/spec
curl http://localhost:8000/runs/demo-001/trace
curl http://localhost:8000/runs/demo-001/output
curl http://localhost:8000/runs/demo-001/graph
curl http://localhost:8000/runs/demo-001/summary
curl http://localhost:8000/runs/demo-001/artifacts
curl http://localhost:8000/runs/demo-001/artifacts/report.md

# Progress SSE stream
curl -N http://localhost:8000/runs/demo-001/trace/stream

# Resume a paused (wait_for_event) run
curl -X POST http://localhost:8000/runs/demo-001/resume \
  -H "Content-Type: application/json" \
  -d '{"event": {"approved": true}}'

# Replay a recorded spec under a new id
curl -X POST http://localhost:8000/runs/demo-001/replay \
  -H "Content-Type: application/json" \
  -d '{"new_run_id": "demo-001-replay"}'

# Iterative chain
curl -X POST http://localhost:8000/chains \
  -H "Content-Type: application/json" \
  -d '{"prompt": "refine this", "max_iterations": 3}'

curl http://localhost:8000/chains/<chain_id>

# Adaptive chain with an LLM iteration judge
curl -X POST http://localhost:8000/chains \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "research and refine this",
    "planner": "llm",
    "provider": "openai",
    "decider": "llm",
    "success_criteria": "Answer with grounded evidence and a clear recommendation.",
    "max_iterations": 3
  }'

# Registry + health
curl http://localhost:8000/registry
curl http://localhost:8000/healthz
```

When `DS_API_KEY` is set, add `-H "Authorization: Bearer <key>"` to every `POST`.
