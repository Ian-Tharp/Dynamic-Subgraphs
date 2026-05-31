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
