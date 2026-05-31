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
