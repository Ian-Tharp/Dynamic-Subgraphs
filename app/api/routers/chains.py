# app/api/routers/chains.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from app.api.deps import (
    AppContext,
    get_context,
    require_auth,
    require_valid_run_id,
    resolve_chain_decider,
    resolve_run_config,
)
from app.api.errors import Conflict, NotFound
from app.api.jobs import Job, JobState
from app.api.schemas import ChainRequest

router = APIRouter(tags=["chains"])


def _job_state_for_chain(status: str) -> JobState:
    """Map an iterative-chain terminal status onto a job state.

    `ask_user` is a *paused* outcome — the LLM judge needs the human before the
    chain can continue — not a failure. Mirror how /runs maps a paused run, so a
    legitimate clarification request isn't recorded as FAILED.
    """
    if status in {"ok", "stopped", "max_iterations"}:
        return JobState.OK
    if status == "ask_user":
        return JobState.PAUSED
    return JobState.FAILED


def _decision_payload(decision: Any | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "action": decision.action,
        "reason": decision.reason,
        "success_criteria_met": decision.success_criteria_met,
        "gaps": list(decision.gaps),
        "next_prompt": decision.next_prompt,
        "question_to_user": decision.question_to_user,
    }


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
                "decision_detail": _decision_payload(s.decision),
                "reason": s.decision.reason,
            }
            for s in result.steps
        ],
        "final_decision": _decision_payload(result.final_decision),
    }


def _make_chain_worker(
    ctx: AppContext,
    config,
    prompt: str,
    run_id: str,
    max_iter: int,
    decider,
):
    supervisor = ctx.supervisor_for(config)

    def work(job: Job) -> None:
        job.set_state(JobState.RUNNING)
        result = supervisor.run_iteratively(
            prompt,
            run_id=run_id,
            max_iterations=max_iter,
            decider=decider,
        )
        job.complete(result=result, state=_job_state_for_chain(result.status))

    return work


@router.post("/chains")
def create_chain(
    request: Request,
    body: ChainRequest,
    _: None = Depends(require_auth),
) -> Response:
    ctx = get_context(request)
    config = resolve_run_config(
        ctx,
        planner=body.planner,
        provider=body.provider,
        model=body.model,
    )
    decider = resolve_chain_decider(
        ctx,
        config=config,
        decider=body.decider,
        success_criteria=body.success_criteria,
        judge_failed_runs=body.judge_failed_runs,
    )
    run_id = body.run_id or f"chain-{__import__('uuid').uuid4().hex[:12]}"
    require_valid_run_id(run_id)
    if ctx.jobs.get(run_id) is not None or ctx.recorder.exists(run_id):
        raise Conflict(f"chain id already exists: {run_id!r}")

    job = ctx.jobs.create(run_id, kind="chain")
    ctx.jobs.submit(
        job,
        _make_chain_worker(
            ctx,
            config,
            body.prompt,
            run_id,
            body.max_iterations,
            decider,
        ),
    )

    if body.mode == "async":
        return JSONResponse(
            status_code=202,
            content={
                "chain_id": run_id,
                "status": job.state.value,
                "links": {"self": f"/chains/{run_id}"},
            },
        )

    timeout = (
        ctx.settings.max_sync_seconds
        if body.mode == "sync"
        else ctx.settings.auto_sync_seconds
    )
    finished = job.wait(timeout=timeout)
    if finished and job.result is not None:
        return JSONResponse(status_code=200, content=_chain_payload(job.result))
    if finished:
        # The worker crashed before producing a result — surface the failure
        # and its message instead of a 202 pointing at nothing.
        return JSONResponse(
            status_code=200,
            content={
                "chain_id": run_id,
                "status": "failed",
                "error": job.error,
                "links": {"self": f"/chains/{run_id}"},
            },
        )
    return JSONResponse(
        status_code=202,
        content={
            "chain_id": run_id,
            "status": job.state.value,
            "links": {"self": f"/chains/{run_id}"},
        },
    )


@router.get("/chains/{chain_id}")
def get_chain(request: Request, chain_id: str) -> dict[str, Any]:
    ctx = get_context(request)
    require_valid_run_id(chain_id)
    job = ctx.jobs.get(chain_id)
    if job is not None and job.result is not None:
        return _chain_payload(job.result)
    if job is not None and job.is_terminal():
        # Terminal without a result: the chain worker crashed.
        return {
            "chain_id": chain_id,
            "status": "failed",
            "error": job.error,
            "links": {"self": f"/chains/{chain_id}"},
        }
    try:
        return ctx.recorder.load_chain(chain_id)
    except FileNotFoundError as exc:
        raise NotFound(f"No chain {chain_id!r}") from exc
