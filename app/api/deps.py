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
from app.supervisor import (
    IterationDecider,
    StatusIterationDecider,
    Supervisor,
    build_openai_iteration_decider,
)


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


def resolve_chain_decider(
    ctx: AppContext,
    *,
    config: RunConfig,
    decider: str,
    success_criteria: str | None,
    judge_failed_runs: bool,
) -> IterationDecider:
    """Resolve the chain-level orchestration judge for `/chains`.

    The status decider is token-free and remains the default. The LLM decider is
    explicit because it makes an additional model call after each successful
    iteration to decide whether to stop, replan, ask the user, or fail.
    """

    if decider == "status":
        return StatusIterationDecider()

    if decider == "llm":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ServiceUnavailable(
                "decider=llm requested but OPENAI_API_KEY is not set"
            )
        if not ctx.settings.is_model_allowed(config.model):
            raise BadRequest(
                f"Model {config.model!r} is not in the allowlist "
                f"{list(ctx.settings.model_allowlist)}"
            )
        return build_openai_iteration_decider(
            model=config.model,
            success_criteria=success_criteria,
            judge_failed_runs=judge_failed_runs,
        )

    raise BadRequest(f"Unknown chain decider {decider!r}")


def require_auth(request: Request) -> None:
    """Guard for POST endpoints. No-op when DS_API_KEY is unset."""
    ctx: AppContext = request.app.state.context
    expected = ctx.settings.api_key
    if not expected:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {expected}":
        raise Unauthorized("Missing or invalid bearer token")
