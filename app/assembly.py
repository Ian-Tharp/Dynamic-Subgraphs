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
