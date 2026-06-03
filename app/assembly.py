# app/assembly.py
"""Shared builder: turn a RunConfig into a wired Supervisor.

Used by both the CLI (`app/main.py`) and the HTTP API so they construct
identical supervisors. The mock path is token-free (StaticPlanner + mock/echo
runners); the LLM path uses a provider-selected model for planner, workers,
reducers, and subagents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from app.models import DynamicRunState, NodeKind
from app.recording import Recorder
from app.runtime import (
    ChatLlmRunner,
    FileArtifactSink,
    LangGraphExecutor,
    LlmReduceRunner,
    ModelRef,
    NodeRunner,
    make_emit_artifact_runner,
    make_spawn_subagent_runner,
    build_grounded_tool_runner,
    build_llm_subagents,
    default_model_providers,
)
from app.runtime.model_providers import ProviderRegistry
from app.runtime.subgraph import build_spawn_subgraph_runner, make_child_launcher
from app.supervisor import (
    LLMPlanner,
    Planner,
    StaticPlanner,
    Supervisor,
)
from app.models import GraphSpec

_LLM_REDUCE_STRATEGIES = {"concat", "merge_dict", "llm_summarize"}
PlannerMode = Literal["mock", "llm", "openai"]


@dataclass(frozen=True)
class RunConfig:
    """Resolved per-run configuration shared by CLI and API."""

    planner: PlannerMode
    model: str
    strict_runners: bool
    provider: str = "openai"

    def __post_init__(self) -> None:
        planner = self.planner
        provider = self.provider.strip().lower()
        if planner == "openai":
            planner = "llm"
            provider = "openai"
        if planner not in ("mock", "llm"):
            raise ValueError("RunConfig.planner must be 'mock', 'llm', or 'openai'")
        if not provider:
            raise ValueError("RunConfig.provider must be non-empty")
        object.__setattr__(self, "planner", planner)
        object.__setattr__(self, "provider", provider)

    @property
    def model_ref(self) -> ModelRef:
        return ModelRef(provider=self.provider, model=self.model)


def mock_llm_runner(
    state: DynamicRunState,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Token-free stand-in: echoes instruction and visible upstream keys."""

    instruction = str(params["instruction"])
    upstream_keys = sorted(state.get("values", {}).keys())
    suffix = f" [seen: {', '.join(upstream_keys)}]" if upstream_keys else ""
    return {"result": f"<mock-llm>{instruction}{suffix}</mock-llm>"}


def _build_planner(
    config: RunConfig,
    *,
    model_providers: ProviderRegistry,
) -> Planner:
    if config.planner == "llm":
        provider = model_providers.get(config.provider)
        structured = provider.build_structured_output(config.model_ref, GraphSpec)
        return LLMPlanner(
            structured,
            executable_reduce_strategies=_LLM_REDUCE_STRATEGIES,
        )
    from app.main import build_demo_spec

    return StaticPlanner(build_demo_spec())


def _build_runners(
    config: RunConfig,
    *,
    runs_dir: str,
    model_providers: ProviderRegistry,
) -> dict[NodeKind, NodeRunner]:
    emit_runner = make_emit_artifact_runner(FileArtifactSink(root_dir=runs_dir))
    runners: dict[NodeKind, NodeRunner] = {NodeKind.EMIT_ARTIFACT: emit_runner}

    if config.planner == "llm":
        provider = model_providers.get(config.provider)
        chat_model = provider.build_chat(config.model_ref)
        runners[NodeKind.LLM_CALL] = ChatLlmRunner(chat_model)
        runners[NodeKind.REDUCE] = LlmReduceRunner(chat_model)
        runners[NodeKind.SPAWN_SUBAGENT] = make_spawn_subagent_runner(
            build_llm_subagents(chat_model=chat_model)
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
    model_providers: ProviderRegistry | None = None,
) -> Supervisor:
    """Construct a Supervisor wired for `config`.

    `recorder` is the persistence target. `checkpointer` (e.g. a shared
    MemorySaver) enables resume across calls for graphs with wait_for_event.
    """

    runs_dir = str(getattr(recorder, "root_dir", "runs"))
    providers = model_providers or default_model_providers()
    planner = _build_planner(config, model_providers=providers)
    runners = _build_runners(config, runs_dir=runs_dir, model_providers=providers)
    executor = LangGraphExecutor(
        runners=runners,
        checkpointer=checkpointer,
        strict_runners=config.strict_runners,
    )
    # Late-bind the spawn_subgraph launcher: a spawn_subgraph node plans + runs a
    # bounded child graph on this same executor (nested, depth-capped). Assigned
    # after the executor exists because the launcher closes over it — and because
    # `runners` is the executor's live dict, the executor picks the entry up, so
    # a child can itself spawn (bounded by the depth ceiling).
    runners[NodeKind.SPAWN_SUBGRAPH] = build_spawn_subgraph_runner(
        make_child_launcher(planner=planner, executor=executor, recorder=recorder)
    )
    return Supervisor(planner=planner, executor=executor, recorder=recorder)
