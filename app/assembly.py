# app/assembly.py
"""Shared builder: turn a RunConfig into a wired Supervisor.

Used by both the CLI (`app/main.py`) and the HTTP API so they construct
identical supervisors. The mock path is token-free (StaticPlanner + mock/echo
runners); the LLM path uses a provider-selected model for planner, workers,
reducers, and subagents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from app.models import DynamicRunState, GraphSpec, NodeKind
from app.policy import ExecutionPolicy
from app.recording import Recorder
from app.runtime import (
    ArtifactSink,
    ChatLlmRunner,
    FileArtifactSink,
    LangGraphExecutor,
    LlmReduceRunner,
    ModelRef,
    NodeRunner,
    build_grounded_tool_runner,
    build_llm_subagents,
    default_model_providers,
    make_emit_artifact_runner,
    make_spawn_subagent_runner,
)
from app.runtime.model_providers import ProviderRegistry
from app.runtime.subgraph import build_spawn_subgraph_runner, make_child_launcher
from app.supervisor import (
    LLMPlanner,
    Planner,
    StaticPlanner,
    Supervisor,
)

_LLM_REDUCE_STRATEGIES = {"concat", "merge_dict", "llm_summarize"}
PlannerMode = Literal["mock", "llm", "openai"]


@dataclass(frozen=True)
class RunConfig:
    """Resolved per-run configuration shared by CLI and API.

    `provider` + `model` define the *base* model used for every role. Any
    role can be overridden with a `ModelRef` (mixing providers — e.g. an
    OpenAI planner with Anthropic workers); unset roles fall back to the
    worker model, which itself falls back to the base ref. This is the
    provider-neutral seam the SDK facade will expose per role.
    """

    planner: PlannerMode
    model: str
    strict_runners: bool
    provider: str = "openai"
    planner_model: ModelRef | None = None
    worker_model: ModelRef | None = None
    reducer_model: ModelRef | None = None
    subagent_model: ModelRef | None = None
    judge_model: ModelRef | None = None

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
    def base_ref(self) -> ModelRef:
        return ModelRef(provider=self.provider, model=self.model)

    # `model_ref` is retained as an alias of `base_ref` for existing callers
    # (API deps, iteration decider) that pass a single ref.
    @property
    def model_ref(self) -> ModelRef:
        return self.base_ref

    @property
    def worker_ref(self) -> ModelRef:
        return self.worker_model or self.base_ref

    @property
    def planner_ref(self) -> ModelRef:
        return self.planner_model or self.worker_ref

    @property
    def reducer_ref(self) -> ModelRef:
        return self.reducer_model or self.worker_ref

    @property
    def subagent_ref(self) -> ModelRef:
        return self.subagent_model or self.worker_ref

    @property
    def judge_ref(self) -> ModelRef:
        return self.judge_model or self.worker_ref

    def providers_in_use(self) -> tuple[str, ...]:
        """Distinct provider names this config will actually build.

        Empty for the mock planner (token-free). Callers use this to check
        credentials for every provider a multi-provider run touches.
        """

        if self.planner == "mock":
            return ()
        names = {
            self.planner_ref.provider,
            self.worker_ref.provider,
            self.reducer_ref.provider,
            self.subagent_ref.provider,
        }
        return tuple(sorted(names))


def mock_llm_runner(
    state: DynamicRunState,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Token-free stand-in: echoes instruction and visible upstream keys."""

    instruction = str(params["instruction"])
    upstream_keys = sorted(state.get("values", {}).keys())
    suffix = f" [seen: {', '.join(upstream_keys)}]" if upstream_keys else ""
    return {"result": f"<mock-llm>{instruction}{suffix}</mock-llm>"}


def _attach_callbacks(ref: ModelRef, callbacks: list[Any] | None) -> ModelRef:
    """Return `ref` with `callbacks` merged into its chat-client kwargs.

    LangChain fires callbacks attached to a model *instance* wherever that
    model runs (including the executor's worker thread), so this is how the
    SDK captures token usage transparently. No-op when `callbacks` is falsy.
    """
    if not callbacks:
        return ref
    extra = dict(ref.extra_kwargs)
    extra["callbacks"] = [*extra.get("callbacks", []), *callbacks]
    return replace(ref, extra_kwargs=extra)


def _build_planner(
    config: RunConfig,
    *,
    model_providers: ProviderRegistry,
    chat_callbacks: list[Any] | None = None,
    planner_guidance: str | None = None,
) -> Planner:
    if config.planner == "llm":
        ref = _attach_callbacks(config.planner_ref, chat_callbacks)
        provider = model_providers.get(ref.provider)
        structured = provider.build_structured_output(ref, GraphSpec)
        return LLMPlanner(
            structured,
            guidance=planner_guidance,
            executable_reduce_strategies=_LLM_REDUCE_STRATEGIES,
        )
    from app.main import build_demo_spec

    return StaticPlanner(build_demo_spec())


def _build_runners(
    config: RunConfig,
    *,
    runs_dir: str,
    model_providers: ProviderRegistry,
    artifact_sink: ArtifactSink | None = None,
    chat_callbacks: list[Any] | None = None,
) -> dict[NodeKind, NodeRunner]:
    sink = artifact_sink or FileArtifactSink(root_dir=runs_dir)
    emit_runner = make_emit_artifact_runner(sink)
    runners: dict[NodeKind, NodeRunner] = {NodeKind.EMIT_ARTIFACT: emit_runner}

    if config.planner == "llm":
        # Build one chat model per distinct role ref, sharing a client when
        # two roles resolve to the same provider+model (the common case).
        # ModelRef carries an extra_kwargs dict, so it isn't hashable — memoize
        # by equality against a small list rather than a dict key.
        built: list[tuple[ModelRef, Any]] = []

        def chat_for(ref: ModelRef) -> Any:
            ref = _attach_callbacks(ref, chat_callbacks)
            for known_ref, chat in built:
                if known_ref == ref:
                    return chat
            chat = model_providers.get(ref.provider).build_chat(ref)
            built.append((ref, chat))
            return chat

        runners[NodeKind.LLM_CALL] = ChatLlmRunner(chat_for(config.worker_ref))
        runners[NodeKind.REDUCE] = LlmReduceRunner(chat_for(config.reducer_ref))
        runners[NodeKind.SPAWN_SUBAGENT] = make_spawn_subagent_runner(
            build_llm_subagents(chat_model=chat_for(config.subagent_ref))
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
    artifact_sink: ArtifactSink | None = None,
    chat_callbacks: list[Any] | None = None,
    policy: ExecutionPolicy | None = None,
    max_plan_attempts: int = 1,
    planner_guidance: str | None = None,
) -> Supervisor:
    """Construct a Supervisor wired for `config`.

    `recorder` is the persistence target. `checkpointer` (e.g. a shared
    MemorySaver) enables resume across calls for graphs with wait_for_event.
    `artifact_sink` overrides where `emit_artifact` nodes write — pass a
    `CollectingArtifactSink` (with a `NullRecorder`) for a no-files run.
    `chat_callbacks` are attached to every chat model built (e.g. a usage
    callback to capture token counts). `policy` is the host-owned
    `ExecutionPolicy` enforced at validation — for the root run **and** every
    nested `spawn_subgraph` child — so a planner can never grant itself a larger
    budget than the host allows.
    """

    effective_policy = policy or ExecutionPolicy()

    runs_dir = str(getattr(recorder, "root_dir", "runs"))
    providers = model_providers or default_model_providers()
    planner = _build_planner(
        config,
        model_providers=providers,
        chat_callbacks=chat_callbacks,
        planner_guidance=planner_guidance,
    )
    runners = _build_runners(
        config,
        runs_dir=runs_dir,
        model_providers=providers,
        artifact_sink=artifact_sink,
        chat_callbacks=chat_callbacks,
    )
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
        make_child_launcher(
            planner=planner,
            executor=executor,
            recorder=recorder,
            policy=effective_policy,
        )
    )
    return Supervisor(
        planner=planner,
        executor=executor,
        recorder=recorder,
        policy=effective_policy,
        max_plan_attempts=max_plan_attempts,
    )
