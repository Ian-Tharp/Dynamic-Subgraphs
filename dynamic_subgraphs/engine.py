"""Public SDK facade over the governed dynamic-graph Supervisor.

`DynamicSubgraphs` turns the multi-step `RunConfig` + recorder + supervisor
wiring into a configured object plus a one-line `run()`. Models are bring
-your-own: each `Model` (a `ModelRef`) carries its own provider, name,
`base_url`, `api_key`, and structured-output method, so a caller can target
OpenAI, Anthropic, a local LM Studio / Ollama server, or a proxy without any
global env setup.

Model selection is layered: the engine holds defaults, and every `run()` may
override any role for that single call — so each graph run can determine the
models used for its subsequent node calls.

    from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

    engine = DynamicSubgraphs(EngineConfig(model=Model("openai", "gpt-5.4-nano")))
    result = engine.run("Compare two sources on X and recommend one.")
    print(result.response)      # synthesized answer
    print(result.values)        # {output_key: value}
    print(result.plan.graph_id) # the GraphSpec that was generated
    print(result.artifacts)     # {filename: Path} under runs/<run_id>/

Recording is opt-in and granular. By default the engine writes **no files** —
nothing is persisted to `runs/`, so embedding it in another app never clutters
that app's working tree. Suggestion: set a `recording` policy while developing
or debugging to capture artifacts under ``runs/<run_id>/`` for inspection and
replay; leave it at the default in production / library use.

    from dynamic_subgraphs import Recording
    engine = DynamicSubgraphs(EngineConfig(model=Model(...),
                                           recording=Recording.debug()))
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler

from app.assembly import RunConfig, build_supervisor
from app.models import GraphSpec
from app.policy import ExecutionPolicy
from app.recording import FileRecorder, NullRecorder
from app.runtime import (
    CollectingArtifactSink,
    ModelRef,
    ProviderRegistry,
    default_model_providers,
)
from dynamic_subgraphs.recording import Artifact, Recording
from dynamic_subgraphs.types import (
    MODEL_ROLES,
    NODE_KINDS,
    PLANNERS,
    RUN_STATUSES,
    STRUCTURED_METHODS,
    Planner,
    RunStatus,
)
from dynamic_subgraphs.usage import TokenUsage

# `Model` is the public, SDK-facing name for a concrete model choice.
Model = ModelRef

# Anything the engine accepts as a recording policy.
RecordingInput = bool | Artifact | Iterable[Artifact] | Recording

# Manual price override: {model_name: {"input_per_1m": float, "output_per_1m": float}}.
# Keyed by the model alias (also matches dated snapshots by prefix). Usually you
# don't need this — install the `cost` extra and cost is computed automatically
# from LiteLLM's maintained price map. Use this to override, or to price local /
# custom-endpoint models LiteLLM doesn't know.
Pricing = Mapping[str, Mapping[str, float]]

__all__ = [
    "Artifact",
    "DynamicSubgraphs",
    "EngineConfig",
    "Model",
    "ModelSelection",
    "Recording",
    "RunResult",
    "TokenUsage",
]


@dataclass(frozen=True)
class ModelSelection:
    """A layered, per-role model choice.

    `model` is the base used for every role; any role left unset falls back to
    the worker model, which falls back to the base — the same precedence the
    underlying `RunConfig` uses.
    """

    model: Model | None = None
    planner_model: Model | None = None
    worker_model: Model | None = None
    reducer_model: Model | None = None
    subagent_model: Model | None = None
    judge_model: Model | None = None

    def merge(self, override: ModelSelection) -> ModelSelection:
        """Return self with any non-None field of `override` applied on top."""

        changed = {
            name: value for name, value in vars(override).items() if value is not None
        }
        return replace(self, **changed) if changed else self

    def _worker(self) -> Model | None:
        return self.worker_model or self.model

    def to_run_config(self, planner: str) -> RunConfig:
        if planner not in PLANNERS:
            raise ValueError(
                f"Unknown planner {planner!r}. Valid planners: {', '.join(PLANNERS)}."
            )
        if planner == "mock":
            base = self.model or self._worker()
            return RunConfig(
                planner="mock",
                provider=base.provider if base else "openai",
                model=base.model if base else "mock-model",
                strict_runners=False,
            )

        worker = self._worker()
        if worker is None:
            raise ValueError(
                "DynamicSubgraphs needs a model for LLM runs — pass model=... "
                "to the engine or to run()."
            )
        planner_ref = self.planner_model or worker
        return RunConfig(
            planner="llm",
            provider=worker.provider,
            model=worker.model,
            strict_runners=True,
            planner_model=planner_ref,
            worker_model=worker,
            reducer_model=self.reducer_model or worker,
            subagent_model=self.subagent_model or worker,
            judge_model=self.judge_model or worker,
        )


def _price_for(name: str, pricing: Pricing) -> Mapping[str, float] | None:
    """Look up a model's price, tolerating dated snapshots.

    Providers report the resolved model name (e.g. `gpt-5.4-nano-2026-03-17`)
    in usage, but devs price by the alias they passed (`gpt-5.4-nano`). Match
    exactly first, then the longest alias that is a prefix of the reported name.
    """
    if name in pricing:
        return pricing[name]
    prefixes = [k for k in pricing if name.startswith(k)]
    return pricing[max(prefixes, key=len)] if prefixes else None


def _litellm_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost via LiteLLM's maintained price map (the `cost` extra), or None.

    Computed locally from the token counts — no network call. Returns None if
    `litellm` isn't installed, or doesn't recognize the model.
    """
    try:
        import litellm
    except ImportError:
        return None
    litellm.suppress_debug_info = True  # quiet its stderr chatter on a miss
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=int(input_tokens),
            completion_tokens=int(output_tokens),
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception:
        return None


def _compute_cost(usage: TokenUsage, pricing: Pricing | None) -> float | None:
    """Total USD cost, or None if no price source resolved any model.

    Per model, a manual `pricing` entry wins (override / local/custom models);
    otherwise LiteLLM's price map is used when the `cost` extra is installed.
    Models neither source can price are skipped, so a partial price book and a
    mixed-provider run both work.
    """
    total = 0.0
    priced = False
    for name, u in usage.by_model.items():
        inp, out = u["input_tokens"], u["output_tokens"]
        price = _price_for(name, pricing) if pricing else None
        if price is not None:
            total += inp / 1_000_000 * float(price.get("input_per_1m", 0.0))
            total += out / 1_000_000 * float(price.get("output_per_1m", 0.0))
            priced = True
            continue
        auto = _litellm_cost(name, inp, out)
        if auto is not None:
            total += auto
            priced = True
    return round(total, 6) if priced else None


@dataclass
class RunResult:
    """The outcome of one `DynamicSubgraphs.run()` call.

    Attributes:
        run_id: the id this run was recorded under.
        status: one of `RUN_STATUSES` (e.g. "ok", "plan_failed",
            "execution_failed", "paused"). Use `.ok` for the common check.
        response: human-readable summary of the outcome.
        values: `{output_key: value}` produced by the graph's nodes.
        plan: the generated `GraphSpec` (Pydantic; `plan.model_dump()` to
            serialize), or None if planning failed.
        artifacts: `{filename: Path}` written to disk — empty unless recording
            selected that artifact.
        errors: list of `{"stage": str, "type": str, "message": str}` entries
            (node-level failures additionally carry "node_id").
        usage: exact `TokenUsage` for the run (always populated; zero for mock
            runs). Includes a per-model breakdown.
        cost: total USD cost. Auto-computed from LiteLLM's price map when the
            `cost` extra is installed; or from `EngineConfig(pricing=...)`;
            else None. (Tokens are always exact and free — cost needs prices.)
        effective_budget: the host-*granted* budget for this run — the planner's
            request capped by the `ExecutionPolicy` (`min(host, request)` per
            field). None if planning/validation failed. Equals `plan.budget`.
        plan_attempts: how many times the planner ran (1 unless the plan-repair
            loop re-planned after a recoverable validation failure).

    Use `to_dict()` for a JSON-safe view (handy for logging or handing to
    another tool/agent).
    """

    run_id: str
    status: RunStatus
    response: str | None
    values: dict[str, Any] = field(default_factory=dict)
    plan: GraphSpec | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: float | None = None
    effective_budget: dict[str, Any] | None = None
    plan_attempts: int = 1

    @property
    def ok(self) -> bool:
        """True iff the run completed successfully (`status == "ok"`)."""
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable view of this result (Path -> str, plan -> dict)."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "ok": self.ok,
            "response": self.response,
            "values": self.values,
            "plan": (
                self.plan.model_dump(mode="json", by_alias=True)
                if self.plan is not None
                else None
            ),
            "artifacts": {name: str(path) for name, path in self.artifacts.items()},
            "errors": self.errors,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
                "by_model": self.usage.by_model,
            },
            "cost": self.cost,
            "effective_budget": self.effective_budget,
            "plan_attempts": self.plan_attempts,
        }

    @classmethod
    def _from_supervisor(
        cls,
        result: Any,
        *,
        runs_dir: str | Path,
        run_id: str,
        usage: TokenUsage,
        cost: float | None,
    ) -> RunResult:
        values: dict[str, Any] = {}
        if result.result is not None:
            values = dict(result.result.state.get("values", {}))

        artifacts: dict[str, Path] = {}
        run_dir = Path(runs_dir) / run_id
        if run_dir.is_dir():
            for path in sorted(run_dir.iterdir()):
                if path.is_file():
                    artifacts[path.name] = path

        # The validated spec's budget is the *granted* (host-resolved) budget —
        # the validator stamped it. Surface it so a caller can see what the host
        # actually allowed vs. what the planner requested (plan.budget == this).
        effective_budget: dict[str, Any] | None = None
        if result.validated_spec is not None:
            effective_budget = result.validated_spec.budget.model_dump(mode="json")

        return cls(
            run_id=run_id,
            status=result.status,
            response=result.response,
            values=values,
            plan=result.validated_spec,
            artifacts=artifacts,
            errors=list(result.errors or []),
            usage=usage,
            cost=cost,
            effective_budget=effective_budget,
            plan_attempts=getattr(result, "plan_attempts", 1),
        )


@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for a `DynamicSubgraphs` engine.

    Build one and pass it in: ``DynamicSubgraphs(EngineConfig(...))``. This is
    the single configuration surface — model roles, recording policy, planner
    mode, run directory, provider registry, and checkpointer all live here.

    Model roles layer the same way as elsewhere: an unset role falls back to
    `worker_model`, which falls back to `model` (the base). Per-call overrides
    on `run()` take precedence over this config for that one run.

    Example:
        >>> from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model, Recording
        >>> config = EngineConfig(
        ...     model=Model.lmstudio("google/gemma-3-27b"),
        ...     recording=Recording.visual_only(),
        ... )
        >>> engine = DynamicSubgraphs(config)

    Note:
        Use a capable model for the **planner** role. Small local models
        (e.g. 7B-class, and in practice models below ~20-30B) frequently emit
        invalid `GraphSpec`s and fail planning. Run them as `worker_model`
        with a stronger `planner_model` instead. See `docs/recipes.md`.
    """

    model: Model | None = None
    planner_model: Model | None = None
    worker_model: Model | None = None
    reducer_model: Model | None = None
    subagent_model: Model | None = None
    judge_model: Model | None = None
    planner: Planner = "llm"
    recording: RecordingInput = False
    runs_dir: str | Path = "runs"
    providers: ProviderRegistry | None = None
    checkpointer: Any | None = None
    # Host-owned execution governance. The planner may *request* a budget, but
    # the effective limit per field is min(host ceiling, request). Defaults are
    # permissive-but-bounded (they match the historical GraphBudget defaults),
    # so leaving this unset preserves prior behavior for typical plans. Set a
    # stricter `ExecutionPolicy(...)` to cap budgets the planner can never widen.
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    # Plan-repair loop. When a plan is rejected by the validator for a
    # *recoverable* reason (e.g. a budget overrun or a dangling edge), the
    # issues + host limits are fed back into a re-plan, up to this many planner
    # attempts. Default 2 (repair once) makes plans "just work" more often; set
    # 1 for strict block-and-report on the first validation failure.
    max_plan_attempts: int = 2
    # Optional manual price override for cost, e.g.
    # {"gpt-5.4-nano": {"input_per_1m": 0.2, "output_per_1m": 1.25}}.
    # With the `cost` extra installed, cost is computed automatically (LiteLLM)
    # and you usually don't need this — set it only to override a price or to
    # cover local / custom-endpoint models LiteLLM doesn't know.
    pricing: Pricing | None = None
    # Optional domain steering appended to the planner's system prompt (the
    # GraphSpec contract is preserved). Bias planning — e.g. "prefer parallel_map
    # over deep recursion for worldbuilding" — without owning the whole prompt.
    # None leaves the default planner prompt unchanged.
    planner_guidance: str | None = None

    def model_selection(self) -> ModelSelection:
        """The per-role `ModelSelection` this config resolves to."""
        return ModelSelection(
            model=self.model,
            planner_model=self.planner_model,
            worker_model=self.worker_model,
            reducer_model=self.reducer_model,
            subagent_model=self.subagent_model,
            judge_model=self.judge_model,
        )

    def recording_policy(self) -> Recording:
        """The normalized `Recording` this config resolves to."""
        return Recording.coerce(self.recording)


class DynamicSubgraphs:
    """Configured engine for planning + running governed dynamic subgraphs.

    Construct from an `EngineConfig`:

        >>> from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model
        >>> engine = DynamicSubgraphs(EngineConfig(model=Model("openai", "gpt-5.4-nano")))
        >>> result = engine.run("Compare two sources and recommend one.")
    """

    @classmethod
    def capabilities(cls, providers: ProviderRegistry | None = None) -> dict[str, Any]:
        """Machine-readable map of every valid option, for agents/tools.

        One call enumerates providers, planners, model roles, node kinds,
        artifact ids, recording presets, run statuses, structured-output
        methods, and the `Model` convenience constructors — so a caller never
        has to guess an option string. JSON-safe (all values are plain lists of
        strings).
        """
        registry = providers or default_model_providers()
        return {
            "providers": list(registry.names()),
            "planners": list(PLANNERS),
            "model_roles": list(MODEL_ROLES),
            "node_kinds": list(NODE_KINDS),
            "artifacts": [a.value for a in Artifact],
            "recording_presets": [
                "none",
                "all",
                "debug",
                "visual_only",
                "replayable",
            ],
            "statuses": list(RUN_STATUSES),
            "structured_methods": list(STRUCTURED_METHODS),
            "model_constructors": ["lmstudio", "ollama", "openai_compatible"],
        }

    def __init__(self, config: EngineConfig | None = None) -> None:
        """Build an engine from an `EngineConfig` (defaults to an empty one)."""
        config = config or EngineConfig()
        self._config = config
        self._defaults = config.model_selection()
        self._planner = config.planner
        self._recording = config.recording_policy()
        self._runs_dir = str(config.runs_dir)
        self._providers = config.providers or default_model_providers()
        self._checkpointer = config.checkpointer
        self._pricing = config.pricing
        self._policy = config.policy
        self._max_plan_attempts = config.max_plan_attempts
        self._planner_guidance = config.planner_guidance

    @property
    def config(self) -> EngineConfig:
        """The `EngineConfig` this engine was built from."""
        return self._config

    def run(
        self,
        prompt: str,
        *,
        run_id: str | None = None,
        planner: str | None = None,
        record: RecordingInput | None = None,
        model: Model | None = None,
        planner_model: Model | None = None,
        worker_model: Model | None = None,
        reducer_model: Model | None = None,
        subagent_model: Model | None = None,
        judge_model: Model | None = None,
    ) -> RunResult:
        """Plan, run, and record one dynamic subgraph for `prompt`.

        Args:
            prompt: the task to plan and execute.
            run_id: id to record under; auto-generated if omitted.
            planner: "llm" (default) or "mock" (token-free). Unknown values
                raise ValueError listing the valid planners.
            record: recording policy for this run only — `True`/`False`, an
                `Artifact`, a set of them, or a `Recording`. It *replaces* the
                engine default (does not merge); `None` inherits the engine's.
            model / planner_model / worker_model / reducer_model /
            subagent_model / judge_model: per-run model overrides. An unset
                role falls back to the worker model, then to the base `model`.

        Returns:
            A `RunResult` — check `.ok`, read `.response`/`.values`, inspect
            the generated `.plan` (a `GraphSpec`), and `.artifacts` (only
            populated for recorded artifacts). `.to_dict()` for JSON.

        Example:
            >>> engine = DynamicSubgraphs(EngineConfig(model=Model("openai", "gpt-5.4-nano")))
            >>> r = engine.run("Compare A and B.", record=Recording.visual_only())
            >>> r.ok, "graph.mmd" in r.artifacts
            (True, True)

        Tips:
            - Mix providers per run: planner_model on a cloud model, worker_model
              on `Model.lmstudio(...)`.
            - Failures don't raise — branch on `result.status` / `result.ok`
              and read `result.errors`.
        """

        selection = self._defaults.merge(
            ModelSelection(
                model=model,
                planner_model=planner_model,
                worker_model=worker_model,
                reducer_model=reducer_model,
                subagent_model=subagent_model,
                judge_model=judge_model,
            )
        )
        config = selection.to_run_config(planner or self._planner)
        run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"

        # Recording is opt-in and granular. By default (none) the engine
        # writes no files; a policy selects exactly which artifacts persist.
        # `records_anything()` decides FileRecorder (with a filename filter)
        # vs NullRecorder; `emits()` decides whether emit_artifact node
        # outputs hit disk (FileArtifactSink) or stay in memory.
        policy = self._recording if record is None else Recording.coerce(record)
        if policy.records_anything():
            recorder: Any = FileRecorder(
                root_dir=Path(self._runs_dir),
                overwrite=True,
                selection=policy.recorder_filenames(),
            )
        else:
            recorder = NullRecorder(root_dir=Path(self._runs_dir))
        artifact_sink: Any = None if policy.emits() else CollectingArtifactSink()

        # Exact token usage comes straight from the providers via LangChain's
        # usage callback, attached to every chat model this run builds (it
        # fires inside the executor's worker thread, where a context-manager
        # callback wouldn't reach).
        usage_handler = UsageMetadataCallbackHandler()

        supervisor = build_supervisor(
            config,
            recorder=recorder,
            checkpointer=self._checkpointer,
            model_providers=self._providers,
            artifact_sink=artifact_sink,
            chat_callbacks=[usage_handler],
            policy=self._policy,
            max_plan_attempts=self._max_plan_attempts,
            planner_guidance=self._planner_guidance,
        )
        result = supervisor.run(prompt, run_id=run_id)

        usage = TokenUsage.from_handler(usage_handler)
        cost = _compute_cost(usage, self._pricing)
        return RunResult._from_supervisor(
            result,
            runs_dir=self._runs_dir,
            run_id=run_id,
            usage=usage,
            cost=cost,
        )
