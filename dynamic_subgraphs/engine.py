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
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from app.assembly import RunConfig, build_supervisor
from app.models import GraphSpec
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
    PLANNERS,
    RUN_STATUSES,
    STRUCTURED_METHODS,
    Planner,
    RunStatus,
)

# `Model` is the public, SDK-facing name for a concrete model choice.
Model = ModelRef

# Anything the engine accepts as a recording policy.
RecordingInput = bool | Artifact | Iterable[Artifact] | Recording

__all__ = [
    "Artifact",
    "DynamicSubgraphs",
    "EngineConfig",
    "Model",
    "ModelSelection",
    "Recording",
    "RunResult",
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
        }

    @classmethod
    def _from_supervisor(
        cls,
        result: Any,
        *,
        runs_dir: str | Path,
        run_id: str,
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

        return cls(
            run_id=run_id,
            status=result.status,
            response=result.response,
            values=values,
            plan=result.validated_spec,
            artifacts=artifacts,
            errors=list(result.errors or []),
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

        One call enumerates providers, planners, model roles, artifact ids,
        recording presets, run statuses, structured-output methods, and the
        `Model` convenience constructors — so a caller never has to guess an
        option string. JSON-safe (all values are plain lists of strings).
        """
        registry = providers or default_model_providers()
        return {
            "providers": list(registry.names()),
            "planners": list(PLANNERS),
            "model_roles": list(MODEL_ROLES),
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
            >>> engine = DynamicSubgraphs(model=Model("openai", "gpt-5.4-nano"))
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

        supervisor = build_supervisor(
            config,
            recorder=recorder,
            checkpointer=self._checkpointer,
            model_providers=self._providers,
            artifact_sink=artifact_sink,
        )
        result = supervisor.run(prompt, run_id=run_id)
        return RunResult._from_supervisor(
            result, runs_dir=self._runs_dir, run_id=run_id
        )
