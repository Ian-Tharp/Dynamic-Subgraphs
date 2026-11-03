"""Governed run service behind the MCP server.

Everything an MCP client can do goes through `GovernedRunService`. It holds no
MCP types — the server module is a thin adapter over it — so the governance
rules are testable without the MCP SDK installed.

What "governed" means at this boundary:

- **Host-owned limits.** Every run is validated against the server's
  `ExecutionPolicy`. A client may *narrow* the budget for one run
  (`max_nodes`, `max_llm_calls`, `max_wall_seconds`) but can never widen it —
  the same `min(host, request)` rule the planner is held to.
- **No token spend the host didn't opt into.** A client can always choose the
  token-free mock planner. It can choose the LLM planner only when the server
  itself is configured for it, and only with an allowlisted model.
- **Every run is recorded.** Recording is always on (successes and failures),
  so each run an agent triggers leaves a spec, trace, and output under
  `runs/<run_id>/` that can be inspected, replayed, or resumed.
- **No clobbering.** Client-supplied run ids are validated and must be new;
  a run or replay can never overwrite another run's audit trail.
- **Bounded reads.** Run files are read by name from a fixed set; artifacts go
  through the recorder's path-safe lookup.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler

from app.assembly import RunConfig, build_supervisor
from app.policy import ExecutionPolicy
from app.recording import FileRecorder
from app.recording.run_config import load_run_config, save_run_config
from app.registry import (
    DEFAULT_SUBAGENTS,
    DEFAULT_TOOLS,
    FORBIDDEN_KINDS,
    default_kind_definitions,
)
from app.run_id import validate_run_id
from app.runtime import (
    MissingModelProviderCredential,
    ProviderRegistry,
    default_model_providers,
)
from app.supervisor import Supervisor, SupervisorResult

from .settings import McpSettings

# Files a client may read from a run directory, keyed by a short name.
RUN_FILES: dict[str, str] = {
    "spec": "spec.json",
    "output": "output.json",
    "trace": "trace.jsonl",
    "graph": "graph.mmd",
    "summary": "summary.md",
    "prompt": "prompt.txt",
    "eval": "eval.json",
}
_MAX_FILE_CHARS = 200_000
# Statuses where the run did what was asked (or is waiting on the caller).
_HEALTHY_STATUSES = frozenset({"ok", "paused"})


class GovernanceError(ValueError):
    """A request the host's governance refuses (bad input, disallowed option)."""


class NotFoundError(LookupError):
    """The requested run or file does not exist."""


def _new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def _clip(value: Any, limit: int) -> Any:
    """Recursively shorten long strings so tool results stay context-sized."""

    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return f"{value[:limit]}… [truncated {len(value) - limit} chars]"
    if isinstance(value, dict):
        return {k: _clip(v, limit) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clip(v, limit) for v in value]
    return value


def _usage_dict(handler: UsageMetadataCallbackHandler) -> dict[str, Any]:
    by_model: dict[str, dict[str, int]] = {}
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for name, usage in (handler.usage_metadata or {}).items():
        i = int(usage.get("input_tokens", 0))
        o = int(usage.get("output_tokens", 0))
        t = int(usage.get("total_tokens", i + o))
        by_model[name] = {"input_tokens": i, "output_tokens": o, "total_tokens": t}
        totals["input_tokens"] += i
        totals["output_tokens"] += o
        totals["total_tokens"] += t
    return {**totals, "by_model": by_model}


def _policy_dict(policy: ExecutionPolicy) -> dict[str, Any]:
    def allow(value: frozenset[Any] | None) -> list[str] | None:
        return None if value is None else sorted(str(v) for v in value)

    return {
        "max_nodes": policy.max_nodes,
        "max_llm_calls": policy.max_llm_calls,
        "max_depth": policy.max_depth,
        "max_fanout": policy.max_fanout,
        "max_wall_seconds": policy.max_wall_seconds,
        "allowed_tools": allow(policy.allowed_tools),
        "allowed_subagents": allow(policy.allowed_subagents),
        "allowed_node_kinds": allow(policy.allowed_node_kinds),
    }


def narrow_policy(
    host: ExecutionPolicy,
    *,
    max_nodes: int | None = None,
    max_llm_calls: int | None = None,
    max_wall_seconds: int | None = None,
) -> ExecutionPolicy:
    """Return `host` tightened by the caller's requested limits.

    Each requested value is capped at the host ceiling, so the result can only
    be stricter than `host`. Out-of-range requests are refused rather than
    clamped silently.
    """

    def cap(name: str, requested: int | None, floor: int) -> int:
        ceiling = int(getattr(host, name))
        if requested is None:
            return ceiling
        if requested < floor:
            raise GovernanceError(f"{name} must be >= {floor}, got {requested}")
        return min(ceiling, int(requested))

    narrowed = replace(
        host,
        max_nodes=cap("max_nodes", max_nodes, 1),
        max_llm_calls=cap("max_llm_calls", max_llm_calls, 0),
        max_wall_seconds=cap("max_wall_seconds", max_wall_seconds, 1),
    )
    return host if narrowed == host else narrowed


class GovernedRunService:
    """Plan, run, resume, replay, and inspect governed runs for MCP clients."""

    def __init__(
        self,
        settings: McpSettings,
        *,
        model_providers: ProviderRegistry | None = None,
        checkpointer: Any | None = None,
        supervisor_factory: Callable[..., Supervisor] | None = None,
    ) -> None:
        if checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
        self.settings = settings
        self.recorder = FileRecorder(root_dir=settings.runs_dir, overwrite=True)
        self.model_providers = model_providers or default_model_providers()
        self._checkpointer = checkpointer
        self._supervisor_factory = supervisor_factory or build_supervisor
        # Ids claimed by in-flight runs, so two concurrent calls can't both pass
        # the "is this id new?" check and then write into the same directory.
        self._reserved: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ info

    def server_info(self) -> dict[str, Any]:
        """The host configuration a client is held to."""

        s = self.settings
        return {
            "planner": s.planner,
            "provider": s.provider,
            "model": s.model,
            "model_allowlist": list(s.model_allowlist),
            "llm_planner_available": s.planner == "llm",
            "policy": _policy_dict(s.policy),
            "max_plan_attempts": s.max_plan_attempts,
            "runs_dir": str(self.recorder.root_dir),
            "recording": "always on (every run, including failures)",
        }

    def describe_registry(self) -> dict[str, Any]:
        """The node kinds, tools, and subagents a plan may draw on."""

        policy = self.settings.policy
        kinds: list[dict[str, Any]] = []
        for kind, definition in default_kind_definitions().items():
            if (
                policy.allowed_node_kinds is not None
                and kind not in policy.allowed_node_kinds
            ):
                continue
            kinds.append(
                {
                    "kind": kind.value,
                    "description": definition.description,
                    "counts_as_llm_call": definition.counts_as_llm_call,
                    "has_side_effects": definition.has_side_effects,
                }
            )
        tools = set(DEFAULT_TOOLS)
        if policy.allowed_tools is not None:
            tools &= policy.allowed_tools
        subagents = set(DEFAULT_SUBAGENTS)
        if policy.allowed_subagents is not None:
            subagents &= policy.allowed_subagents
        return {
            "node_kinds": kinds,
            "tools": sorted(tools),
            "subagents": sorted(subagents),
            "forbidden_kinds": sorted(FORBIDDEN_KINDS),
        }

    # ------------------------------------------------------------------ runs

    def run(
        self,
        prompt: str,
        *,
        run_id: str | None = None,
        planner: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_nodes: int | None = None,
        max_llm_calls: int | None = None,
        max_wall_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Plan, validate, execute, and record one governed subgraph run."""

        if not prompt or not prompt.strip():
            raise GovernanceError("prompt must be a non-empty string")
        config = self._resolve_config(planner=planner, provider=provider, model=model)
        policy = narrow_policy(
            self.settings.policy,
            max_nodes=max_nodes,
            max_llm_calls=max_llm_calls,
            max_wall_seconds=max_wall_seconds,
        )
        chosen_id = run_id or _new_run_id()
        with self._claim(chosen_id):
            handler = UsageMetadataCallbackHandler()
            supervisor = self._supervisor(config, policy=policy, callbacks=[handler])
            result = supervisor.run(prompt, run_id=chosen_id)
            self._save_config(chosen_id, config)
        return self._payload(result, usage=_usage_dict(handler), policy=policy)

    def resume(self, run_id: str, *, event: Any) -> dict[str, Any]:
        """Feed the awaited event to a paused run and continue it."""

        self._require_run(run_id)
        config = self._persisted_config(run_id)
        with self._exclusive(run_id):
            handler = UsageMetadataCallbackHandler()
            supervisor = self._supervisor(
                config, policy=self.settings.policy, callbacks=[handler]
            )
            result = supervisor.resume(run_id, event=event)
        return self._payload(
            result, usage=_usage_dict(handler), policy=self.settings.policy
        )

    def replay(self, run_id: str, *, new_run_id: str | None = None) -> dict[str, Any]:
        """Re-execute a recorded plan verbatim under a new run id.

        The recorded spec is re-validated against the *current* host policy, so
        a plan recorded under looser limits will not replay.
        """

        self._require_run(run_id)
        config = self._persisted_config(run_id)
        target = new_run_id or f"{run_id}-replay-{uuid.uuid4().hex[:6]}"
        with self._claim(target):
            handler = UsageMetadataCallbackHandler()
            supervisor = self._supervisor(
                config, policy=self.settings.policy, callbacks=[handler]
            )
            result = supervisor.replay(run_id, new_run_id=target)
            self._save_config(target, config)
        payload = self._payload(
            result, usage=_usage_dict(handler), policy=self.settings.policy
        )
        payload["replayed_from"] = run_id
        return payload

    # ----------------------------------------------------------------- reads

    def list_runs(self, *, limit: int = 50) -> dict[str, Any]:
        if limit < 1:
            raise GovernanceError("limit must be >= 1")
        runs = self.recorder.list_runs()
        return {"count": len(runs), "runs": runs[-limit:]}

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Status, values, errors, and the list of readable files for a run."""

        self._require_run(run_id)
        run_dir = self.recorder.run_dir(run_id)
        output: dict[str, Any] = {}
        try:
            output = self.recorder.load_output(run_id)
        except (FileNotFoundError, json.JSONDecodeError):
            output = {}
        art_dir = run_dir / "artifacts"
        artifacts = (
            sorted(p.name for p in art_dir.iterdir()) if art_dir.is_dir() else []
        )
        files = [key for key, name in RUN_FILES.items() if (run_dir / name).exists()]
        limit = self.settings.max_value_chars
        return {
            "run_id": run_id,
            "ok": bool(output.get("ok")) if output else None,
            "error": output.get("error"),
            "values": _clip(output.get("values", {}), limit),
            "errors": _clip(output.get("errors", []), limit),
            "config": load_run_config(run_dir),
            "eval": self.recorder.load_eval(run_id),
            "files": files,
            "artifacts": artifacts,
        }

    def get_run_file(self, run_id: str, name: str) -> str:
        """Read one recorded file (`spec`, `trace`, `graph`, ...) as text."""

        filename = RUN_FILES.get(name)
        if filename is None:
            raise GovernanceError(
                f"unknown run file {name!r}; choose one of {sorted(RUN_FILES)}"
            )
        self._require_run(run_id)
        path = self.recorder.run_dir(run_id) / filename
        if not path.is_file():
            raise NotFoundError(f"run {run_id!r} has no {filename}")
        return _read_capped(path)

    def get_artifact(self, run_id: str, name: str) -> str:
        """Read an `emit_artifact` output as text."""

        self._require_run(run_id)
        try:
            path = self.recorder.artifact_path(run_id, name)
        except ValueError as exc:
            raise GovernanceError(str(exc)) from exc
        if not path.is_file():
            raise NotFoundError(f"run {run_id!r} has no artifact {name!r}")
        return _read_capped(path)

    # --------------------------------------------------------------- helpers

    def _resolve_config(
        self, *, planner: str | None, provider: str | None, model: str | None
    ) -> RunConfig:
        s = self.settings
        chosen_planner = (planner or s.planner).strip().lower()
        chosen_provider = (provider or s.provider).strip().lower()
        if chosen_planner == "openai":
            chosen_planner, chosen_provider = "llm", "openai"
        if chosen_planner not in ("mock", "llm"):
            raise GovernanceError(
                f"planner must be 'mock' or 'llm', got {chosen_planner!r}"
            )
        if chosen_planner == "llm" and s.planner != "llm":
            raise GovernanceError(
                "the LLM planner is disabled on this server (DS_PLANNER=mock); "
                "use planner='mock' or ask the operator to enable it"
            )
        chosen_model = model or s.model
        if not s.is_model_allowed(chosen_model, provider=chosen_provider):
            raise GovernanceError(
                f"model {chosen_provider}:{chosen_model} is not in the allowlist "
                f"{list(s.model_allowlist)}"
            )
        config = RunConfig(
            planner=chosen_planner,  # type: ignore[arg-type]
            provider=chosen_provider,
            model=chosen_model,
            # Real-model runs must never fall back to echo runners silently.
            strict_runners=chosen_planner == "llm",
        )
        if config.planner == "llm":
            try:
                for name in config.providers_in_use():
                    self.model_providers.require_credentials(name)
            except KeyError as exc:
                raise GovernanceError(str(exc)) from exc
            except MissingModelProviderCredential as exc:
                raise GovernanceError(str(exc)) from exc
        return config

    def _persisted_config(self, run_id: str) -> RunConfig:
        stored = load_run_config(self.recorder.run_dir(run_id))
        if stored is None:
            return self._resolve_config(planner=None, provider=None, model=None)
        # A run recorded with the LLM planner is resumed/replayed with it, but
        # only if this server still permits that planner and model.
        return self._resolve_config(
            planner=stored["planner"],
            provider=stored["provider"],
            model=stored["model"],
        )

    def _supervisor(
        self, config: RunConfig, *, policy: ExecutionPolicy, callbacks: list[Any]
    ) -> Supervisor:
        return self._supervisor_factory(
            config,
            recorder=self.recorder,
            checkpointer=self._checkpointer,
            model_providers=self.model_providers,
            chat_callbacks=callbacks,
            policy=policy,
            max_plan_attempts=self.settings.max_plan_attempts,
        )

    def _require_run(self, run_id: str) -> None:
        self._validate_id(run_id)
        if not self.recorder.exists(run_id):
            raise NotFoundError(f"no recorded run {run_id!r}")

    @staticmethod
    def _validate_id(run_id: str) -> None:
        try:
            validate_run_id(run_id)
        except ValueError as exc:
            raise GovernanceError(str(exc)) from exc

    @contextmanager
    def _claim(self, run_id: str) -> Iterator[None]:
        """Reserve a *fresh* run id for the duration of a run or replay."""

        with self._reserve(run_id, must_be_new=True):
            yield

    @contextmanager
    def _exclusive(self, run_id: str) -> Iterator[None]:
        """Hold an existing run so two resumes can't race on it."""

        with self._reserve(run_id, must_be_new=False):
            yield

    @contextmanager
    def _reserve(self, run_id: str, *, must_be_new: bool) -> Iterator[None]:
        self._validate_id(run_id)
        with self._lock:
            if run_id in self._reserved:
                raise GovernanceError(f"run {run_id!r} is busy in another call")
            if must_be_new and self.recorder.exists(run_id):
                raise GovernanceError(f"run_id already exists: {run_id!r}")
            self._reserved.add(run_id)
        try:
            yield
        finally:
            with self._lock:
                self._reserved.discard(run_id)

    def _save_config(self, run_id: str, config: RunConfig) -> None:
        if self.recorder.exists(run_id):
            save_run_config(
                self.recorder.run_dir(run_id),
                planner=config.planner,
                provider=config.provider,
                model=config.model,
            )

    def _payload(
        self,
        result: SupervisorResult,
        *,
        usage: dict[str, Any],
        policy: ExecutionPolicy,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        artifacts: list[str] = []
        if result.result is not None:
            state = result.result.state
            values = dict(state.get("values", {}))
            artifacts = sorted(state.get("artifacts", {}).keys())
        spec = result.validated_spec
        plan = None
        if spec is not None:
            plan = {
                "graph_id": spec.graph_id,
                "nodes": [
                    {"id": node.id, "kind": node.kind.value} for node in spec.nodes
                ],
                "edges": [
                    edge.model_dump(mode="json", by_alias=True) for edge in spec.edges
                ],
            }
        limit = self.settings.max_value_chars
        return {
            "run_id": result.run_id,
            "status": result.status,
            "ok": result.status == "ok",
            "healthy": result.status in _HEALTHY_STATUSES,
            "response": _clip(result.response, limit),
            "values": _clip(values, limit),
            "errors": _clip(list(result.errors), limit),
            "plan": plan,
            "granted_budget": (
                spec.budget.model_dump(mode="json") if spec is not None else None
            ),
            "host_policy": _policy_dict(policy),
            "plan_attempts": result.plan_attempts,
            "usage": usage,
            "artifacts": artifacts,
            "recorded": self.recorder.exists(result.run_id),
        }


def _read_capped(path: Any) -> str:
    text: str = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > _MAX_FILE_CHARS:
        return f"{text[:_MAX_FILE_CHARS]}\n… [truncated {len(text) - _MAX_FILE_CHARS} chars]"
    return text
