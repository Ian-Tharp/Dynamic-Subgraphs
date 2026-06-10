"""Persist a single run as a directory of artifacts under `runs/<run_id>/`.

The recorder is the system's memory. Failed runs are recorded too — every
attempt produces a directory, so the future retrieval / eval / diff layer
has a complete record to work with.

Layout:

    runs/
      <run_id>/
        spec.json     -- the validated GraphSpec (alias-preserving)
        trace.jsonl   -- one TraceEvent per line
        output.json   -- final state minus events; includes ok/error
        graph.mmd     -- Mermaid diagram of the topology
        summary.md    -- human-readable summary
        prompt.md     -- optional, only written if a prompt was provided
        eval.json     -- optional, only written when eval_gate is configured
      <chain_id>/
        chain.json    -- emitted by record_chain() for iterative runs;
                         sits alongside <chain_id>_iter_N/ directories
        chain.md      -- human-readable chain summary
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from app.models import GraphSpec
from app.recording.mermaid import render_mermaid
from app.runtime import ExecutionResult

if TYPE_CHECKING:
    from app.supervisor.iteration import IterationDecision, IterativeSupervisorResult

_SAFE_RUN_ID_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)


@dataclass(frozen=True)
class RunRecord:
    """Receipt returned after a successful recording."""

    run_id: str
    directory: Path
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class ChainRecord:
    """Receipt returned after a successful chain recording."""

    chain_id: str
    directory: Path
    artifacts: dict[str, Path]


CHAIN_SCHEMA_VERSION = 1


class Recorder(Protocol):
    """Anything that can durably persist a run."""

    def record(
        self,
        *,
        spec: GraphSpec,
        result: ExecutionResult,
        prompt: str | None = None,
        overwrite: bool | None = None,
    ) -> RunRecord: ...

    def load_validated_spec(self, run_id: str) -> GraphSpec:
        """Load the spec previously persisted for `run_id`. Used for resume/replay."""
        ...

    def record_chain(
        self,
        result: IterativeSupervisorResult,
        *,
        original_prompt: str,
        overwrite: bool | None = None,
    ) -> ChainRecord:
        """Persist a meta-loop chain's metadata, decisions, and gap progression."""
        ...


@dataclass(frozen=True)
class ArtifactContext:
    """Everything an artifact producer may need to write its file."""

    run_id: str
    directory: Path
    spec: GraphSpec
    result: ExecutionResult
    prompt: str | None
    # Pre-serialized EvalResult mapping (the eval layer lives in the SDK facade;
    # app/ must not import it — spec layering rule). None = no score.
    eval_result: Mapping[str, Any] | None = None


class ArtifactProducer(Protocol):
    """One recordable run artifact.

    `key` is the stable short name used in `RunRecord.artifacts` (e.g.
    "spec"); `filename` is the on-disk name (e.g. "spec.json") and is also
    the identifier a selection filters on. `applies()` lets a producer
    opt out for a given run (e.g. the prompt file when no prompt was given).
    """

    key: str
    filename: str

    def applies(self, ctx: ArtifactContext) -> bool: ...

    def write(self, ctx: ArtifactContext) -> Path: ...


class _SpecProducer:
    key = "spec"
    filename = "spec.json"

    def applies(self, ctx: ArtifactContext) -> bool:
        return True

    def write(self, ctx: ArtifactContext) -> Path:
        path = ctx.directory / self.filename
        path.write_text(
            json.dumps(ctx.spec.model_dump(mode="json", by_alias=True), indent=2),
            encoding="utf-8",
        )
        return path


class _TraceProducer:
    key = "trace"
    filename = "trace.jsonl"

    def applies(self, ctx: ArtifactContext) -> bool:
        return True

    def write(self, ctx: ArtifactContext) -> Path:
        path = ctx.directory / self.filename
        with path.open("w", encoding="utf-8") as fh:
            for event in ctx.result.trace.events:
                fh.write(json.dumps(event.model_dump(mode="json")) + "\n")
        return path


class _OutputProducer:
    key = "output"
    filename = "output.json"

    def applies(self, ctx: ArtifactContext) -> bool:
        return True

    def write(self, ctx: ArtifactContext) -> Path:
        path = ctx.directory / self.filename
        path.write_text(
            json.dumps(_extract_output(ctx.result), indent=2, default=str),
            encoding="utf-8",
        )
        return path


class _MermaidProducer:
    key = "mermaid"
    filename = "graph.mmd"

    def applies(self, ctx: ArtifactContext) -> bool:
        return True

    def write(self, ctx: ArtifactContext) -> Path:
        path = ctx.directory / self.filename
        path.write_text(render_mermaid(ctx.spec), encoding="utf-8")
        return path


class _PromptProducer:
    key = "prompt"
    filename = "prompt.md"

    def applies(self, ctx: ArtifactContext) -> bool:
        return ctx.prompt is not None

    def write(self, ctx: ArtifactContext) -> Path:
        path = ctx.directory / self.filename
        path.write_text(ctx.prompt or "", encoding="utf-8")
        return path


class _SummaryProducer:
    key = "summary"
    filename = "summary.md"

    def applies(self, ctx: ArtifactContext) -> bool:
        return True

    def write(self, ctx: ArtifactContext) -> Path:
        path = ctx.directory / self.filename
        path.write_text(
            _render_summary(ctx.run_id, ctx.spec, ctx.result, ctx.prompt),
            encoding="utf-8",
        )
        return path


class _EvalProducer:
    key = "eval"
    filename = "eval.json"

    def applies(self, ctx: ArtifactContext) -> bool:
        return ctx.eval_result is not None

    def write(self, ctx: ArtifactContext) -> Path:
        assert ctx.eval_result is not None  # guaranteed by applies()
        return _write_eval_json(ctx.directory, ctx.eval_result)


def _write_eval_json(directory: Path, eval_result: Mapping[str, Any]) -> Path:
    """Write eval.json to a directory. Shared by _EvalProducer and record_eval()."""
    path = directory / _EvalProducer.filename
    path.write_text(
        json.dumps(dict(eval_result), indent=2, default=str),
        encoding="utf-8",
    )
    return path


# Order is preserved in RunRecord.artifacts and matches the historical layout.
DEFAULT_PRODUCERS: tuple[ArtifactProducer, ...] = (
    _SpecProducer(),
    _TraceProducer(),
    _OutputProducer(),
    _MermaidProducer(),
    _PromptProducer(),
    _SummaryProducer(),
    _EvalProducer(),
)


class FileRecorder:
    """Filesystem-backed recorder. Writes one directory per run.

    Args:
        root_dir: parent directory for all runs. Created on first write.
        overwrite: if False (default), `record()` raises FileExistsError when
            the per-run directory already exists. If True, files are
            overwritten in place; other files in the directory are left alone.
        selection: a set of artifact *filenames* to write (e.g.
            ``{"graph.mmd", "trace.jsonl"}``). ``None`` (default) writes the
            full set. Unselected producers never run, so e.g. the Mermaid
            diagram and summary aren't rendered when not requested.
        producers: the artifact producers to iterate. Defaults to
            `DEFAULT_PRODUCERS`; pass a superset to register custom artifacts.

    Note: excluding ``spec.json`` is allowed, but `resume`/`replay` need it —
    they fail loudly (`load_validated_spec`) if it wasn't recorded.
    """

    def __init__(
        self,
        root_dir: Path | str = "runs",
        *,
        overwrite: bool = False,
        selection: frozenset[str] | None = None,
        producers: tuple[ArtifactProducer, ...] = DEFAULT_PRODUCERS,
    ) -> None:
        self._root = Path(root_dir)
        self._overwrite = overwrite
        self._selection = selection
        self._producers = producers

    @property
    def root_dir(self) -> Path:
        return self._root

    def _selects(self, producer: ArtifactProducer) -> bool:
        return self._selection is None or producer.filename in self._selection

    def record(
        self,
        *,
        spec: GraphSpec,
        result: ExecutionResult,
        prompt: str | None = None,
        overwrite: bool | None = None,
    ) -> RunRecord:
        run_id = result.trace.run_id
        validate_run_id(run_id)

        effective_overwrite = self._overwrite if overwrite is None else overwrite

        directory = self._root / run_id
        if directory.exists() and not effective_overwrite:
            raise FileExistsError(
                f"Run directory already exists (set overwrite=True to replace): {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)

        ctx = ArtifactContext(
            run_id=run_id,
            directory=directory,
            spec=spec,
            result=result,
            prompt=prompt,
        )

        artifacts: dict[str, Path] = {}
        for producer in self._producers:
            if not self._selects(producer):
                continue
            if not producer.applies(ctx):
                continue
            artifacts[producer.key] = producer.write(ctx)

        return RunRecord(run_id=run_id, directory=directory, artifacts=artifacts)

    def load_validated_spec(self, run_id: str) -> GraphSpec:
        """Read a previously-persisted `spec.json` for `run_id` and parse it."""
        validate_run_id(run_id)
        spec_path = self._root / run_id / "spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(
                f"No recorded spec for run_id {run_id!r} at {spec_path}. "
                "resume/replay need spec.json — re-run including the SPEC "
                "artifact (e.g. record=Recording.replayable() or record=True)."
            )
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        return GraphSpec.model_validate(raw)

    def record_chain(
        self,
        result: IterativeSupervisorResult,
        *,
        original_prompt: str,
        overwrite: bool | None = None,
    ) -> ChainRecord:
        """Persist a meta-loop chain alongside its per-iteration run directories.

        Writes `runs/<chain_id>/chain.json` (structured) + `chain.md`
        (human-readable summary). The per-iteration directories
        (`<chain_id>_iter_N/`) are written separately by `record()` on each
        iteration's `SupervisorResult` — this method records only the
        chain-level metadata that connects them.
        """

        chain_id = result.chain_id
        validate_run_id(chain_id)

        effective_overwrite = self._overwrite if overwrite is None else overwrite
        directory = self._root / chain_id
        if directory.exists() and not effective_overwrite:
            raise FileExistsError(
                f"Chain directory already exists (set overwrite=True): {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, Path] = {}

        chain_payload = _serialize_chain(result, original_prompt=original_prompt)
        chain_path = directory / "chain.json"
        chain_path.write_text(
            json.dumps(chain_payload, indent=2, default=str),
            encoding="utf-8",
        )
        artifacts["chain"] = chain_path

        summary_path = directory / "chain.md"
        summary_path.write_text(
            _render_chain_summary(result, original_prompt=original_prompt),
            encoding="utf-8",
        )
        artifacts["summary"] = summary_path

        return ChainRecord(chain_id=chain_id, directory=directory, artifacts=artifacts)

    def load_chain(self, chain_id: str) -> dict[str, Any]:
        """Read a previously-persisted `chain.json` for `chain_id`."""
        validate_run_id(chain_id)
        chain_path = self._root / chain_id / "chain.json"
        if not chain_path.exists():
            raise FileNotFoundError(
                f"No recorded chain for chain_id {chain_id!r} at {chain_path}"
            )
        return json.loads(chain_path.read_text(encoding="utf-8"))

    def exists(self, run_id: str) -> bool:
        validate_run_id(run_id)
        return (self._root / run_id).is_dir()

    def run_dir(self, run_id: str) -> Path:
        validate_run_id(run_id)
        return self._contained(self._root / run_id, run_id)

    def load_output(self, run_id: str) -> dict[str, Any]:
        validate_run_id(run_id)
        output_path = self._root / run_id / "output.json"
        if not output_path.exists():
            raise FileNotFoundError(
                f"No output.json for run_id {run_id!r} at {output_path}"
            )
        return json.loads(output_path.read_text(encoding="utf-8"))

    def record_eval(self, run_id: str, eval_result: Mapping[str, Any]) -> Path | None:
        """Write eval.json into an existing run dir (scoring happens after record()).

        Returns None (writes nothing) when the EVAL artifact isn't selected, so
        the engine can call this unconditionally for FileRecorder instances.
        """
        validate_run_id(run_id)
        if (
            self._selection is not None
            and _EvalProducer.filename not in self._selection
        ):
            return None
        directory = self._contained(self._root / run_id, run_id)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"No recorded run directory for run_id {run_id!r} at {directory}. "
                "Call record() before record_eval()."
            )
        return _write_eval_json(directory, eval_result)

    def load_eval(self, run_id: str) -> dict[str, Any] | None:
        """Read eval.json for run_id; None when the run was never scored.

        Unlike load_output/load_chain, absence is not an error — many runs are
        recorded without being scored. Raises on corrupt/unreadable JSON.
        """
        validate_run_id(run_id)
        path = self._root / run_id / "eval.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def artifact_path(self, run_id: str, name: str) -> Path:
        validate_run_id(run_id)
        name_ok = bool(name) and all(ch in _SAFE_RUN_ID_CHARS for ch in name)
        if not name_ok or set(name) == {"."}:
            raise ValueError(f"artifact name contains unsafe characters: {name!r}")
        return self._contained(self._root / run_id / "artifacts" / name, run_id)

    def _contained(self, path: Path, run_id: str) -> Path:
        """Defense in depth: confirm a built path stays under the runs root.

        `validate_run_id` already rejects traversal tokens, so this never fires
        in practice — it's a backstop so any future charset change can't silently
        reintroduce an escape.
        """
        if not path.resolve().is_relative_to(self._root.resolve()):
            raise ValueError(f"run_id {run_id!r} resolves outside the runs root")
        return path

    def list_runs(self) -> list[dict[str, Any]]:
        """Summaries of every recorded run directory (id, status, node count).

        Skips chain directories (those with chain.json but no spec.json).
        """
        if not self._root.exists():
            return []
        summaries: list[dict[str, Any]] = []
        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            spec_path = child / "spec.json"
            if not spec_path.exists():
                continue
            status = "unknown"
            output_path = child / "output.json"
            if output_path.exists():
                try:
                    out = json.loads(output_path.read_text(encoding="utf-8"))
                    # Trust the persisted `ok` flag (the authoritative field) rather
                    # than re-inferring from error/errors — the old triple-fallback
                    # could report "ok" for a record with ok=False and no errors list.
                    status = "ok" if out.get("ok") else "failed"
                except (json.JSONDecodeError, OSError):
                    status = "unknown"
            node_count = 0
            try:
                spec_raw = json.loads(spec_path.read_text(encoding="utf-8"))
                node_count = len(spec_raw.get("nodes", []))
            except (json.JSONDecodeError, OSError):
                node_count = 0
            summary: dict[str, Any] = {
                "run_id": child.name,
                "status": status,
                "nodes": node_count,
            }
            eval_path = child / "eval.json"
            if eval_path.exists():
                try:
                    ev = json.loads(eval_path.read_text(encoding="utf-8"))
                    summary["quality"] = ev.get("quality")
                    summary["value_per_ktok"] = ev.get("value_per_ktok")
                    summary["origin"] = (ev.get("fingerprint") or {}).get("origin")
                    summary["deterministic"] = ev.get("deterministic")
                    summary["quality_floor_met"] = ev.get("quality_floor_met")
                except (json.JSONDecodeError, OSError):
                    pass
            summaries.append(summary)
        return summaries


class NullRecorder:
    """Recorder that persists nothing — for SDK/library use.

    Satisfies the `Recorder` protocol but writes no files, so embedding the
    engine in someone else's app doesn't flood their working directory with
    `runs/` output. `record()` returns a receipt with an empty artifact map.
    Resume/replay need real persistence, so `load_validated_spec` raises with
    guidance to enable recording.
    """

    def __init__(self, root_dir: Path | str = "runs") -> None:
        self._root = Path(root_dir)

    @property
    def root_dir(self) -> Path:
        return self._root

    def record(
        self,
        *,
        spec: GraphSpec,
        result: ExecutionResult,
        prompt: str | None = None,
        overwrite: bool | None = None,
    ) -> RunRecord:
        del spec, prompt, overwrite
        return RunRecord(
            run_id=result.trace.run_id,
            directory=self._root / result.trace.run_id,
            artifacts={},
        )

    def load_validated_spec(self, run_id: str) -> GraphSpec:
        raise FileNotFoundError(
            f"NullRecorder persists nothing, so spec for {run_id!r} is "
            "unavailable. Enable recording (record=True) to use resume/replay."
        )

    def record_chain(
        self,
        result: IterativeSupervisorResult,
        *,
        original_prompt: str,
        overwrite: bool | None = None,
    ) -> ChainRecord:
        del original_prompt, overwrite
        return ChainRecord(
            chain_id=result.chain_id,
            directory=self._root / result.chain_id,
            artifacts={},
        )


def validate_run_id(run_id: str) -> None:
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if not all(ch in _SAFE_RUN_ID_CHARS for ch in run_id):
        raise ValueError(
            f"run_id contains unsafe characters (allowed: letters, digits, '-', '_', '.'): {run_id!r}"
        )
    # `.` is allowed *inside* an id (e.g. a model tag like "gpt-5.4"), but an id
    # that is nothing but dots ('.', '..', ...) is a path-traversal token: with no
    # separator in the charset, `run_dir("..")` would otherwise select the runs
    # root's parent. Refuse it so a run_id can never escape the root.
    if set(run_id) == {"."}:
        raise ValueError(f"run_id may not be a bare dot path segment: {run_id!r}")


def _extract_output(result: ExecutionResult) -> dict[str, object]:
    """Pull just the parts of state that belong in output.json.

    Events live in trace.jsonl; we don't duplicate them here.
    """

    state = result.state
    return {
        "ok": result.ok,
        "error": result.error,
        "values": dict(state.get("values", {})),
        "artifacts": dict(state.get("artifacts", {})),
        "errors": list(state.get("errors", [])),
        "metadata": dict(state.get("metadata", {})),
    }


def _render_summary(
    run_id: str,
    spec: GraphSpec,
    result: ExecutionResult,
    prompt: str | None,
) -> str:
    if getattr(result, "paused", False):
        status = "paused"
    elif result.ok:
        status = "ok"
    else:
        status = "failed"
    lines: list[str] = [
        f"# Run `{run_id}`",
        "",
        f"- **Graph:** `{spec.graph_id}`",
        f"- **Goal:** {spec.goal}",
        f"- **Status:** {status}",
        f"- **Nodes:** {len(spec.nodes)}",
        f"- **Edges:** {len(spec.edges)}",
        f"- **Trace events:** {len(result.trace.events)}",
    ]
    if result.error:
        lines.append(f"- **Error:** {result.error}")
    if prompt is not None:
        lines.extend(["", "## Prompt", "", prompt])

    values = result.state.get("values", {})
    if values:
        lines.extend(["", "## Final values", ""])
        for key in sorted(values.keys()):
            lines.append(f"- `{key}`: `{values[key]!r}`")

    errors = result.state.get("errors") or []
    if errors:
        lines.extend(["", "## Errors", ""])
        for entry in errors:
            node = entry.get("node_id", "?")
            etype = entry.get("type", "Error")
            msg = entry.get("message", "")
            lines.append(f"- `{node}` ({etype}): {msg}")

    lines.append("")
    return "\n".join(lines)


def _decision_to_dict(decision: IterationDecision) -> dict[str, Any]:
    """JSON shape for one iteration decision — the chain's persisted memory.

    Used for both per-step and final decisions so the two can't drift.
    """
    return {
        "action": decision.action,
        "reason": decision.reason,
        "success_criteria_met": decision.success_criteria_met,
        "gaps": list(decision.gaps),
        "next_prompt": decision.next_prompt,
        "question_to_user": decision.question_to_user,
    }


def _serialize_chain(
    result: IterativeSupervisorResult,
    *,
    original_prompt: str,
) -> dict[str, Any]:
    """Render an IterativeSupervisorResult as a JSON-serializable payload."""

    steps_payload: list[dict[str, Any]] = []
    for step in result.steps:
        steps_payload.append(
            {
                "iteration": step.iteration,
                "run_id": step.run_id,
                "prompt": step.prompt,
                "result": {
                    "status": step.result.status,
                    "response": step.result.response,
                    "record_dir": (
                        str(step.result.record.directory)
                        if step.result.record is not None
                        else None
                    ),
                },
                "decision": _decision_to_dict(step.decision),
            }
        )

    final_decision_payload: dict[str, Any] | None = None
    if result.final_decision is not None:
        final_decision_payload = _decision_to_dict(result.final_decision)

    return {
        "schema_version": CHAIN_SCHEMA_VERSION,
        "chain_id": result.chain_id,
        "original_prompt": original_prompt,
        "status": result.status,
        "response": result.response,
        "recorded_at": datetime.now(UTC).isoformat(),
        "step_count": len(result.steps),
        "steps": steps_payload,
        "final_decision": final_decision_payload,
        "final_run_id": (
            result.final_result.run_id if result.final_result is not None else None
        ),
    }


def _render_chain_summary(
    result: IterativeSupervisorResult,
    *,
    original_prompt: str,
) -> str:
    lines: list[str] = [
        f"# Chain `{result.chain_id}`",
        "",
        f"- **Status:** {result.status}",
        f"- **Iterations:** {len(result.steps)}",
        f"- **Response:** {result.response}",
        "",
        "## Original prompt",
        "",
        original_prompt,
        "",
        "## Iterations",
        "",
    ]
    for step in result.steps:
        d = step.decision
        lines.append(f"### Iteration {step.iteration} — `{step.run_id}`")
        lines.append("")
        lines.append(f"- Run status: `{step.result.status}`")
        lines.append(f"- Decision: `{d.action}` (success_met={d.success_criteria_met})")
        lines.append(f"- Reason: {d.reason}")
        if d.gaps:
            lines.append("- Gaps:")
            for gap in d.gaps:
                lines.append(f"  - {gap}")
        lines.append("")
    return "\n".join(lines)
