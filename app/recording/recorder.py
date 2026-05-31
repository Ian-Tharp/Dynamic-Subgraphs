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
      <chain_id>/
        chain.json    -- emitted by record_chain() for iterative runs;
                         sits alongside <chain_id>_iter_N/ directories
        chain.md      -- human-readable chain summary
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from app.models import GraphSpec
from app.recording.mermaid import render_mermaid
from app.runtime import ExecutionResult

if TYPE_CHECKING:
    from app.supervisor.iteration import IterativeSupervisorResult

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
        result: "IterativeSupervisorResult",
        *,
        original_prompt: str,
        overwrite: bool | None = None,
    ) -> ChainRecord:
        """Persist a meta-loop chain's metadata, decisions, and gap progression."""
        ...


class FileRecorder:
    """Filesystem-backed recorder. Writes one directory per run.

    Args:
        root_dir: parent directory for all runs. Created on first write.
        overwrite: if False (default), `record()` raises FileExistsError when
            the per-run directory already exists. If True, files are
            overwritten in place; other files in the directory are left alone.
    """

    def __init__(
        self,
        root_dir: Path | str = "runs",
        *,
        overwrite: bool = False,
    ) -> None:
        self._root = Path(root_dir)
        self._overwrite = overwrite

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
        run_id = result.trace.run_id
        _validate_run_id(run_id)

        effective_overwrite = self._overwrite if overwrite is None else overwrite

        directory = self._root / run_id
        if directory.exists() and not effective_overwrite:
            raise FileExistsError(
                f"Run directory already exists (set overwrite=True to replace): {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, Path] = {}

        spec_path = directory / "spec.json"
        spec_path.write_text(
            json.dumps(spec.model_dump(mode="json", by_alias=True), indent=2),
            encoding="utf-8",
        )
        artifacts["spec"] = spec_path

        trace_path = directory / "trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as fh:
            for event in result.trace.events:
                fh.write(json.dumps(event.model_dump(mode="json")) + "\n")
        artifacts["trace"] = trace_path

        output_path = directory / "output.json"
        output_path.write_text(
            json.dumps(_extract_output(result), indent=2, default=str),
            encoding="utf-8",
        )
        artifacts["output"] = output_path

        mermaid_path = directory / "graph.mmd"
        mermaid_path.write_text(render_mermaid(spec), encoding="utf-8")
        artifacts["mermaid"] = mermaid_path

        if prompt is not None:
            prompt_path = directory / "prompt.md"
            prompt_path.write_text(prompt, encoding="utf-8")
            artifacts["prompt"] = prompt_path

        summary_path = directory / "summary.md"
        summary_path.write_text(
            _render_summary(run_id, spec, result, prompt),
            encoding="utf-8",
        )
        artifacts["summary"] = summary_path

        return RunRecord(run_id=run_id, directory=directory, artifacts=artifacts)

    def load_validated_spec(self, run_id: str) -> GraphSpec:
        """Read a previously-persisted `spec.json` for `run_id` and parse it."""
        _validate_run_id(run_id)
        spec_path = self._root / run_id / "spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(
                f"No recorded spec for run_id {run_id!r} at {spec_path}"
            )
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        return GraphSpec.model_validate(raw)

    def record_chain(
        self,
        result: "IterativeSupervisorResult",
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
        _validate_run_id(chain_id)

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

        return ChainRecord(
            chain_id=chain_id, directory=directory, artifacts=artifacts
        )

    def load_chain(self, chain_id: str) -> dict[str, Any]:
        """Read a previously-persisted `chain.json` for `chain_id`."""
        _validate_run_id(chain_id)
        chain_path = self._root / chain_id / "chain.json"
        if not chain_path.exists():
            raise FileNotFoundError(
                f"No recorded chain for chain_id {chain_id!r} at {chain_path}"
            )
        return json.loads(chain_path.read_text(encoding="utf-8"))


def _validate_run_id(run_id: str) -> None:
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if not all(ch in _SAFE_RUN_ID_CHARS for ch in run_id):
        raise ValueError(
            f"run_id contains unsafe characters (allowed: letters, digits, '-', '_', '.'): {run_id!r}"
        )


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


def _serialize_chain(
    result: "IterativeSupervisorResult",
    *,
    original_prompt: str,
) -> dict[str, Any]:
    """Render an IterativeSupervisorResult as a JSON-serializable payload."""

    steps_payload: list[dict[str, Any]] = []
    for step in result.steps:
        decision = step.decision
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
                "decision": {
                    "action": decision.action,
                    "reason": decision.reason,
                    "success_criteria_met": decision.success_criteria_met,
                    "gaps": list(decision.gaps),
                    "next_prompt": decision.next_prompt,
                    "question_to_user": decision.question_to_user,
                },
            }
        )

    final_decision_payload: dict[str, Any] | None = None
    if result.final_decision is not None:
        fd = result.final_decision
        final_decision_payload = {
            "action": fd.action,
            "reason": fd.reason,
            "success_criteria_met": fd.success_criteria_met,
            "gaps": list(fd.gaps),
            "next_prompt": fd.next_prompt,
            "question_to_user": fd.question_to_user,
        }

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
    result: "IterativeSupervisorResult",
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
