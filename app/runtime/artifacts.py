"""Side-effect boundary for `emit_artifact` nodes.

`emit_artifact` is where a synthesized workflow's output reaches the
outside world — a file on disk, a message to a channel, a structured
report. The runner itself is a thin shim; the actual persistence
behavior is encapsulated in an injected `ArtifactSink`.

The default `run_emit_artifact` (in `runtime/runners.py`) is a no-I/O
echo for tests and no-LLM demos. Production callers wire a real sink:

    from app.runtime import FileArtifactSink, make_emit_artifact_runner
    runner = make_emit_artifact_runner(FileArtifactSink(root_dir="runs"))
    executor = LangGraphExecutor(runners={NodeKind.EMIT_ARTIFACT: runner})

The runner reads `state.values[content_key]`, hands it to the sink,
and returns the sink's reference dict via `outputs[0]`. The sink is
free to attach any additional fields (e.g., `path` for filesystem,
`url` for webhooks).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol

from app.models import DynamicRunState


class ArtifactSink(Protocol):
    """Pluggable persistence for emitted artifacts.

    `emit(...)` returns a reference dict that becomes the node's output.
    Common fields: `name`, `type`, plus a sink-specific locator like
    `path` (for files) or `url` (for webhooks).
    """

    def emit(
        self,
        *,
        name: str,
        artifact_type: str,
        content: Any,
        run_id: str | None = None,
    ) -> dict[str, Any]: ...


class CollectingArtifactSink:
    """Sink that keeps emitted artifacts in memory.

    Useful for tests and for no-side-effect demos. Each call appends an
    entry to `emitted` and returns a reference (without `path`).
    """

    def __init__(self) -> None:
        self.emitted: list[dict[str, Any]] = []

    def emit(
        self,
        *,
        name: str,
        artifact_type: str,
        content: Any,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        reference: dict[str, Any] = {"name": name, "type": artifact_type}
        if run_id is not None:
            reference["run_id"] = run_id
        # Store the full content alongside the reference so tests can assert
        # on what was emitted without parsing files.
        self.emitted.append({**reference, "content": content})
        return reference


class FileArtifactSink:
    """Sink that persists artifacts to disk under `<root>/<run_id>/artifacts/`.

    Strings are written as-is. Dicts and lists are JSON-encoded with
    `default=str`. The file extension is derived from `artifact_type`
    (`file` -> `.txt`, `report` -> `.md`, `message` -> `.txt`) and forced
    to `.json` when the content is structured.

    Names are sanitized to filesystem-safe characters; duplicates within
    the same run overwrite. If `run_id` is missing, falls back to a
    `_unscoped` subdirectory so the sink stays usable in ad-hoc contexts.
    """

    _EXTENSIONS: ClassVar[dict[str, str]] = {
        "file": ".txt",
        "report": ".md",
        "message": ".txt",
    }

    def __init__(self, root_dir: Path | str = "runs") -> None:
        self._root = Path(root_dir)

    @property
    def root_dir(self) -> Path:
        return self._root

    def emit(
        self,
        *,
        name: str,
        artifact_type: str,
        content: Any,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        scope = run_id or "_unscoped"
        target_dir = self._root / _check_scope(scope) / "artifacts"
        # Backstop: the resolved destination must stay under the root.
        if not target_dir.resolve().is_relative_to(self._root.resolve()):
            raise ValueError(f"run_id {run_id!r} resolves outside the artifacts root")
        target_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _sanitize_filename(name)
        ext = self._EXTENSIONS.get(artifact_type, ".txt")
        path = target_dir / f"{safe_name}{ext}"

        if isinstance(content, (dict, list)):
            path = path.with_suffix(".json")
            path.write_text(
                json.dumps(content, indent=2, default=str),
                encoding="utf-8",
            )
        elif isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_text(str(content), encoding="utf-8")

        reference: dict[str, Any] = {
            "name": name,
            "type": artifact_type,
            "path": str(path),
        }
        if run_id is not None:
            reference["run_id"] = run_id
        return reference


def make_emit_artifact_runner(
    sink: ArtifactSink,
) -> Callable[[DynamicRunState, Mapping[str, Any]], dict[str, Any]]:
    """Build the `NodeRunner` for `EMIT_ARTIFACT` given an injected sink."""

    def _runner(
        state: DynamicRunState,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        name = str(params["name"])
        artifact_type = str(params["artifact_type"])
        content_key = str(params["content_key"])

        values = state.get("values", {}) or {}
        if content_key not in values:
            raise RuntimeError(
                f"emit_artifact: content_key {content_key!r} is not present in "
                f"state.values. Either declare it in `inputs` and ensure an "
                f"upstream node produces it, or pick an existing key."
            )

        content = values[content_key]
        run_id = (state.get("metadata", {}) or {}).get("run_id")
        reference = sink.emit(
            name=name,
            artifact_type=artifact_type,
            content=content,
            run_id=run_id,
        )
        return {"result": reference}

    return _runner


_SAFE_SCOPE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)
# Windows device names: `CON.txt` opens the device, not a file.
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{d}{i}" for d in ("COM", "LPT") for i in range(1, 10)}
)


def _check_scope(run_id: str) -> str:
    """Reject run ids that could leave the root (same charset as the recorder's
    `validate_run_id`; duplicated because recording imports runtime, not vice versa).
    """
    if not all(c in _SAFE_SCOPE_CHARS for c in run_id) or set(run_id) == {"."}:
        raise ValueError(f"emit_artifact: unsafe run_id {run_id!r}")
    return run_id


def _sanitize_filename(name: str) -> str:
    """Coerce an artifact name to a filesystem-safe basename."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    safe = safe.lstrip(".")[:100] or "artifact"
    if safe.split(".")[0].upper() in _WINDOWS_RESERVED:
        safe = f"_{safe}"
    return safe
