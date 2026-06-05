"""Selectable recording artifacts for the SDK.

`Artifact` is a `StrEnum` whose **values are the on-disk filenames**, so the
member is self-describing (``Artifact.MERMAID == "graph.mmd"``) and an agent
reading `result.artifacts` sees the same strings it selects with. `Recording`
is a small, composable policy that says *which* artifacts a run writes.

    from dynamic_subgraphs import Artifact, Recording

    Recording.visual_only()             # just the Mermaid diagram
    Recording.all() - {Artifact.SPEC}   # everything except spec.json
    {Artifact.MERMAID, Artifact.TRACE}  # a raw set works too (engine coerces)

Selecting *nothing* writes no files (the engine uses an in-memory recorder).
Note: `resume`/`replay` need `Artifact.SPEC`; if you exclude it, those calls
fail loudly when they try to reload the spec.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum


class Artifact(StrEnum):
    """A selectable run artifact. Values are the on-disk filenames.

    Tips:
        - `MERMAID` (`graph.mmd`) is the visual of the generated graph.
        - `SPEC` (`spec.json`) is required by `resume`/`replay`.
        - `EMITTED` toggles whether `emit_artifact` node outputs are written
          to disk (under ``runs/<run_id>/artifacts/``) vs kept in memory.
          It has no single filename, so its value is the sentinel "emitted".
    """

    SPEC = "spec.json"
    TRACE = "trace.jsonl"
    OUTPUT = "output.json"
    MERMAID = "graph.mmd"
    SUMMARY = "summary.md"
    PROMPT = "prompt.md"
    EMITTED = "emitted"


# Everything the FileRecorder writes (i.e. all artifacts except the EMITTED
# sink toggle, which controls a different write path).
RECORDER_ARTIFACTS: frozenset[Artifact] = frozenset(Artifact) - {Artifact.EMITTED}


def _coerce_member(value: Artifact | str) -> Artifact:
    if isinstance(value, Artifact):
        return value
    if isinstance(value, str):
        try:
            return Artifact(value)  # by filename value, e.g. "graph.mmd"
        except ValueError:
            pass
        try:
            return Artifact[value.upper()]  # by member name, e.g. "mermaid"
        except KeyError:
            pass
        raise ValueError(
            f"Unknown artifact {value!r}. Valid artifacts (by filename): "
            f"{', '.join(a.value for a in Artifact)}"
        )
    raise TypeError(
        f"Artifact selection must be an Artifact or str, got {type(value).__name__}"
    )


def _as_set(value: object) -> frozenset[Artifact]:
    if isinstance(value, Recording):
        return value.kinds
    if isinstance(value, (Artifact, str)):
        return frozenset({_coerce_member(value)})
    if isinstance(value, Iterable):
        return frozenset(_coerce_member(v) for v in value)
    raise TypeError(f"Cannot combine Recording with {type(value).__name__}")


@dataclass(frozen=True)
class Recording:
    """Which artifacts a run should write.

    Compose with set operators (`|`, `-`) and the `with_`/`without` helpers,
    or start from a preset. The engine accepts a `Recording`, a single
    `Artifact`, a set/list of `Artifact`, or a plain `bool` anywhere a
    recording policy is expected (see `Recording.coerce`).
    """

    kinds: frozenset[Artifact] = frozenset()

    # ----- presets -----
    @classmethod
    def none(cls) -> Recording:
        """Write nothing (default). Results still return in memory."""
        return cls(frozenset())

    @classmethod
    def all(cls) -> Recording:
        """Write every artifact, including emitted node outputs (== record=True)."""
        return cls(frozenset(Artifact))

    @classmethod
    def debug(cls) -> Recording:
        """Full capture for debugging — alias of `all()`."""
        return cls.all()

    @classmethod
    def visual_only(cls) -> Recording:
        """Just `graph.mmd` — the visual of the generated graph."""
        return cls(frozenset({Artifact.MERMAID}))

    @classmethod
    def replayable(cls) -> Recording:
        """The minimum needed to `resume`/`replay`: spec + output."""
        return cls(frozenset({Artifact.SPEC, Artifact.OUTPUT}))

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        """The preset constructor names — the single source for `capabilities()`.

        Co-located with the presets above so the agent-facing capability list can't
        drift from the methods that implement it (a guard test asserts the match).
        """
        return ("none", "all", "debug", "visual_only", "replayable")

    @classmethod
    def coerce(cls, value: object) -> Recording:
        """Normalize any accepted recording input to a `Recording`.

        `None`/`False` -> none; `True` -> all; an `Artifact`, a set/list of
        them, or a `Recording` -> the matching policy. Unknown artifact
        strings raise `ValueError` listing the valid filenames.
        """
        if value is None or value is False:
            return cls.none()
        if value is True:
            return cls.all()
        if isinstance(value, Recording):
            return value
        return cls(_as_set(value))

    # ----- composition -----
    def __or__(self, other: object) -> Recording:
        return Recording(self.kinds | _as_set(other))

    def __sub__(self, other: object) -> Recording:
        return Recording(self.kinds - _as_set(other))

    def with_(self, *flags: Artifact | str) -> Recording:
        return Recording(self.kinds | _as_set(flags))

    def without(self, *flags: Artifact | str) -> Recording:
        return Recording(self.kinds - _as_set(flags))

    def __contains__(self, item: object) -> bool:
        return item in self.kinds

    def __iter__(self) -> Iterator[Artifact]:
        return iter(self.kinds)

    def __bool__(self) -> bool:
        return bool(self.kinds)

    # ----- introspection (used by the engine) -----
    def records_anything(self) -> bool:
        """True if any FileRecorder artifact is selected."""
        return bool(self.kinds & RECORDER_ARTIFACTS)

    def emits(self) -> bool:
        """True if emit_artifact outputs should be written to disk."""
        return Artifact.EMITTED in self.kinds

    def recorder_filenames(self) -> frozenset[str]:
        """The on-disk filenames the FileRecorder should write."""
        return frozenset(a.value for a in self.kinds & RECORDER_ARTIFACTS)
