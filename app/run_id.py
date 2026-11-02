"""Shared run-id validation.

A run id becomes a directory name, ledger key and checkpointer thread_id, so the
recorder and the artifact sink must agree on what is safe. Lives at the package
root so both can import it without an import cycle (recording imports runtime).
"""

from __future__ import annotations

SAFE_RUN_ID_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)

# Windows device names: a directory or file called `CON` / `CON.txt` opens the device.
WINDOWS_RESERVED: frozenset[str] = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{d}{i}" for d in ("COM", "LPT") for i in range(1, 10)}
)


def validate_run_id(run_id: str) -> None:
    if not run_id:
        raise ValueError("run_id must be non-empty")
    if not all(ch in SAFE_RUN_ID_CHARS for ch in run_id):
        raise ValueError(
            f"run_id contains unsafe characters (allowed: letters, digits, '-', '_', '.'): {run_id!r}"
        )
    # `.` is allowed *inside* an id (e.g. a model tag like "gpt-5.4"), but an id
    # that is nothing but dots ('.', '..', ...) is a path-traversal token: with no
    # separator in the charset, `run_dir("..")` would otherwise select the runs
    # root's parent. Refuse it so a run_id can never escape the root.
    if set(run_id) == {"."}:
        raise ValueError(f"run_id may not be a bare dot path segment: {run_id!r}")
    # Windows strips trailing dots (`run.` aliases `run`) and reserves device names.
    if run_id.endswith("."):
        raise ValueError(f"run_id may not end with a dot: {run_id!r}")
    if run_id.split(".")[0].upper() in WINDOWS_RESERVED:
        raise ValueError(f"run_id is a reserved Windows device name: {run_id!r}")
