"""Persist the planner/model a run was created with, so resume/replay reuse it.

Without this, resume/replay fall back to the server's default planner — a run
created with planner=openai would be *finished* by the mock model. The config
sidecar (`api_config.json`) lives in the run directory next to the recorder's
own artifacts and is read back when a run is resumed or replayed.
"""

from __future__ import annotations

import json
from pathlib import Path

_CONFIG_FILENAME = "api_config.json"


def save_run_config(run_dir: Path, *, planner: str, model: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / _CONFIG_FILENAME
    path.write_text(
        json.dumps({"planner": planner, "model": model}, indent=2),
        encoding="utf-8",
    )


def load_run_config(run_dir: Path) -> dict[str, str] | None:
    """Return {"planner", "model"} for the run, or None if absent/unreadable."""
    path = run_dir / _CONFIG_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    planner = data.get("planner")
    model = data.get("model")
    if not planner or not model:
        return None
    return {"planner": planner, "model": model}
