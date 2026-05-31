# app/api/serialize.py
"""Turn internal result objects into JSON-ready response dicts."""

from __future__ import annotations

from typing import Any

from app.supervisor import SupervisorResult


def _spec_dict(result: SupervisorResult) -> dict[str, Any] | None:
    if result.validated_spec is None:
        return None
    return result.validated_spec.model_dump(mode="json", by_alias=True)


def _budget_wall_seconds(result: SupervisorResult) -> int | None:
    spec = result.validated_spec
    if spec is None or spec.budget is None:
        return None
    return getattr(spec.budget, "max_wall_seconds", None)


def run_links(run_id: str) -> dict[str, str]:
    base = f"/runs/{run_id}"
    return {
        "self": base,
        "spec": f"{base}/spec",
        "trace": f"{base}/trace",
        "stream": f"{base}/trace/stream",
        "output": f"{base}/output",
        "graph": f"{base}/graph",
        "summary": f"{base}/summary",
        "artifacts": f"{base}/artifacts",
    }


def run_result_payload(result: SupervisorResult) -> dict[str, Any]:
    values: dict[str, Any] = {}
    artifacts: list[str] = []
    if result.result is not None:
        state = result.result.state
        values = dict(state.get("values", {}))
        artifacts = sorted(state.get("artifacts", {}).keys())
    record_dir = str(result.record.directory) if result.record is not None else None
    return {
        "run_id": result.run_id,
        "status": result.status,
        "response": result.response,
        "spec": _spec_dict(result),
        "values": values,
        "errors": list(result.errors),
        "record_dir": record_dir,
        "artifacts": artifacts,
        "links": run_links(result.run_id),
    }


def _state_from_status(status: str) -> str:
    if status == "ok":
        return "ok"
    if status == "paused":
        return "paused"
    return "failed"


def run_status_payload(result: SupervisorResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "state": _state_from_status(result.status),
        "status": result.status,
        "response": result.response,
        "budget_wall_seconds": _budget_wall_seconds(result),
        "on_disk": result.record is not None,
        "links": run_links(result.run_id),
    }
