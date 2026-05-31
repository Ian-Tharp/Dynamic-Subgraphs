# tests/test_api_resume_replay.py
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings
from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec, GraphBudget
from app.supervisor import StaticPlanner


def _wait_spec() -> GraphSpec:
    return GraphSpec(
        graph_id="wait-graph",
        goal="pause then finish",
        budget=GraphBudget(max_nodes=8),
        nodes=[
            NodeSpec(
                id="hold",
                kind=NodeKind.WAIT_FOR_EVENT,
                outputs=["signal"],
                params={"event_type": "human_input", "output_key": "signal"},
            ),
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "hold"}),
            EdgeSpec.model_validate({"from": "hold", "to": "END"}),
        ],
    )


def _client(tmp_path) -> TestClient:
    app = create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)}))
    # Force the wait-graph planner so a run pauses. The supervisor compiles its
    # graph (and captures the planner) at construction time, so we must rebuild
    # the Supervisor with the wait planner rather than swap ._planner after the
    # fact. We reuse the executor (with its shared checkpointer) and recorder so
    # resume across requests still works.
    from app.supervisor import Supervisor

    original = app.state.context.supervisor_for

    def patched(config):
        sup = original(config)
        return Supervisor(
            planner=StaticPlanner(_wait_spec()),
            executor=sup._executor,
            recorder=sup._recorder,
        )

    app.state.context.supervisor_for = patched  # type: ignore[assignment]
    return TestClient(app)


def test_resume_completes_paused_run(tmp_path) -> None:
    client = _client(tmp_path)
    created = client.post(
        "/runs", json={"prompt": "go", "run_id": "wait-1", "mode": "sync"}
    )
    assert created.status_code == 200
    assert created.json()["status"] == "paused"

    resumed = client.post(
        "/runs/wait-1/resume", json={"event": {"event_type": "human_input", "value": "hello"}}
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "ok"


def test_replay_creates_new_run(tmp_path) -> None:
    # Replay needs a completed run; use the default mock planner instead.
    app = create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)}))
    client = TestClient(app)
    client.post("/runs", json={"prompt": "x", "run_id": "rp-1", "mode": "sync"})
    resp = client.post("/runs/rp-1/replay", json={"new_run_id": "rp-1-replay"})
    assert resp.status_code == 200
    assert resp.json()["run_id"] == "rp-1-replay"
