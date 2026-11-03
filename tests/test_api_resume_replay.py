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
        "/runs/wait-1/resume",
        json={"event": {"event_type": "human_input", "value": "hello"}},
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


def test_second_resume_of_completed_run_conflicts(tmp_path) -> None:
    # Regression: "resuming" a completed run must not re-apply the event and
    # re-execute the post-wait graph (or overwrite the recorded trace).
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "go", "run_id": "wait-2", "mode": "sync"})
    first = client.post(
        "/runs/wait-2/resume",
        json={"event": {"event_type": "human_input", "value": "a"}},
    )
    assert first.status_code == 200
    assert first.json()["status"] == "ok"

    second = client.post(
        "/runs/wait-2/resume",
        json={"event": {"event_type": "human_input", "value": "b"}},
    )

    assert second.status_code == 409
    assert second.json()["status"] == "resume_failed"


def test_get_run_reflects_state_after_resume(tmp_path) -> None:
    # Regression: the job store kept the terminal PAUSED result forever, so
    # GET /runs/{id} reported "paused" after a successful resume.
    client = _client(tmp_path)
    client.post("/runs", json={"prompt": "go", "run_id": "wait-3", "mode": "sync"})
    resumed = client.post(
        "/runs/wait-3/resume",
        json={"event": {"event_type": "human_input", "value": "a"}},
    )
    assert resumed.json()["status"] == "ok"

    status = client.get("/runs/wait-3")

    assert status.status_code == 200
    assert status.json()["state"] == "ok"


def test_replay_refuses_existing_new_run_id(tmp_path) -> None:
    # Regression: replay silently overwrote another run's recording.
    app = create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)}))
    client = TestClient(app)
    client.post("/runs", json={"prompt": "x", "run_id": "rp-a", "mode": "sync"})
    client.post("/runs", json={"prompt": "y", "run_id": "rp-b", "mode": "sync"})
    original_output = (tmp_path / "rp-b" / "output.json").read_text("utf-8")

    resp = client.post("/runs/rp-a/replay", json={"new_run_id": "rp-b"})

    assert resp.status_code == 409
    assert (tmp_path / "rp-b" / "output.json").read_text("utf-8") == original_output


def test_worker_crash_is_surfaced_not_404(tmp_path) -> None:
    # Regression: a worker exception produced a 202 whose links all 404'd and
    # an error message unreachable from any endpoint.
    app = create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)}))

    class _ExplodingSupervisor:
        def run(self, prompt, *, run_id):
            raise RuntimeError("planner infrastructure exploded")

    app.state.context.supervisor_for = (  # type: ignore[assignment]
        lambda config: _ExplodingSupervisor()
    )
    client = TestClient(app)

    created = client.post(
        "/runs", json={"prompt": "boom", "run_id": "crash-1", "mode": "sync"}
    )

    assert created.status_code == 200
    body = created.json()
    assert body["status"] == "failed"
    assert "planner infrastructure exploded" in body["error"]

    status = client.get("/runs/crash-1")
    assert status.status_code == 200
    assert status.json()["state"] == "failed"
    assert "planner infrastructure exploded" in status.json()["error"]
