# tests/test_api_chains.py
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.api.jobs import JobState
from app.api.routers.chains import _job_state_for_chain
from app.api.settings import ApiSettings
from app.supervisor import IterationContext, IterationDecision


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(ApiSettings.from_env({"DS_RUNS_DIR": str(tmp_path)})))


@pytest.mark.parametrize(
    "status,expected",
    [
        ("ok", JobState.OK),
        ("stopped", JobState.OK),
        ("max_iterations", JobState.OK),
        ("ask_user", JobState.PAUSED),
        ("failed", JobState.FAILED),
    ],
)
def test_job_state_for_chain_maps_ask_user_to_paused(status, expected) -> None:
    # `ask_user` is the LLM judge asking the human to clarify — a pause, not a
    # failure. Guards the regression where it was lumped in with FAILED.
    assert _job_state_for_chain(status) == expected


def test_create_chain_sync(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/chains",
        json={
            "prompt": "investigate",
            "run_id": "chain-1",
            "mode": "sync",
            "max_iterations": 1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chain_id"] == "chain-1"
    assert body["status"] in {"ok", "stopped", "max_iterations"}
    assert isinstance(body["steps"], list)


def test_get_chain(tmp_path) -> None:
    client = _client(tmp_path)
    client.post(
        "/chains",
        json={
            "prompt": "investigate",
            "run_id": "chain-2",
            "mode": "sync",
            "max_iterations": 1,
        },
    )
    resp = client.get("/chains/chain-2")
    assert resp.status_code == 200
    assert resp.json()["chain_id"] == "chain-2"


def test_get_chain_404(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.get("/chains/ghost").status_code == 404


def test_create_chain_async_202(tmp_path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/chains",
        json={"prompt": "x", "run_id": "chain-3", "mode": "async", "max_iterations": 1},
    )
    assert resp.status_code == 202


def test_create_chain_with_llm_decider_can_replan(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    decisions = [
        IterationDecision(
            action="replan",
            reason="Need one more pass.",
            gaps=["missing orchestration judgment"],
        ),
        IterationDecision(
            action="stop",
            reason="Second pass satisfies the rubric.",
            success_criteria_met=True,
        ),
    ]
    captured = {}

    def fake_builder(model_provider, model_ref, **kwargs):
        assert model_provider.name == "openai"
        assert model_ref.qualified_name == "openai:gpt-5.4-nano"
        captured.update(kwargs)

        def fake_decider(context: IterationContext) -> IterationDecision:
            return decisions[context.iteration - 1]

        return fake_decider

    monkeypatch.setattr("app.api.deps.build_provider_iteration_decider", fake_builder)
    client = _client(tmp_path)

    resp = client.post(
        "/chains",
        json={
            "prompt": "investigate",
            "run_id": "chain-llm-decider",
            "mode": "sync",
            "max_iterations": 2,
            "decider": "llm",
            "success_criteria": "Must mention CORE orchestration.",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["chain_id"] == "chain-llm-decider"
    assert body["status"] == "ok"
    assert [step["decision"] for step in body["steps"]] == ["replan", "stop"]
    assert body["steps"][0]["decision_detail"]["gaps"] == [
        "missing orchestration judgment"
    ]
    assert body["final_decision"]["success_criteria_met"] is True
    assert captured["success_criteria"] == "Must mention CORE orchestration."
