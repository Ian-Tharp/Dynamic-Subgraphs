"""Real end-to-end API tests that make ACTUAL model + tool calls.

These are gated and skipped by default — they cost tokens. To run them:

    # PowerShell
    $env:DS_RUN_INTEGRATION="1"; .venv/Scripts/python.exe -m pytest tests/test_integration_api.py -v

    # bash
    DS_RUN_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/test_integration_api.py -v

Requires OPENAI_API_KEY (loaded from .env) and network access (web_search tool).
They exercise the path the mock suite cannot: HTTP -> real LLM planner ->
validate -> compile -> execute (real tools + model) -> record -> serve back.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.api.settings import ApiSettings

_ENABLED = os.environ.get("DS_RUN_INTEGRATION") == "1" and bool(
    os.environ.get("OPENAI_API_KEY")
)

pytestmark = pytest.mark.skipif(
    not _ENABLED,
    reason="set DS_RUN_INTEGRATION=1 (and OPENAI_API_KEY) to run real-call e2e",
)

MODEL = os.environ.get("DS_MODEL", "gpt-5.4-nano")


def _client(tmp_path: Path) -> TestClient:
    settings = ApiSettings.from_env(
        {"DS_RUNS_DIR": str(tmp_path), "DS_MODEL_ALLOWLIST": MODEL}
    )
    return TestClient(create_app(settings))


def _openai(prompt: str, **extra) -> dict:
    body = {"prompt": prompt, "planner": "openai", "model": MODEL}
    body.update(extra)
    return body


def _poll_until_terminal(client: TestClient, run_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = client.get(f"/runs/{run_id}").json()
        if last.get("state") in {"ok", "failed", "paused"}:
            return last
        time.sleep(1.0)
    return last


def test_real_sync_run_uses_real_model(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/runs",
        json=_openai("Compare SQLite and DuckDB for local analytics; recommend one.", mode="sync"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["spec"]["nodes"]
    # The real model must have produced output — never the mock echo.
    import json as _json

    assert "<mock-llm>" not in _json.dumps(body["values"])


def test_real_async_run_completes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/runs",
        json=_openai("Summarize two vector-index approaches and recommend one.", mode="async", run_id="it-async"),
    )
    assert resp.status_code == 202
    final = _poll_until_terminal(client, "it-async")
    assert final["state"] == "ok"


def test_real_resume_finishes_with_real_model(tmp_path: Path) -> None:
    """Regression guard for the resume config-preservation bug.

    A run created with planner=openai that pauses MUST be finished by the real
    model, not the server-default mock runner.
    """
    client = _client(tmp_path)
    created = client.post(
        "/runs",
        json=_openai(
            "Draft a short product announcement. Then wait for human approval "
            "before finalizing. After approval, produce the final announcement.",
            mode="sync",
            run_id="it-resume",
        ),
    )
    assert created.status_code == 200
    if created.json()["status"] != "paused":
        pytest.skip("planner did not emit a wait_for_event graph this run")

    resumed = client.post(
        "/runs/it-resume/resume",
        json={"event": {"event_type": "human_approval", "approved": True}},
    )
    assert resumed.status_code == 200
    body = resumed.json()
    assert body["status"] == "ok"

    import json as _json

    # The whole point: resume must NOT fall back to the mock runner.
    assert "<mock-llm>" not in _json.dumps(body["values"])


def test_real_chain_runs(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post(
        "/chains",
        json=_openai(
            "Investigate whether Postgres or MySQL fits a small SaaS better.",
            mode="sync",
            run_id="it-chain",
            max_iterations=2,
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "stopped", "max_iterations"}
    assert len(body["steps"]) >= 1


def test_real_chain_with_llm_decider(tmp_path: Path) -> None:
    """Real-call proof for POST /chains with decider='llm'.

    The structured LLM iteration judge must actually run after each iteration —
    proven by a model-generated final reason (never the StatusIterationDecider's
    fixed fallback string) and structured decision detail on every step. This is
    the path the mock suite cannot exercise.
    """
    client = _client(tmp_path)
    resp = client.post(
        "/chains",
        json=_openai(
            "Compare Redis and Memcached for caching a small web app; recommend one.",
            mode="sync",
            run_id="it-chain-llm",
            max_iterations=2,
            decider="llm",
            success_criteria="A clear recommendation with at least one concrete tradeoff.",
        ),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "stopped", "max_iterations"}
    assert len(body["steps"]) >= 1

    # Every step carries the structured judge decision.
    for step in body["steps"]:
        assert step["decision_detail"] is not None
        assert step["decision_detail"]["reason"]

    # The real LLM judge ran: its reason is model-generated, never the
    # StatusIterationDecider's fixed fallback string. This distinguishes a true
    # decider='llm' run from a silent fallback to the token-free status decider.
    final = body["final_decision"]
    assert final is not None and final["reason"]
    assert final["reason"] != "The bounded run completed successfully."

    import json as _json

    assert "<mock-llm>" not in _json.dumps(body)


def test_real_spawn_subgraph_runs_a_real_child(tmp_path: Path) -> None:
    """Real-call proof for the spawn_subgraph primitive (nested-subgraphs slice 3).

    A hand-built parent graph whose only node is a spawn_subgraph: it plans and
    runs a child with the REAL planner + model, on its own envelope one level
    deeper, and hands the child's produced values back to the parent. The
    planner is gated from emitting spawn_subgraph itself, so we wire it
    explicitly — exactly the late-binding `assembly.build_supervisor` uses.
    """
    from app.models import GraphSpec, NodeKind, NodeSpec
    from app.models.graph_spec import EdgeSpec
    from app.registry import validate_graph_spec
    from app.runtime import (
        build_grounded_tool_runner,
        build_openai_llm_runner,
        build_openai_reduce_runner,
    )
    from app.runtime.executor import LangGraphExecutor
    from app.runtime.subgraph import build_spawn_subgraph_runner, make_child_launcher
    from app.supervisor import build_openai_planner

    planner = build_openai_planner(model=MODEL)
    runners = {
        NodeKind.LLM_CALL: build_openai_llm_runner(model=MODEL),
        NodeKind.TOOL_CALL: build_grounded_tool_runner(),
        NodeKind.REDUCE: build_openai_reduce_runner(model=MODEL),
    }
    executor = LangGraphExecutor(runners=runners)
    runners[NodeKind.SPAWN_SUBGRAPH] = build_spawn_subgraph_runner(
        make_child_launcher(planner=planner, executor=executor)
    )

    parent = GraphSpec(
        graph_id="parent-real",
        goal="delegate a sub-task to a child graph",
        nodes=[
            NodeSpec(
                id="spawn",
                kind=NodeKind.SPAWN_SUBGRAPH,
                outputs=["child_report"],
                params={
                    "sub_goal": "List two concrete benefits of SQLite for a small "
                    "local app, one sentence each.",
                    "name": "bench",
                },
            )
        ],
        edges=[
            EdgeSpec.model_validate({"from": "START", "to": "spawn"}),
            EdgeSpec.model_validate({"from": "spawn", "to": "END"}),
        ],
    )
    validated = validate_graph_spec(parent)
    result = executor.execute(executor.compile(validated), run_id="real-parent")

    assert result.ok is True
    report = result.state["values"]["child_report"]

    import json as _json

    dumped = _json.dumps(report)
    # The real model produced the child's output — never the mock echo.
    assert "<mock-llm>" not in dumped
    assert len(dumped) > 40  # substantive real output, not an empty child
