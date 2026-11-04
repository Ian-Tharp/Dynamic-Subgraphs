"""Governance tests for the MCP run service (no MCP SDK needed)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.assembly import build_supervisor
from app.mcp import GovernanceError, GovernedRunService, McpSettings, NotFoundError
from app.mcp.service import narrow_policy
from app.mcp.settings import policy_from_env
from app.models import GraphSpec, NodeKind, NodeSpec
from app.models.graph_spec import EdgeSpec
from app.policy import ExecutionPolicy


def _service(tmp_path, **overrides) -> GovernedRunService:
    settings = McpSettings(runs_dir=str(tmp_path), **overrides)
    return GovernedRunService(settings)


def _wait_spec() -> GraphSpec:
    def edge(a: str, b: str) -> EdgeSpec:
        return EdgeSpec.model_validate({"from": a, "to": b})

    return GraphSpec(
        graph_id="wait-test",
        goal="pause for an approval",
        nodes=[
            NodeSpec(
                id="draft",
                kind=NodeKind.LLM_CALL,
                outputs=["draft"],
                params={"instruction": "draft it"},
            ),
            NodeSpec(
                id="approve",
                kind=NodeKind.WAIT_FOR_EVENT,
                outputs=["approval"],
                params={"event_type": "approval"},
            ),
            NodeSpec(
                id="finalize",
                kind=NodeKind.LLM_CALL,
                inputs=["approval"],
                outputs=["final"],
                params={"instruction": "finalize"},
            ),
        ],
        edges=[
            edge("START", "draft"),
            edge("draft", "approve"),
            edge("approve", "finalize"),
            edge("finalize", "END"),
        ],
    )


def _fixed_spec_factory(spec: GraphSpec):
    """A supervisor factory that runs `spec` instead of the demo plan."""

    def factory(config, **kwargs):
        return build_supervisor(replace(config, fixed_spec=spec), **kwargs)

    return factory


# ---------------------------------------------------------------- settings


def test_settings_defaults_are_mock_and_stdio() -> None:
    s = McpSettings.from_env({})
    assert s.planner == "mock"
    assert s.transport == "stdio"
    assert s.policy == ExecutionPolicy()


def test_settings_legacy_openai_alias() -> None:
    s = McpSettings.from_env({"DS_PLANNER": "openai", "DS_PROVIDER": "anthropic"})
    assert (s.planner, s.provider) == ("llm", "openai")


@pytest.mark.parametrize(
    "env",
    [
        {"DS_PLANNER": "magic"},
        {"DS_MCP_MAX_NODES": "lots"},
        {"DS_MCP_MAX_NODES": "0"},
        {"DS_MCP_TRANSPORT": "carrier-pigeon"},
        {"DS_MCP_ALLOWED_NODE_KINDS": "llm_call,python_eval"},
        {"DS_MCP_MAX_PLAN_ATTEMPTS": "0"},
    ],
)
def test_settings_fail_closed_on_bad_values(env) -> None:
    with pytest.raises(ValueError):
        McpSettings.from_env(env)


def test_policy_allow_sets_keep_sentinel_contract() -> None:
    assert policy_from_env({}).allowed_tools is None
    assert policy_from_env({"DS_MCP_ALLOWED_TOOLS": "*"}).allowed_tools is None
    # Set-but-empty is ban-all, never "no narrowing".
    assert policy_from_env({"DS_MCP_ALLOWED_TOOLS": ""}).allowed_tools == frozenset()
    policy = policy_from_env(
        {
            "DS_MCP_ALLOWED_TOOLS": "web_search, policy_lookup",
            "DS_MCP_ALLOWED_NODE_KINDS": "llm_call,reduce",
            "DS_MCP_MAX_NODES": "5",
        }
    )
    assert policy.allowed_tools == {"web_search", "policy_lookup"}
    assert policy.allowed_node_kinds == {NodeKind.LLM_CALL, NodeKind.REDUCE}
    assert policy.max_nodes == 5


# ------------------------------------------------------------ narrowing


def test_narrow_policy_can_only_tighten() -> None:
    host = ExecutionPolicy(max_nodes=6, max_llm_calls=4, max_wall_seconds=30)
    narrowed = narrow_policy(host, max_nodes=3, max_llm_calls=100)
    assert narrowed.max_nodes == 3
    assert narrowed.max_llm_calls == 4  # request above host is capped
    assert narrow_policy(host) is host


@pytest.mark.parametrize(
    "kwargs", [{"max_nodes": 0}, {"max_llm_calls": -1}, {"max_wall_seconds": 0}]
)
def test_narrow_policy_rejects_out_of_range(kwargs) -> None:
    with pytest.raises(GovernanceError):
        narrow_policy(ExecutionPolicy(), **kwargs)


# ------------------------------------------------------------------ runs


def test_run_mock_is_recorded_with_audit_trail(tmp_path) -> None:
    service = _service(tmp_path)
    out = service.run("compare A and B", run_id="mcp-1")

    assert out["status"] == "ok" and out["ok"] and out["recorded"]
    assert out["plan"]["nodes"]
    assert out["values"]
    assert out["granted_budget"]["max_nodes"] <= service.settings.policy.max_nodes
    assert out["usage"]["total_tokens"] == 0  # mock is token-free
    assert out["receipt"]["status"] == "ok"
    assert out["receipt"]["recorder_run_id"] == "mcp-1"
    stored = json.loads(service.get_run_file("mcp-1", "receipt"))
    assert stored["receipt_hash"] == out["receipt"]["receipt_hash"]

    run = service.get_run("mcp-1")
    assert run["ok"] is True
    assert run["config"] == {
        "planner": "mock",
        "provider": "openai",
        "model": "gpt-5.4-nano",
    }
    assert {"spec", "output", "trace", "graph"} <= set(run["files"])
    spec = json.loads(service.get_run_file("mcp-1", "spec"))
    assert spec["graph_id"] == out["plan"]["graph_id"]
    assert "graph TD" in service.get_run_file("mcp-1", "graph")
    assert service.list_runs()["runs"][0]["run_id"] == "mcp-1"


def test_run_generates_unique_ids(tmp_path) -> None:
    service = _service(tmp_path)
    a = service.run("x")["run_id"]
    b = service.run("x")["run_id"]
    assert a != b


def test_run_refuses_to_clobber_existing_run(tmp_path) -> None:
    service = _service(tmp_path)
    service.run("x", run_id="dup")
    with pytest.raises(GovernanceError, match="already exists"):
        service.run("y", run_id="dup")


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "."])
def test_run_rejects_unsafe_run_ids(tmp_path, bad_id) -> None:
    with pytest.raises(GovernanceError):
        _service(tmp_path).run("x", run_id=bad_id)


def test_run_rejects_empty_prompt(tmp_path) -> None:
    with pytest.raises(GovernanceError):
        _service(tmp_path).run("   ")


def test_llm_planner_refused_on_mock_server(tmp_path) -> None:
    with pytest.raises(GovernanceError, match="disabled"):
        _service(tmp_path).run("x", planner="llm")


def test_model_must_be_allowlisted(tmp_path) -> None:
    with pytest.raises(GovernanceError, match="allowlist"):
        _service(tmp_path).run("x", model="gpt-huge")


def test_llm_planner_without_credentials_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = _service(tmp_path, planner="llm")
    with pytest.raises(GovernanceError):
        service.run("x")
    assert service.list_runs()["count"] == 0  # refused before any work


def test_caller_budget_narrowing_fails_closed(tmp_path) -> None:
    # The demo plan has more than one node, so a 1-node ceiling must reject it
    # at validation — and the failure is still recorded.
    service = _service(tmp_path, max_plan_attempts=1)
    out = service.run("compare A and B", run_id="tight", max_nodes=1)
    assert out["status"] == "validation_failed"
    assert out["host_policy"]["max_nodes"] == 1
    assert out["errors"]
    assert service.recorder.exists("tight")


def test_plan_gate_denial_returns_policy_denied_with_receipt(tmp_path) -> None:
    # A plan-length ceiling of 1 refuses the multi-node demo plan at the #60
    # plan gate; the tool result must report it, not crash, and carry a receipt.
    service = _service(
        tmp_path, policy=ExecutionPolicy(max_plan_length=1), max_plan_attempts=1
    )
    out = service.run("compare A and B", run_id="denied")
    assert out["status"] == "policy_denied"
    assert not out["ok"] and not out["healthy"]
    assert out["receipt"]["status"] == "policy_denied"


def test_host_policy_node_kind_ban_is_enforced(tmp_path) -> None:
    policy = ExecutionPolicy(allowed_node_kinds=frozenset({NodeKind.LLM_CALL}))
    service = _service(tmp_path, policy=policy, max_plan_attempts=1)
    out = service.run("compare A and B")  # demo plan uses tool_call + reduce
    assert out["status"] == "validation_failed"
    assert [k["kind"] for k in service.describe_registry()["node_kinds"]] == [
        "llm_call"
    ]


def test_values_are_clipped_in_tool_results(tmp_path) -> None:
    service = _service(tmp_path, max_value_chars=10)
    out = service.run("compare A and B", run_id="clip")
    text = json.dumps(out["values"])
    assert "truncated" in text
    # Full values stay on disk.
    full = json.loads(service.get_run_file("clip", "output"))["values"]
    assert "truncated" not in json.dumps(full)


# ------------------------------------------------------- resume / replay


def test_pause_then_resume(tmp_path) -> None:
    service = GovernedRunService(
        McpSettings(runs_dir=str(tmp_path)),
        supervisor_factory=_fixed_spec_factory(_wait_spec()),
    )
    paused = service.run("draft and wait", run_id="wait-1")
    assert paused["status"] == "paused" and paused["healthy"]

    resumed = service.resume("wait-1", event={"approved": True})
    assert resumed["status"] == "ok"
    assert resumed["values"]["approval"] == {"approved": True}


def test_resume_unknown_run(tmp_path) -> None:
    with pytest.raises(NotFoundError):
        _service(tmp_path).resume("nope", event={})


def test_replay_runs_recorded_plan_under_new_id(tmp_path) -> None:
    service = _service(tmp_path)
    original = service.run("compare A and B", run_id="orig")
    replayed = service.replay("orig")
    assert replayed["status"] == "ok"
    assert replayed["replayed_from"] == "orig"
    assert replayed["run_id"] != "orig"
    assert replayed["plan"]["graph_id"] == original["plan"]["graph_id"]
    assert service.get_run(replayed["run_id"])["config"]["planner"] == "mock"


def test_replay_refuses_to_clobber(tmp_path) -> None:
    service = _service(tmp_path)
    service.run("x", run_id="a")
    service.run("x", run_id="b")
    with pytest.raises(GovernanceError, match="already exists"):
        service.replay("a", new_run_id="b")


def test_replay_of_llm_run_refused_after_server_downgrade(tmp_path) -> None:
    service = _service(tmp_path)
    service.run("x", run_id="was-llm")
    # Simulate a run recorded by an LLM-enabled server.
    (service.recorder.run_dir("was-llm") / "api_config.json").write_text(
        json.dumps({"planner": "llm", "provider": "openai", "model": "gpt-5.4-nano"})
    )
    with pytest.raises(GovernanceError, match="disabled"):
        service.replay("was-llm")


# ----------------------------------------------------------------- reads


def test_get_run_file_rejects_unknown_names(tmp_path) -> None:
    service = _service(tmp_path)
    service.run("x", run_id="r")
    with pytest.raises(GovernanceError, match="unknown run file"):
        service.get_run_file("r", "../../etc/passwd")
    with pytest.raises(NotFoundError):
        service.get_run_file("r", "eval")  # not scored


def test_get_artifact_is_path_safe(tmp_path) -> None:
    service = _service(tmp_path)
    service.run("x", run_id="r")
    with pytest.raises(GovernanceError):
        service.get_artifact("r", "../spec.json")
    with pytest.raises(NotFoundError):
        service.get_artifact("r", "missing.txt")


def test_get_run_unknown(tmp_path) -> None:
    with pytest.raises(NotFoundError):
        _service(tmp_path).get_run("ghost")


def test_server_info_reports_host_limits(tmp_path) -> None:
    info = _service(tmp_path, policy=ExecutionPolicy(max_nodes=4)).server_info()
    assert info["policy"]["max_nodes"] == 4
    assert info["llm_planner_available"] is False
