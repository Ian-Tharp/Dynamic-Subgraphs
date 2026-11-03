"""MCP protocol round-trip tests (in-memory client ↔ server)."""

from __future__ import annotations

import json

import anyio
import pytest

pytest.importorskip("mcp")

from mcp.client.client import Client

from app.mcp import GovernedRunService, McpSettings
from app.mcp.server import build_server


def _server(tmp_path):
    return build_server(GovernedRunService(McpSettings(runs_dir=str(tmp_path))))


def _run(coro_fn):
    return anyio.run(coro_fn)


def test_tools_are_listed_with_annotations(tmp_path) -> None:
    async def main():
        async with Client(_server(tmp_path)) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in _run(main)}
    assert set(tools) == {
        "run_subgraph",
        "resume_run",
        "replay_run",
        "server_info",
        "describe_registry",
        "list_runs",
        "get_run",
        "get_run_file",
        "get_artifact",
    }
    assert tools["get_run"].annotations.read_only_hint is True
    assert tools["run_subgraph"].annotations.read_only_hint is False
    assert "prompt" in tools["run_subgraph"].input_schema["required"]


def test_run_subgraph_round_trip(tmp_path) -> None:
    async def main():
        async with Client(_server(tmp_path)) as client:
            ran = await client.call_tool(
                "run_subgraph", {"prompt": "compare A and B", "run_id": "mcp-rt"}
            )
            spec = await client.call_tool(
                "get_run_file", {"run_id": "mcp-rt", "name": "spec"}
            )
            resource = await client.read_resource("ds://runs/mcp-rt/graph")
            return ran, spec, resource

    ran, spec, resource = _run(main)
    assert not ran.is_error
    assert ran.structured_content["status"] == "ok"
    assert ran.structured_content["run_id"] == "mcp-rt"
    assert not spec.is_error
    assert json.loads(spec.content[0].text)["graph_id"]
    assert "graph TD" in resource.contents[0].text


def test_governance_refusal_is_a_tool_error(tmp_path) -> None:
    async def main():
        async with Client(_server(tmp_path)) as client:
            llm = await client.call_tool(
                "run_subgraph", {"prompt": "x", "planner": "llm"}
            )
            missing = await client.call_tool("get_run", {"run_id": "ghost"})
            return llm, missing

    llm, missing = _run(main)
    assert llm.is_error and "disabled" in llm.content[0].text
    assert missing.is_error and "ghost" in missing.content[0].text


def test_registry_resource(tmp_path) -> None:
    async def main():
        async with Client(_server(tmp_path)) as client:
            return await client.read_resource("ds://registry")

    body = json.loads(_run(main).contents[0].text)
    assert any(k["kind"] == "llm_call" for k in body["node_kinds"])
