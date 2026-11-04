"""MCP adapter: expose `GovernedRunService` as MCP tools and resources.

This module is the only place that imports the MCP SDK (the optional `mcp`
extra). It adds no behavior of its own — every tool delegates to the service,
which owns the governance rules — so an MCP host gets exactly the same
plan → validate → run → record contract as the SDK and HTTP API.

Runs execute in a worker thread so a long run never blocks the server's event
loop; each is bounded by the host policy's `max_wall_seconds`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .service import RUN_FILES, GovernanceError, GovernedRunService, NotFoundError

T = TypeVar("T")

SERVER_NAME = "dynamic-subgraphs"

INSTRUCTIONS = """\
Dynamic Subgraphs runs *governed* LangGraph workflows. Give `run_subgraph` a
task; a planner turns it into a validated plan (a GraphSpec — data, never code)
built only from allowlisted node kinds, which is compiled, executed under the
host's budget, and recorded under a run id.

Typical flow:
1. `server_info` / `describe_registry` to see the limits and building blocks.
2. `run_subgraph` with the task. Check `status` (ok, paused, plan_failed,
   validation_failed, policy_denied, execution_failed, ...) and read
   `response` / `values`. `receipt` is the run's sealed provenance receipt.
3. If `status` is "paused", the plan is waiting on an external event: call
   `resume_run` with that event.
4. Inspect any run with `get_run`, `get_run_file` (spec, trace, graph, ...),
   and `get_artifact`; `replay_run` re-executes a recorded plan verbatim.
5. Audit: `get_receipt` returns a run's receipt and its receipt-ledger history;
   `verify_receipt` re-checks it against the recorded files and the ledger;
   `verify_ledger` checks the ledger's whole hash chain and signatures.

You may tighten a run's limits (max_nodes, max_llm_calls, max_wall_seconds) but
never exceed the host policy. Failed runs are recorded too — read their errors.
"""

RunFileName = Literal[
    "spec", "output", "trace", "graph", "summary", "prompt", "eval", "receipt"
]

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


async def _call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking service call off the event loop, mapping refusals."""

    try:
        return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))
    except (GovernanceError, NotFoundError) as exc:
        raise ToolError(str(exc)) from exc


def build_server(service: GovernedRunService) -> MCPServer:
    """Create the MCP server bound to `service`."""

    server = MCPServer(
        name=SERVER_NAME,
        title="Dynamic Subgraphs",
        instructions=INSTRUCTIONS,
    )

    # ------------------------------------------------------------------ tools

    @server.tool(
        title="Run a governed subgraph",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            # LLM runs may call the web_search tool.
            open_world_hint=True,
        ),
    )
    async def run_subgraph(
        prompt: Annotated[str, Field(description="The task to plan and execute.")],
        run_id: Annotated[
            str | None,
            Field(description="Optional new run id; generated when omitted."),
        ] = None,
        planner: Annotated[
            Literal["mock", "llm"] | None,
            Field(
                description=(
                    "'mock' is token-free; 'llm' only works when the server "
                    "enables it. Defaults to the server's planner."
                )
            ),
        ] = None,
        provider: Annotated[
            str | None, Field(description="Model provider (e.g. openai).")
        ] = None,
        model: Annotated[
            str | None, Field(description="Model name; must be allowlisted.")
        ] = None,
        max_nodes: Annotated[
            int | None, Field(description="Tighten the node budget for this run.")
        ] = None,
        max_llm_calls: Annotated[
            int | None, Field(description="Tighten the LLM-call budget.")
        ] = None,
        max_wall_seconds: Annotated[
            int | None, Field(description="Tighten the wall-clock limit.")
        ] = None,
    ) -> dict[str, Any]:
        """Plan, validate, execute, and record a governed dynamic subgraph.

        The planner emits a GraphSpec from allowlisted node kinds; the host
        policy caps its budget; the run is always recorded (even on failure).
        Returns status, response, values, the plan, the granted budget, and
        token usage.
        """

        return await _call(
            service.run,
            prompt,
            run_id=run_id,
            planner=planner,
            provider=provider,
            model=model,
            max_nodes=max_nodes,
            max_llm_calls=max_llm_calls,
            max_wall_seconds=max_wall_seconds,
        )

    @server.tool(
        title="Resume a paused run",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        ),
    )
    async def resume_run(
        run_id: Annotated[str, Field(description="The paused run's id.")],
        event: Annotated[
            Any, Field(description="The event payload the run is waiting for.")
        ],
    ) -> dict[str, Any]:
        """Resume a run paused on `wait_for_event` by supplying the event.

        Paused state lives in this server process, so resume works for runs
        started by the same server session.
        """

        return await _call(service.resume, run_id, event=event)

    @server.tool(
        title="Replay a recorded run",
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        ),
    )
    async def replay_run(
        run_id: Annotated[str, Field(description="The recorded run to replay.")],
        new_run_id: Annotated[
            str | None,
            Field(description="Optional id for the replay; must be new."),
        ] = None,
    ) -> dict[str, Any]:
        """Re-execute a recorded plan verbatim under a new run id (no re-plan).

        The plan is re-validated against the current host policy first.
        """

        return await _call(service.replay, run_id, new_run_id=new_run_id)

    @server.tool(title="Server limits and configuration", annotations=_READ_ONLY)
    async def server_info() -> dict[str, Any]:
        """The host policy, planner, and model allowlist every run is held to."""

        return await _call(service.server_info)

    @server.tool(title="Describe the node registry", annotations=_READ_ONLY)
    async def describe_registry() -> dict[str, Any]:
        """Node kinds, tools, and subagents a plan may use under the host policy."""

        return await _call(service.describe_registry)

    @server.tool(title="List recorded runs", annotations=_READ_ONLY)
    async def list_runs(
        limit: Annotated[
            int, Field(description="Return at most this many (most recent).")
        ] = 50,
    ) -> dict[str, Any]:
        """Summaries of recorded runs: id, status, node count, eval scores."""

        return await _call(service.list_runs, limit=limit)

    @server.tool(title="Get a run", annotations=_READ_ONLY)
    async def get_run(
        run_id: Annotated[str, Field(description="The run id.")],
    ) -> dict[str, Any]:
        """A recorded run's outcome, values, errors, config, and readable files."""

        return await _call(service.get_run, run_id)

    @server.tool(title="Read a run file", annotations=_READ_ONLY)
    async def get_run_file(
        run_id: Annotated[str, Field(description="The run id.")],
        name: Annotated[
            RunFileName,
            Field(
                description=(
                    "spec (the validated GraphSpec), output, trace (per-node "
                    "events, JSONL), graph (Mermaid), summary, prompt, eval, "
                    "or receipt (provenance receipt)."
                )
            ),
        ],
    ) -> str:
        """Read one recorded file from a run's audit trail."""

        return await _call(service.get_run_file, run_id, name)

    @server.tool(title="Read a run artifact", annotations=_READ_ONLY)
    async def get_artifact(
        run_id: Annotated[str, Field(description="The run id.")],
        name: Annotated[str, Field(description="Artifact file name.")],
    ) -> str:
        """Read a file an `emit_artifact` node wrote during the run."""

        return await _call(service.get_artifact, run_id, name)

    @server.tool(title="Get a run's provenance receipt", annotations=_READ_ONLY)
    async def get_receipt(
        run_id: Annotated[str, Field(description="The run id.")],
    ) -> dict[str, Any]:
        """A run's sealed provenance receipt and its receipt-ledger entries.

        `ledger.history` lists every receipt anchored for the run, oldest
        first; each entry's `supersedes` names the receipt it replaced (for
        example the paused receipt before a resume).
        """

        return await _call(service.get_receipt, run_id)

    @server.tool(title="Verify a run's receipt", annotations=_READ_ONLY)
    async def verify_receipt(
        run_id: Annotated[str, Field(description="The run id.")],
    ) -> dict[str, Any]:
        """Re-check a run's receipt against its recorded files and the ledger.

        `ok` is false when the receipt, spec, trace, logged gate decisions or
        output were changed, when the receipt isn't the run's latest anchored
        receipt, or when the ledger entry's signature doesn't verify.
        `problems` says what failed.
        """

        return await _call(service.verify_receipt, run_id)

    @server.tool(title="Verify the receipt ledger", annotations=_READ_ONLY)
    async def verify_ledger() -> dict[str, Any]:
        """Check the receipt ledger's hash chain and entry signatures.

        Returns `ok`, `problems`, the entry count, how many entries are
        signed, and `head` (the newest entry hash; anchor it elsewhere to
        detect truncation later).
        """

        return await _call(service.verify_ledger)

    # -------------------------------------------------------------- resources

    @server.resource(
        "ds://server",
        name="server-info",
        description="Host policy and model configuration.",
        mime_type="application/json",
    )
    def server_resource() -> str:
        return json.dumps(service.server_info(), indent=2)

    @server.resource(
        "ds://registry",
        name="registry",
        description="Node kinds, tools, and subagents available to plans.",
        mime_type="application/json",
    )
    def registry_resource() -> str:
        return json.dumps(service.describe_registry(), indent=2)

    @server.resource(
        "ds://runs/{run_id}/{name}",
        name="run-file",
        description=f"A recorded run file; name is one of {sorted(RUN_FILES)}.",
    )
    def run_file_resource(run_id: str, name: str) -> str:
        return service.get_run_file(run_id, name)

    return server
