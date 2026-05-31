"""Default `NodeRunner` implementations registered in `default_runners()`.

These are **placeholders** for the production runners. Each one is small,
deterministic, and pure: no network, no filesystem, no real model calls.
They exist so:

- the pipeline can be exercised end-to-end in tests without spending tokens;
- `python -m app.main` (without `--llm`) demonstrates the full flow with
  predictable output;
- the planner can advertise the kinds even when production wiring is partial.

For real execution, callers pass concrete `runners=` overrides to
`LangGraphExecutor`. Each kind has a production factory:

| Kind            | Echo (here)            | Production factory                                  |
|-----------------|------------------------|-----------------------------------------------------|
| llm_call        | `run_llm_call`         | `build_openai_llm_runner` -> `OpenAILlmRunner`      |
| tool_call       | `run_tool_call` (fake) | `build_grounded_tool_runner`                        |
| reduce          | `run_reduce`           | `build_openai_reduce_runner` -> `LlmReduceRunner`   |
| spawn_subagent  | `run_spawn_subagent`   | `build_openai_spawn_subagent_runner` (via subagents)|
| emit_artifact   | `run_emit_artifact`    | `make_emit_artifact_runner(FileArtifactSink(...))`  |
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from app.models import DynamicRunState, NodeKind


class NodeRunner(Protocol):
    """Execute one registry-approved node kind and return raw node outputs."""

    def __call__(
        self,
        state: DynamicRunState,
        params: Mapping[str, Any],
    ) -> dict[str, Any]: ...


ToolCallable = Callable[[Mapping[str, Any]], Any]


def _echo_tool(tool_name: str) -> ToolCallable:
    def _run(args: Mapping[str, Any]) -> dict[str, Any]:
        return {"tool_name": tool_name, "args": dict(args)}

    return _run


# Echo "tools" used by the default `run_tool_call`. These do no real I/O —
# they exist so the planner can synthesize `tool_call` nodes against
# names matching the registry's allowlist without crashing tests/demos.
# Wire real tool implementations via `runners={NodeKind.TOOL_CALL: ...}`.
DEFAULT_FAKE_TOOLS: dict[str, ToolCallable] = {
    "web_search": _echo_tool("web_search"),
    "policy_lookup": _echo_tool("policy_lookup"),
    "create_follow_up_task": _echo_tool("create_follow_up_task"),
    "mock_document_extract": _echo_tool("mock_document_extract"),
}


def run_llm_call(
    state: DynamicRunState,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Echo / placeholder default for `llm_call`.

    Returns the planner's `instruction` verbatim as the result — no model
    call. For real LLM execution, override with
    `OpenAILlmRunner` (via `build_openai_llm_runner(...)`).
    """

    del state
    return {"result": params["instruction"]}


def run_tool_call(
    state: DynamicRunState,
    params: Mapping[str, Any],
    *,
    tools: Mapping[str, ToolCallable] | None = None,
) -> dict[str, Any]:
    """Echo / placeholder default for `tool_call`.

    Resolves `params.tool_name` against `DEFAULT_FAKE_TOOLS` (or the
    `tools=` override) and returns the fake tool's structured echo.
    For production use, pass a runner backed by your real tool registry via
    `runners={NodeKind.TOOL_CALL: ...}`. The project ships
    `build_grounded_tool_runner()` for the default allowlist.
    """

    del state
    available_tools = tools or DEFAULT_FAKE_TOOLS
    tool_name = str(params["tool_name"])
    try:
        tool = available_tools[tool_name]
    except KeyError as exc:
        raise ValueError(f"No runtime tool registered for '{tool_name}'") from exc
    return {"result": tool(params.get("args", {}))}


def run_spawn_subagent(
    state: DynamicRunState,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Echo / placeholder default for `spawn_subagent`.

    Returns `<subagent:NAME>TASK</subagent>` — no real subagent call.
    For role-delegated reasoning, wire a real registry via
    `make_spawn_subagent_runner(subagents=...)` or use the
    `build_openai_spawn_subagent_runner(...)` factory.
    """

    del state
    agent_name = str(params["agent_name"])
    task = str(params["task"])
    return {"result": f"<subagent:{agent_name}>{task}</subagent>"}


def run_reduce(
    state: DynamicRunState,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministic reducers for `concat` and `merge_dict` strategies.

    Note: `llm_summarize` is NOT implemented here — it requires a chat
    model. For that strategy, override with `LlmReduceRunner` (via
    `build_openai_reduce_runner(...)`). Calling this runner with
    `strategy="llm_summarize"` raises `NotImplementedError`.
    """

    values = state.get("values", {})
    strategy = params.get("strategy", "concat")
    input_keys = list(params["input_keys"])
    output_key = str(params["output_key"])

    if strategy == "concat":
        joined = "\n".join(str(values[key]) for key in input_keys)
        return {output_key: joined}

    if strategy == "merge_dict":
        merged: dict[str, Any] = {}
        for key in input_keys:
            value = values[key]
            if not isinstance(value, dict):
                raise TypeError(f"reduce.merge_dict expected '{key}' to hold a dict")
            merged.update(value)
        return {output_key: merged}

    raise NotImplementedError(
        "reduce.llm_summarize is not implemented by the default reducer. "
        "Override with LlmReduceRunner (build_openai_reduce_runner)."
    )


def run_emit_artifact(
    state: DynamicRunState,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Echo / placeholder default for `emit_artifact`.

    Returns a fake reference dict without performing any I/O. For real
    persistence, wire a real sink via
    `make_emit_artifact_runner(sink=FileArtifactSink(...))` (or any
    other `ArtifactSink` implementation).
    """

    del state
    return {
        "result": {
            "name": str(params["name"]),
            "type": str(params["artifact_type"]),
            "path": "<unpersisted>",
        }
    }


def default_runners() -> dict[NodeKind, NodeRunner]:
    """Echo / placeholder runners for every registry kind that has one.

    Compiler-handled kinds (`parallel_map`, `branch`, `wait_for_event`)
    aren't here — they're handled directly by the compiler.

    For any meaningful production use, override the entries that matter
    via the `runners=` arg to `LangGraphExecutor`.
    """

    return {
        NodeKind.LLM_CALL: run_llm_call,
        NodeKind.TOOL_CALL: run_tool_call,
        NodeKind.REDUCE: run_reduce,
        NodeKind.SPAWN_SUBAGENT: run_spawn_subagent,
        NodeKind.EMIT_ARTIFACT: run_emit_artifact,
    }
