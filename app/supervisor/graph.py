"""Build the supervisor StateGraph.

Topology:

    START -> receive -> plan -> validate -> execute -> record -> respond -> END

Plan/validate/execute each catch their *known* failure exceptions (planner
errors, RegistryValidationError, GraphCompilationError) and set `status` to
the matching failure code. A conditional edge after each of those stages
routes to `respond` on failure, skipping the rest of the pipeline. Recording
failures don't short-circuit — `respond` runs either way, so the user always
gets a response. Inner-graph execution failures (`ok=False`) are *not*
treated as supervisor failures: the inner result is recorded normally and
the supervisor reports `execution_failed`.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.compiler import GraphCompilationError
from app.recording import Recorder
from app.registry import RegistryValidationError, validate_graph_spec
from app.runtime import GraphExecutor
from app.supervisor.planner import Planner
from app.supervisor.state import SupervisorState

_PLAN_FAILED = "plan_failed"
_VALIDATION_FAILED = "validation_failed"
_COMPILE_FAILED = "compile_failed"


def build_supervisor_graph(
    *,
    planner: Planner,
    executor: GraphExecutor,
    recorder: Recorder,
) -> StateGraph:
    """Wire the static supervisor StateGraph with injected dependencies."""

    graph = StateGraph(SupervisorState)
    graph.add_node("receive", _make_receive_node())
    graph.add_node("plan", _make_plan_node(planner))
    graph.add_node("validate", _make_validate_node())
    graph.add_node("execute", _make_execute_node(executor))
    graph.add_node("record", _make_record_node(recorder))
    graph.add_node("respond", _make_respond_node())

    graph.add_edge(START, "receive")
    graph.add_edge("receive", "plan")
    graph.add_conditional_edges("plan", _route_after_plan, ["validate", "respond"])
    graph.add_conditional_edges(
        "validate", _route_after_validate, ["execute", "respond"]
    )
    graph.add_conditional_edges("execute", _route_after_execute, ["record", "respond"])
    graph.add_edge("record", "respond")
    graph.add_edge("respond", END)

    return graph


# ---------- routers ----------


def _route_after_plan(state: SupervisorState) -> str:
    return "respond" if state.get("status") == _PLAN_FAILED else "validate"


def _route_after_validate(state: SupervisorState) -> str:
    return "respond" if state.get("status") == _VALIDATION_FAILED else "execute"


def _route_after_execute(state: SupervisorState) -> str:
    # Inner execution failures (ok=False) still need recording.
    return "respond" if state.get("status") == _COMPILE_FAILED else "record"


# ---------- nodes ----------


def _make_receive_node():
    def receive(state: SupervisorState) -> SupervisorState:
        del state
        return {"status": "pending"}

    return receive


def _make_plan_node(planner: Planner):
    def plan(state: SupervisorState) -> SupervisorState:
        try:
            spec = planner(state["prompt"])
        except Exception as exc:
            return {
                "status": _PLAN_FAILED,
                "errors": [
                    {
                        "stage": "plan",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            }
        return {"spec": spec}

    return plan


def _make_validate_node():
    def validate(state: SupervisorState) -> SupervisorState:
        try:
            validated = validate_graph_spec(state["spec"])
        except RegistryValidationError as exc:
            return {
                "status": _VALIDATION_FAILED,
                "errors": [
                    {
                        "stage": "validate",
                        "type": "RegistryValidationError",
                        "message": str(exc),
                        "issues": [
                            {
                                "code": issue.code,
                                "message": issue.message,
                                "node_id": issue.node_id,
                                "field": issue.field,
                            }
                            for issue in exc.issues
                        ],
                    }
                ],
            }
        return {"validated_spec": validated}

    return validate


def _make_execute_node(executor: GraphExecutor):
    def execute(state: SupervisorState) -> SupervisorState:
        try:
            compiled = executor.compile(state["validated_spec"])
        except GraphCompilationError as exc:
            return {
                "status": _COMPILE_FAILED,
                "errors": [
                    {
                        "stage": "compile",
                        "type": "GraphCompilationError",
                        "message": str(exc),
                    }
                ],
            }

        result = executor.execute(compiled, run_id=state["run_id"])
        if result.paused:
            status = "paused"
        elif result.ok:
            status = "ok"
        else:
            status = "execution_failed"
        return {"result": result, "status": status}

    return execute


def _make_record_node(recorder: Recorder):
    def record(state: SupervisorState) -> SupervisorState:
        if state.get("validated_spec") is None or state.get("result") is None:
            return {}

        try:
            rec = recorder.record(
                spec=state["validated_spec"],
                result=state["result"],
                prompt=state.get("prompt"),
            )
        except Exception as exc:
            return {
                "status": "record_failed",
                "errors": [
                    {
                        "stage": "record",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            }
        return {"record": rec}

    return record


def _make_respond_node():
    def respond(state: SupervisorState) -> SupervisorState:
        status = state.get("status", "unknown")

        if status == "ok":
            values = state["result"].state.get("values", {})
            preview_keys = sorted(values.keys())[:6]
            response = (
                f"Run completed successfully. Produced {len(values)} output value(s): "
                f"{', '.join(preview_keys) if preview_keys else '(none)'}."
            )
        elif status == "paused":
            payloads = state["result"].interrupt_payloads
            event_types = [
                str(p.get("event_type")) if isinstance(p, dict) else str(p)
                for p in payloads
            ]
            label = ", ".join(event_types) if event_types else "an unspecified event"
            response = (
                f"Run paused waiting for {label}. "
                f"Call supervisor.resume(run_id, event=...) to continue."
            )
        elif status in {_PLAN_FAILED, _VALIDATION_FAILED, _COMPILE_FAILED}:
            last_error = (state.get("errors") or [{}])[-1]
            response = (
                f"Run halted at stage '{last_error.get('stage', '?')}': "
                f"{last_error.get('message', 'unknown error')}"
            )
        elif status == "execution_failed":
            inner_error = state["result"].error or "unknown inner error"
            response = f"Run executed but reported failure: {inner_error}"
        elif status == "record_failed":
            last_error = (state.get("errors") or [{}])[-1]
            response = (
                "Run completed but could not be persisted: "
                f"{last_error.get('message', 'unknown record error')}"
            )
        else:
            response = f"Run finished with unrecognized status: {status}"

        return {"response": response}

    return respond
