"""Build the supervisor StateGraph.

Topology:

    START -> receive -> plan -> validate -> execute -> record -> respond -> END

Plan/validate/execute each catch their failure exceptions (planner errors,
RegistryValidationError, GraphCompilationError, unexpected executor errors)
and set `status` to the matching failure code. A conditional edge after plan
and validate routes failures straight to `record` — every attempt, success or
failure, leaves a record (pre-execution failures via the recorder's
`record_failure` path). Recording failures don't short-circuit — `respond`
runs either way, so the user always gets a response — and never mask the
run's own outcome (only an OK run downgrades to `record_failed`).
Inner-graph execution failures (`ok=False`) are *not* treated as supervisor
failures: the inner result is recorded normally and the supervisor reports
`execution_failed`, with the node-level errors surfaced on `errors`.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.compiler import GraphCompilationError
from app.policy import ExecutionPolicy
from app.recording import Recorder
from app.registry import RegistryValidationError, validate_graph_spec
from app.runtime import GraphExecutor
from app.supervisor.planner import Planner
from app.supervisor.responses import render_interrupt_label, render_value_summary
from app.supervisor.state import RunStatus, SupervisorState

# Local aliases sourced from the canonical RunStatus vocabulary (state.py), so the
# status strings have one definition.
_PLAN_FAILED = RunStatus.PLAN_FAILED
_VALIDATION_FAILED = RunStatus.VALIDATION_FAILED
_COMPILE_FAILED = RunStatus.COMPILE_FAILED
# Transient: validation failed but a repair attempt remains. Always routes
# straight back to `plan`, so it is never a final status.
_PLAN_REPAIR_NEEDED = RunStatus.PLAN_REPAIR_NEEDED

# Validation issues a re-plan can't fix (a code/schema bug, not a plan the
# planner can correct) — never retry when *all* issues are of these kinds.
_NON_REPAIRABLE = frozenset({"unsupported_schema"})


def _issues_repairable(issues: list[dict[str, Any]]) -> bool:
    """True if at least one issue is something a re-plan could plausibly fix."""
    codes = {i.get("code") for i in issues}
    return bool(codes) and not codes.issubset(_NON_REPAIRABLE)


def _build_repair_prompt(
    original_prompt: str,
    issues: list[dict[str, Any]],
    policy: ExecutionPolicy,
    rejected_spec: Any | None,
) -> str:
    """Augment the prompt with the validator's findings + the host limits.

    The planner keeps its single-string interface; repair is just a richer
    prompt: the concrete issues to fix, the host ceilings to stay within, and
    the rejected plan so the model can correct it directly.
    """
    lines = [
        original_prompt,
        "",
        "Your previous GraphSpec was REJECTED by the validator. Produce a "
        "corrected plan that fixes ALL of these issues:",
    ]
    for issue in issues:
        where = f" (node: {issue['node_id']})" if issue.get("node_id") else ""
        lines.append(f"- [{issue.get('code')}] {issue.get('message')}{where}")
    lines += [
        "",
        "You MUST stay within these host limits (you cannot exceed them):",
        f"- at most {policy.max_nodes} nodes",
        f"- at most {policy.max_llm_calls} LLM calls",
        f"- nesting depth at most {policy.max_depth}",
    ]
    if rejected_spec is not None:
        lines += ["", "The rejected plan was:", rejected_spec.model_dump_json()]
    return "\n".join(lines)


def build_supervisor_graph(
    *,
    planner: Planner,
    executor: GraphExecutor,
    recorder: Recorder,
    policy: ExecutionPolicy | None = None,
    max_plan_attempts: int = 1,
) -> StateGraph:
    """Wire the static supervisor StateGraph with injected dependencies.

    `policy` is the host-owned `ExecutionPolicy` the validate step enforces
    budgets against (defaults to a permissive-but-bounded `ExecutionPolicy()`).

    `max_plan_attempts` bounds the plan-repair loop: when a plan is rejected by
    the validator for a *repairable* reason and attempts remain, the validator's
    issues + the host limits are fed back into a re-plan (`validate -> plan`).
    `1` (the default here) disables repair — block and report on first failure.
    """

    graph = StateGraph(SupervisorState)
    graph.add_node("receive", _make_receive_node())
    graph.add_node("plan", _make_plan_node(planner, policy))
    graph.add_node("validate", _make_validate_node(policy, max_plan_attempts))
    graph.add_node("execute", _make_execute_node(executor))
    graph.add_node("record", _make_record_node(recorder))
    graph.add_node("respond", _make_respond_node())

    graph.add_edge(START, "receive")
    graph.add_edge("receive", "plan")
    # EVERY terminal outcome routes through `record` — success or failure —
    # so failed attempts (plan/validate/compile) leave a record too, per the
    # recorder's "every attempt produces a directory" contract.
    graph.add_conditional_edges("plan", _route_after_plan, ["validate", "record"])
    graph.add_conditional_edges(
        "validate", _route_after_validate, ["execute", "plan", "record"]
    )
    graph.add_edge("execute", "record")
    graph.add_edge("record", "respond")
    graph.add_edge("respond", END)

    return graph


# ---------- routers ----------


def _route_after_plan(state: SupervisorState) -> str:
    return "record" if state.get("status") == _PLAN_FAILED else "validate"


def _route_after_validate(state: SupervisorState) -> str:
    status = state.get("status")
    if status == _PLAN_REPAIR_NEEDED:
        return "plan"  # the validator decided a repair attempt remains
    if status == _VALIDATION_FAILED:
        return "record"
    return "execute"


# ---------- nodes ----------


def _make_receive_node():
    def receive(state: SupervisorState) -> SupervisorState:
        del state
        return {"status": RunStatus.PENDING}

    return receive


def _make_plan_node(planner: Planner, policy: ExecutionPolicy | None = None):
    effective_policy = policy or ExecutionPolicy()

    def plan(state: SupervisorState) -> SupervisorState:
        attempt = state.get("plan_attempts", 0) + 1
        # On a repair pass the validator stashed the issues; feed them (plus the
        # host limits and the rejected plan) back to the planner.
        issues = state.get("last_validation_issues")
        prompt = state["prompt"]
        if issues:
            prompt = _build_repair_prompt(
                state["prompt"],
                issues,
                effective_policy,
                state.get("last_rejected_spec"),
            )
        try:
            spec = planner(prompt)
        except Exception as exc:
            return {
                "status": _PLAN_FAILED,
                "plan_attempts": attempt,
                "errors": [
                    {
                        "stage": "plan",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            }
        # Clear the transient repair context; status resets so validate re-runs.
        return {
            "spec": spec,
            "plan_attempts": attempt,
            "status": RunStatus.PENDING,
            "last_validation_issues": None,
        }

    return plan


def _make_validate_node(
    policy: ExecutionPolicy | None = None, max_plan_attempts: int = 1
):
    def validate(state: SupervisorState) -> SupervisorState:
        try:
            validated = validate_graph_spec(state["spec"], policy=policy)
        except RegistryValidationError as exc:
            issues = [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "node_id": issue.node_id,
                    "field": issue.field,
                }
                for issue in exc.issues
            ]
            attempts = state.get("plan_attempts", 1)
            if attempts < max_plan_attempts and _issues_repairable(issues):
                # A repair attempt remains: stash the issues for the re-plan and
                # do NOT write to `errors` — this isn't a terminal failure.
                return {
                    "status": _PLAN_REPAIR_NEEDED,
                    "last_validation_issues": issues,
                    "last_rejected_spec": state.get("spec"),
                }
            return {
                "status": _VALIDATION_FAILED,
                "errors": [
                    {
                        "stage": "validate",
                        "type": "RegistryValidationError",
                        "message": str(exc),
                        "issues": issues,
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

        try:
            result = executor.execute(compiled, run_id=state["run_id"])
        except Exception as exc:
            # Executor infrastructure can raise outside a node runner (e.g. a
            # checkpointer serialization failure, LangGraph's recursion rail).
            # Contain it here — the supervisor's contract is "failures don't
            # raise" — and surface it through the errors channel.
            return {
                "status": RunStatus.EXECUTION_FAILED,
                "errors": [
                    {
                        "stage": "execute",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            }
        if result.paused:
            status: str = RunStatus.PAUSED
        elif result.ok:
            status = RunStatus.OK
        else:
            status = RunStatus.EXECUTION_FAILED
        update: SupervisorState = {"result": result, "status": status}
        # Surface node-level failures on the supervisor's errors channel — the
        # documented place callers read structured errors from. They were
        # previously visible only inside the inner state envelope (and thus
        # invisible to `SupervisorResult.errors` / the SDK's `result.errors`).
        inner_errors = [
            {
                "stage": "execute",
                "type": str(entry.get("type", "Error")),
                "message": str(entry.get("message", "")),
                "node_id": entry.get("node_id"),
            }
            for entry in (result.state or {}).get("errors", []) or []
            if isinstance(entry, dict)
        ]
        if inner_errors:
            update["errors"] = inner_errors
        return update

    return execute


def _make_record_node(recorder: Recorder):
    def record(state: SupervisorState) -> SupervisorState:
        # A run that produced an ExecutionResult records normally. A run that
        # failed earlier (plan/validate/compile, or the executor raised) still
        # leaves a failure record — those are exactly the attempts you most
        # want evidence for. Recording problems never mask the run's own
        # outcome: only a run that actually succeeded is downgraded to
        # RECORD_FAILED; a failed run keeps its failure status and the record
        # error is appended to `errors`.
        if state.get("validated_spec") is not None and state.get("result") is not None:
            try:
                rec = recorder.record(
                    spec=state["validated_spec"],
                    result=state["result"],
                    prompt=state.get("prompt"),
                )
            except Exception as exc:
                update: SupervisorState = {
                    "errors": [
                        {
                            "stage": "record",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                }
                if state.get("status") == RunStatus.OK:
                    update["status"] = RunStatus.RECORD_FAILED
                return update
            return {"record": rec}

        # Pre-execution failure: persist what exists (prompt, errors, the
        # rejected spec). Feature-detected so custom Recorder implementations
        # without `record_failure` keep working; failures here never override
        # the run's own failure status.
        record_failure = getattr(recorder, "record_failure", None)
        if record_failure is None:
            return {}
        try:
            rec = record_failure(
                run_id=state["run_id"],
                status=str(state.get("status", "unknown")),
                prompt=state.get("prompt"),
                errors=list(state.get("errors", []) or []),
                rejected_spec=state.get("last_rejected_spec") or state.get("spec"),
                overwrite=True,
            )
        except Exception as exc:
            return {
                "errors": [
                    {
                        "stage": "record",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            }
        return {"record": rec} if rec is not None else {}

    return record


def _make_respond_node():
    def respond(state: SupervisorState) -> SupervisorState:
        status = state.get("status", "unknown")

        if status == RunStatus.OK:
            values = state["result"].state.get("values", {})
            response = (
                f"Run completed successfully. Produced {render_value_summary(values)}."
            )
        elif status == RunStatus.PAUSED:
            label = render_interrupt_label(state["result"].interrupt_payloads)
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
        elif status == RunStatus.EXECUTION_FAILED:
            result = state.get("result")
            if result is not None:
                inner_error = result.error or "unknown inner error"
                response = f"Run executed but reported failure: {inner_error}"
            else:
                # The executor itself raised — there is no ExecutionResult.
                last_error = (state.get("errors") or [{}])[-1]
                response = (
                    "Run execution failed: "
                    f"{last_error.get('message', 'unknown executor error')}"
                )
        elif status == RunStatus.RECORD_FAILED:
            last_error = (state.get("errors") or [{}])[-1]
            response = (
                "Run completed but could not be persisted: "
                f"{last_error.get('message', 'unknown record error')}"
            )
        else:
            response = f"Run finished with unrecognized status: {status}"

        return {"response": response}

    return respond
