"""Supervisor — the public entry point that runs one prompt end-to-end."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.governance import (
    PlanGateReport,
    ProvenanceReceipt,
    build_receipt,
    check_plan_policy,
    plan_hash,
    policy_decision_events,
    policy_hash,
)
from app.models import GraphSpec
from app.policy import ExecutionPolicy
from app.recording import Recorder, RunRecord
from app.registry import RegistryValidationError, validate_graph_spec
from app.runtime import ExecutionResult, GraphExecutor
from app.supervisor.graph import build_supervisor_graph
from app.supervisor.iteration import (
    IterationContext,
    IterationDecider,
    IterationDecision,
    IterationStep,
    IterativeSupervisorResult,
    StatusIterationDecider,
    build_replan_prompt,
)
from app.supervisor.planner import Planner
from app.supervisor.responses import render_interrupt_label, render_value_summary
from app.supervisor.state import RunStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupervisorResult:
    """Receipt returned from `Supervisor.run` or `Supervisor.resume`.

    `status` is always one of the values documented on `SupervisorState`.
    The remaining fields populate as far as the pipeline progressed:
        - `validated_spec` is None until validate succeeded.
        - `result` is None until execute succeeded (compile_failed leaves it None).
        - `record` is None when validation/compile failed or recording itself failed.
    """

    run_id: str
    status: str
    response: str
    validated_spec: GraphSpec | None
    result: ExecutionResult | None
    record: RunRecord | None
    errors: list[dict]
    # How many times the planner ran (1 unless the plan-repair loop re-planned
    # after a recoverable validation failure).
    plan_attempts: int = 1
    # Plan-level policy gate outcome for the last checked plan (None if the
    # run never produced a registry-valid plan).
    gate_report: PlanGateReport | None = None
    # Tamper-evident provenance receipt (see app.governance.receipt). Built for
    # every run()/replay() outcome, including refused and failed runs.
    receipt: ProvenanceReceipt | None = None


class Supervisor:
    """Compile the supervisor graph once; expose `run(prompt)` per call.

    For paused runs (graphs containing `wait_for_event`), call `resume(...)`
    with the same `run_id` and the event payload. Resume requires the executor
    to have been constructed with a `checkpointer` so the paused state was
    actually persisted.
    """

    def __init__(
        self,
        *,
        planner: Planner,
        executor: GraphExecutor,
        recorder: Recorder,
        policy: ExecutionPolicy | None = None,
        max_plan_attempts: int = 1,
        run_models: Iterable[str] = (),
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._recorder = recorder
        self._policy = policy or ExecutionPolicy()
        self._max_plan_attempts = max(1, max_plan_attempts)
        # Model names this supervisor's runs are configured with (planner,
        # worker, ...): checked against policy.allowed_models and listed on
        # every receipt.
        self._run_models = tuple(sorted(set(run_models)))
        self._graph = build_supervisor_graph(
            planner=planner,
            executor=executor,
            recorder=recorder,
            policy=self._policy,
            max_plan_attempts=self._max_plan_attempts,
            run_models=self._run_models,
        ).compile()

    @property
    def policy(self) -> ExecutionPolicy:
        return self._policy

    def run(self, prompt: str, *, run_id: str) -> SupervisorResult:
        started_at = datetime.now(UTC)
        final_state = self._graph.invoke({"prompt": prompt, "run_id": run_id})
        result = SupervisorResult(
            run_id=run_id,
            status=final_state.get("status", "unknown"),
            response=final_state.get("response", ""),
            validated_spec=final_state.get("validated_spec"),
            result=final_state.get("result"),
            record=final_state.get("record"),
            errors=list(final_state.get("errors", [])),
            plan_attempts=final_state.get("plan_attempts", 1),
            gate_report=final_state.get("gate_report"),
        )
        refused = final_state.get("last_rejected_spec")
        return self._with_receipt(
            result,
            started_at=started_at,
            refused_spec=refused if result.validated_spec is None else None,
        )

    def _with_receipt(
        self,
        result: SupervisorResult,
        *,
        started_at: datetime,
        refused_spec: GraphSpec | None = None,
    ) -> SupervisorResult:
        """Attach a sealed receipt and persist it next to the run's record.

        Never raises: a receipt problem is logged and the run's outcome stands.
        """
        try:
            spec = result.validated_spec or refused_spec
            exec_result = result.result
            counters = (
                (exec_result.state or {}).get("counters", {}) if exec_result else {}
            )
            llm_calls = (
                int(counters.get("llm_calls_consumed", 0)) if exec_result else None
            )
            declared = set(self._run_models)
            if spec is not None:
                from app.governance.plan_gate import node_models

                for node in spec.nodes:
                    declared.update(node_models(node))
            receipt = build_receipt(
                run_id=result.run_id,
                status=str(result.status),
                policy_hash=policy_hash(self._policy),
                started_at=started_at,
                finished_at=datetime.now(UTC),
                plan_hash=plan_hash(spec) if spec is not None else None,
                gate_report=result.gate_report,
                llm_calls=llm_calls,
                model_ids=declared,
                # Only a record that persisted something can back the receipt
                # (NullRecorder returns a record with no artifacts).
                recorder_run_id=(
                    result.record.run_id
                    if result.record is not None and result.record.artifacts
                    else None
                ),
                trace_events=exec_result.trace.events if exec_result else None,
            )
        except Exception:
            logger.warning(
                "failed to build receipt for %r", result.run_id, exc_info=True
            )
            return result
        self.persist_receipt(receipt)
        return _replace_result(result, receipt=receipt)

    def persist_receipt(self, receipt: ProvenanceReceipt) -> None:
        """Write ``receipt.json`` if the recorder supports it and the run was recorded."""
        writer = getattr(self._recorder, "record_receipt", None)
        if writer is None or receipt.recorder_run_id is None:
            return
        try:
            writer(receipt.recorder_run_id, receipt.model_dump(mode="json"))
        except Exception:
            logger.warning(
                "failed to persist receipt for %r", receipt.run_id, exc_info=True
            )

    def run_iteratively(
        self,
        prompt: str,
        *,
        run_id: str,
        max_iterations: int = 3,
        decider: IterationDecider | None = None,
        record_chain: bool = True,
    ) -> IterativeSupervisorResult:
        """Run bounded GraphSpecs in sequence until a typed decision stops.

        Each iteration is a normal supervisor run with its own run id and run
        directory. A custom `decider` can inspect the previous result and
        return `replan` with a `next_prompt`; if it omits `next_prompt`, the
        supervisor builds a compact replan prompt from the prior run record,
        outputs, errors, and decision gaps.

        When `record_chain=True` (default), the chain metadata is persisted
        to `runs/<run_id>/chain.json` alongside the per-iteration record
        directories (`runs/<run_id>_iter_N/`). Chain-record failures are
        caught and don't affect the returned result — the chain succeeded
        even if persisting its summary didn't.
        """

        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        effective_decider = decider or StatusIterationDecider()
        current_prompt = prompt
        steps: list[IterationStep] = []
        outcome: IterativeSupervisorResult | None = None

        for iteration in range(1, max_iterations + 1):
            iteration_run_id = _iteration_run_id(run_id, iteration)
            result = self.run(current_prompt, run_id=iteration_run_id)

            try:
                decision = effective_decider(
                    IterationContext(
                        original_prompt=prompt,
                        current_prompt=current_prompt,
                        iteration=iteration,
                        max_iterations=max_iterations,
                        result=result,
                        previous_steps=tuple(steps),
                    )
                )
            except Exception as exc:
                decision = IterationDecision(
                    action="fail",
                    reason=f"Iteration decider failed: {exc}",
                    gaps=[type(exc).__name__],
                )

            step = IterationStep(
                iteration=iteration,
                run_id=iteration_run_id,
                prompt=current_prompt,
                result=result,
                decision=decision,
            )
            steps.append(step)

            if decision.action == "stop":
                status = "ok" if decision.success_criteria_met else "stopped"
                outcome = IterativeSupervisorResult(
                    chain_id=run_id,
                    status=status,
                    response=decision.reason,
                    steps=tuple(steps),
                    final_result=result,
                    final_decision=decision,
                )
                break

            if decision.action == "ask_user":
                outcome = IterativeSupervisorResult(
                    chain_id=run_id,
                    status="ask_user",
                    response=decision.question_to_user or decision.reason,
                    steps=tuple(steps),
                    final_result=result,
                    final_decision=decision,
                )
                break

            if decision.action == "fail":
                outcome = IterativeSupervisorResult(
                    chain_id=run_id,
                    status="failed",
                    response=decision.reason,
                    steps=tuple(steps),
                    final_result=result,
                    final_decision=decision,
                )
                break

            if iteration == max_iterations:
                outcome = IterativeSupervisorResult(
                    chain_id=run_id,
                    status="max_iterations",
                    response=(
                        f"Stopped after {max_iterations} iteration(s): "
                        f"{decision.reason}"
                    ),
                    steps=tuple(steps),
                    final_result=result,
                    final_decision=decision,
                )
                break

            current_prompt = decision.next_prompt or build_replan_prompt(
                original_prompt=prompt,
                current_prompt=current_prompt,
                result=result,
                decision=decision,
            )

        if outcome is None:
            # Unreachable — validated max_iterations and exhaustive branches above.
            raise RuntimeError("iterative supervisor exited without a result")

        if record_chain and hasattr(self._recorder, "record_chain"):
            # Intentionally best-effort: a chain that ran shouldn't be invalidated
            # because persistence failed (the per-iteration records are already on
            # disk). Log a breadcrumb so a chronically failing recorder is visible.
            try:
                self._recorder.record_chain(
                    outcome,
                    original_prompt=prompt,
                    overwrite=True,
                )
            except Exception:
                logger.warning(
                    "failed to record chain %r", outcome.chain_id, exc_info=True
                )

        return outcome

    def replay(
        self,
        run_id: str,
        *,
        new_run_id: str | None = None,
    ) -> SupervisorResult:
        """Re-execute a previously-recorded spec as a fresh run.

        Replay loads the spec the recorder persisted on `run_id`, compiles
        it, executes it under a *new* run id (so the original recording is
        untouched), and persists the new run. Replay does **not** re-plan
        — it uses the original spec verbatim. The point is to study how
        the same shape behaves on a fresh execution (different LLM output,
        a runner code change, a different model, etc.).

        `new_run_id` defaults to `<run_id>_replay_<utc_iso>` so the
        original and replay runs are colocated under `runs/` for easy
        diffing.

        Replay does not inherit checkpointer state from the original run.
        If the spec contains `wait_for_event`, the replay will pause
        fresh — it does not auto-resume from where the original left off.
        """

        started_at = datetime.now(UTC)
        result = self._replay(run_id, new_run_id=new_run_id)
        return self._with_receipt(result, started_at=started_at)

    def _replay(
        self,
        run_id: str,
        *,
        new_run_id: str | None = None,
    ) -> SupervisorResult:
        effective_new_run_id = new_run_id or self._default_replay_run_id(run_id)

        try:
            spec = self._recorder.load_validated_spec(run_id)
        except Exception as exc:
            return SupervisorResult(
                run_id=effective_new_run_id,
                status=RunStatus.REPLAY_FAILED,
                response=(
                    f"Replay halted: could not load recorded spec for {run_id!r}: {exc}"
                ),
                validated_spec=None,
                result=None,
                record=None,
                errors=[
                    {
                        "stage": "replay_load_spec",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )

        # Re-validate against the CURRENT host policy: a spec recorded under a
        # looser policy must not replay if the host has since tightened limits
        # (the recorded budget is otherwise trusted verbatim). Idempotent when
        # the policy is unchanged. Re-stamps the granted budget.
        try:
            spec = validate_graph_spec(spec, policy=self._policy)
        except RegistryValidationError as exc:
            return SupervisorResult(
                run_id=effective_new_run_id,
                status=RunStatus.REPLAY_FAILED,
                response=(
                    f"Replay halted: recorded spec for {run_id!r} violates the "
                    f"current policy: {exc}"
                ),
                validated_spec=spec,
                result=None,
                record=None,
                errors=[
                    {
                        "stage": "replay_validate",
                        "type": "RegistryValidationError",
                        "message": str(exc),
                        "issues": [
                            {
                                "code": i.code,
                                "message": i.message,
                                "node_id": i.node_id,
                                "field": i.field,
                            }
                            for i in exc.issues
                        ],
                    }
                ],
            )

        # The plan gate applies to replays too: a recorded plan that the current
        # policy would refuse (tightened budget, new data-flow rule, catalog
        # mode) does not run again.
        gate_report = check_plan_policy(spec, self._policy, run_models=self._run_models)
        if not gate_report.allowed:
            return SupervisorResult(
                run_id=effective_new_run_id,
                status=RunStatus.POLICY_DENIED,
                response=(
                    f"Replay halted: recorded spec for {run_id!r} is refused by "
                    f"the current plan policy: "
                    + "; ".join(v.message for v in gate_report.violations)
                ),
                validated_spec=None,
                result=None,
                record=None,
                errors=[
                    {
                        "stage": "replay_policy_gate",
                        "type": "PlanPolicyViolation",
                        "message": "; ".join(v.message for v in gate_report.violations),
                        "issues": [v.as_dict() for v in gate_report.violations],
                    }
                ],
                gate_report=gate_report,
            )

        try:
            compiled = self._executor.compile(spec)
        except Exception as exc:
            return SupervisorResult(
                run_id=effective_new_run_id,
                status=RunStatus.REPLAY_FAILED,
                response=f"Replay halted at compile: {exc}",
                validated_spec=spec,
                result=None,
                record=None,
                errors=[
                    {
                        "stage": "replay_compile",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )

        try:
            # Seed `replay_of` (the ORIGINAL run id) so any spawn_subgraph node
            # pins its child to the originally recorded child spec instead of
            # re-planning — making nested replay reproduce the same shape.
            result = self._executor.execute(
                compiled,
                run_id=effective_new_run_id,
                initial_metadata={"replay_of": run_id},
            )
            result.trace.events[:0] = policy_decision_events(gate_report)
        except Exception as exc:
            return SupervisorResult(
                run_id=effective_new_run_id,
                status=RunStatus.REPLAY_FAILED,
                response=f"Replay halted at execute: {exc}",
                validated_spec=spec,
                result=None,
                record=None,
                errors=[
                    {
                        "stage": "replay_execute",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )

        errors: list[dict] = []
        record: RunRecord | None = None
        try:
            record = self._recorder.record(spec=spec, result=result, prompt=None)
        except Exception as exc:
            errors.append(
                {
                    "stage": "replay_record",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )

        if result.paused:
            status = RunStatus.PAUSED
            label = render_interrupt_label(result.interrupt_payloads)
            response = (
                f"Replay of {run_id!r} as {effective_new_run_id!r} paused "
                f"waiting for {label}. "
                f"Call supervisor.resume(run_id, event=...) to continue."
            )
        elif result.ok:
            status = RunStatus.OK
            response = (
                f"Replayed {run_id!r} as {effective_new_run_id!r}. "
                f"Produced {render_value_summary(result.state.get('values', {}))}."
            )
        else:
            status = RunStatus.EXECUTION_FAILED
            response = (
                f"Replay of {run_id!r} executed but reported failure: "
                f"{result.error or 'unknown error'}"
            )

        if errors and status == RunStatus.OK:
            status = RunStatus.RECORD_FAILED
            response = (
                f"Replay of {run_id!r} completed but the new record could "
                f"not be persisted: {errors[-1]['message']}"
            )

        return SupervisorResult(
            run_id=effective_new_run_id,
            status=status,
            response=response,
            validated_spec=spec,
            result=result,
            record=record,
            errors=errors,
            gate_report=gate_report,
        )

    @staticmethod
    def _default_replay_run_id(original_run_id: str) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{original_run_id}_replay_{timestamp}"

    def resume(self, run_id: str, *, event: Any) -> SupervisorResult:
        """Resume a previously-paused run by feeding it the awaited event.

        Loads the spec the recorder persisted on the original run, compiles
        it (state is restored from the executor's checkpointer), resumes
        execution with the event payload, and records the updated state
        (overwriting the previous run directory).
        """

        errors: list[dict] = []

        try:
            spec = self._recorder.load_validated_spec(run_id)
        except Exception as exc:
            errors.append(
                {
                    "stage": "resume_load_spec",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return SupervisorResult(
                run_id=run_id,
                status=RunStatus.RESUME_FAILED,
                response=(
                    f"Resume halted: could not load recorded spec for {run_id!r}: {exc}"
                ),
                validated_spec=None,
                result=None,
                record=None,
                errors=errors,
            )

        try:
            compiled = self._executor.compile(spec)
        except Exception as exc:
            errors.append(
                {
                    "stage": "resume_compile",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return SupervisorResult(
                run_id=run_id,
                status=RunStatus.RESUME_FAILED,
                response=f"Resume halted at compile: {exc}",
                validated_spec=spec,
                result=None,
                record=None,
                errors=errors,
            )

        # Refuse to resume a run that is not actually paused. Without this
        # check, "resuming" a completed run re-applies the event and re-executes
        # the post-wait portion of the graph — re-spending LLM calls, re-firing
        # side effects, and overwriting the recorded trace. Feature-detected so
        # custom executors without the probe keep the old (permissive) behavior.
        pending_probe = getattr(self._executor, "has_pending_interrupt", None)
        if pending_probe is not None and not pending_probe(compiled, run_id=run_id):
            errors.append(
                {
                    "stage": "resume_check",
                    "type": "NotResumable",
                    "message": (
                        f"run {run_id!r} has no pending pause — it either "
                        "completed, was already resumed, or its checkpointed "
                        "state is gone"
                    ),
                }
            )
            return SupervisorResult(
                run_id=run_id,
                status=RunStatus.RESUME_FAILED,
                response=(
                    f"Resume refused: run {run_id!r} is not paused (it either "
                    "completed, was already resumed, or its checkpointed state "
                    "is no longer available)."
                ),
                validated_spec=spec,
                result=None,
                record=None,
                errors=errors,
            )

        try:
            result = self._executor.resume(compiled, run_id=run_id, event=event)
        except Exception as exc:
            errors.append(
                {
                    "stage": "resume_execute",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return SupervisorResult(
                run_id=run_id,
                status=RunStatus.RESUME_FAILED,
                response=f"Resume halted at execute: {exc}",
                validated_spec=spec,
                result=None,
                record=None,
                errors=errors,
            )

        record: RunRecord | None = None
        try:
            record = self._recorder.record(
                spec=spec,
                result=result,
                prompt=None,
                overwrite=True,
            )
        except Exception as exc:
            errors.append(
                {
                    "stage": "resume_record",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )

        if result.paused:
            status = RunStatus.PAUSED
            label = render_interrupt_label(result.interrupt_payloads)
            response = (
                f"Run paused again waiting for {label}. "
                f"Call supervisor.resume(run_id, event=...) to continue."
            )
        elif result.ok:
            status = RunStatus.OK
            response = (
                f"Run resumed and completed. "
                f"Produced {render_value_summary(result.state.get('values', {}))}."
            )
        else:
            status = RunStatus.EXECUTION_FAILED
            response = f"Resumed run failed: {result.error or 'unknown error'}"

        # Record-step failures don't change the underlying status, but they
        # surface in the errors list and influence the response.
        if errors and status == RunStatus.OK:
            status = RunStatus.RECORD_FAILED
            response = (
                "Run resumed but the new state could not be persisted: "
                f"{errors[-1]['message']}"
            )

        return SupervisorResult(
            run_id=run_id,
            status=status,
            response=response,
            validated_spec=spec,
            result=result,
            record=record,
            errors=errors,
        )


def _replace_result(result: SupervisorResult, **changes: Any) -> SupervisorResult:
    from dataclasses import replace

    return replace(result, **changes)


def _iteration_run_id(chain_id: str, iteration: int) -> str:
    return f"{chain_id}_iter_{iteration}"
