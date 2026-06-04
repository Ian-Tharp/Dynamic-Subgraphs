"""Supervisor — the public entry point that runs one prompt end-to-end."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.models import GraphSpec
from app.policy import ExecutionPolicy
from app.recording import Recorder, RunRecord
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
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._recorder = recorder
        self._policy = policy or ExecutionPolicy()
        self._graph = build_supervisor_graph(
            planner=planner,
            executor=executor,
            recorder=recorder,
            policy=self._policy,
        ).compile()

    def run(self, prompt: str, *, run_id: str) -> SupervisorResult:
        final_state = self._graph.invoke({"prompt": prompt, "run_id": run_id})
        return SupervisorResult(
            run_id=run_id,
            status=final_state.get("status", "unknown"),
            response=final_state.get("response", ""),
            validated_spec=final_state.get("validated_spec"),
            result=final_state.get("result"),
            record=final_state.get("record"),
            errors=list(final_state.get("errors", [])),
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
            # Intentionally best-effort: a chain that ran shouldn't be
            # invalidated because persistence failed. The per-iteration
            # records are already on disk.
            with contextlib.suppress(Exception):
                self._recorder.record_chain(
                    outcome,
                    original_prompt=prompt,
                    overwrite=True,
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

        effective_new_run_id = new_run_id or self._default_replay_run_id(run_id)

        try:
            spec = self._recorder.load_validated_spec(run_id)
        except Exception as exc:
            return SupervisorResult(
                run_id=effective_new_run_id,
                status="replay_failed",
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

        try:
            compiled = self._executor.compile(spec)
        except Exception as exc:
            return SupervisorResult(
                run_id=effective_new_run_id,
                status="replay_failed",
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
        except Exception as exc:
            return SupervisorResult(
                run_id=effective_new_run_id,
                status="replay_failed",
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
            status = "paused"
            payloads = result.interrupt_payloads
            event_types = [
                str(p.get("event_type")) if isinstance(p, dict) else str(p)
                for p in payloads
            ]
            label = ", ".join(event_types) if event_types else "an unspecified event"
            response = (
                f"Replay of {run_id!r} as {effective_new_run_id!r} paused "
                f"waiting for {label}. "
                f"Call supervisor.resume(run_id, event=...) to continue."
            )
        elif result.ok:
            values = result.state.get("values", {})
            preview_keys = sorted(values.keys())[:6]
            status = "ok"
            response = (
                f"Replayed {run_id!r} as {effective_new_run_id!r}. "
                f"Produced {len(values)} output value(s): "
                f"{', '.join(preview_keys) if preview_keys else '(none)'}."
            )
        else:
            status = "execution_failed"
            response = (
                f"Replay of {run_id!r} executed but reported failure: "
                f"{result.error or 'unknown error'}"
            )

        if errors and status == "ok":
            status = "record_failed"
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
                status="resume_failed",
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
                status="resume_failed",
                response=f"Resume halted at compile: {exc}",
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
                status="resume_failed",
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
            status = "paused"
            payloads = result.interrupt_payloads
            event_types = [
                str(p.get("event_type")) if isinstance(p, dict) else str(p)
                for p in payloads
            ]
            label = ", ".join(event_types) if event_types else "an unspecified event"
            response = (
                f"Run paused again waiting for {label}. "
                f"Call supervisor.resume(run_id, event=...) to continue."
            )
        elif result.ok:
            status = "ok"
            values = result.state.get("values", {})
            preview_keys = sorted(values.keys())[:6]
            response = (
                f"Run resumed and completed. "
                f"Produced {len(values)} output value(s): "
                f"{', '.join(preview_keys) if preview_keys else '(none)'}."
            )
        else:
            status = "execution_failed"
            response = f"Resumed run failed: {result.error or 'unknown error'}"

        # Record-step failures don't change the underlying status, but they
        # surface in the errors list and influence the response.
        if errors and status == "ok":
            status = "record_failed"
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


def _iteration_run_id(chain_id: str, iteration: int) -> str:
    return f"{chain_id}_iter_{iteration}"
