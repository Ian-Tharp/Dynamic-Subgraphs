"""Bounded meta-loop types for supervisor-level iterative runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

from app.supervisor.state import DEFER_STATUSES, RunStatus

if TYPE_CHECKING:
    from app.supervisor.supervisor import SupervisorResult


IterationAction = Literal["stop", "replan", "ask_user", "fail"]
_VALID_ACTIONS = {"stop", "replan", "ask_user", "fail"}


@dataclass(frozen=True)
class IterationDecision:
    """Typed control decision after one bounded supervisor run."""

    action: IterationAction
    reason: str
    success_criteria_met: bool = False
    gaps: Sequence[str] = field(default_factory=tuple)
    next_prompt: str | None = None
    question_to_user: str | None = None

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"Unknown iteration action {self.action!r}; "
                f"expected one of {sorted(_VALID_ACTIONS)}"
            )
        if not self.reason:
            raise ValueError("IterationDecision.reason must be non-empty")
        object.__setattr__(self, "gaps", tuple(self.gaps))

        if self.action == "ask_user" and not self.question_to_user:
            raise ValueError(
                "IterationDecision.question_to_user is required for ask_user"
            )


@dataclass(frozen=True)
class IterationContext:
    """Context handed to an iteration decider after each supervisor run."""

    original_prompt: str
    current_prompt: str
    iteration: int
    max_iterations: int
    result: SupervisorResult
    previous_steps: tuple[IterationStep, ...] = ()


class IterationDecider(Protocol):
    """Decides whether the supervisor should stop, replan, ask, or fail."""

    def __call__(self, context: IterationContext) -> IterationDecision: ...


@dataclass(frozen=True)
class IterationStep:
    """One completed bounded run in an iterative supervisor chain."""

    iteration: int
    run_id: str
    prompt: str
    result: SupervisorResult
    decision: IterationDecision


@dataclass(frozen=True)
class IterativeSupervisorResult:
    """Receipt returned by `Supervisor.run_iteratively`."""

    chain_id: str
    status: str
    response: str
    steps: tuple[IterationStep, ...]
    final_result: SupervisorResult | None
    final_decision: IterationDecision | None


class StatusIterationDecider:
    """Safe default decider.

    It stops after a successful run, asks the caller to resume paused runs, and
    fails closed for other statuses. Real adaptive behavior can be plugged in
    by passing a custom `IterationDecider` to `run_iteratively`.
    """

    def __call__(self, context: IterationContext) -> IterationDecision:
        result = context.result
        if result.status == RunStatus.OK:
            return IterationDecision(
                action="stop",
                reason="The bounded run completed successfully.",
                success_criteria_met=True,
            )

        if result.status == RunStatus.PAUSED:
            return IterationDecision(
                action="ask_user",
                reason="The bounded run paused waiting for external input.",
                question_to_user=result.response,
            )

        return IterationDecision(
            action="fail",
            reason=result.response or f"Run ended with status {result.status!r}.",
            gaps=[f"status={result.status}"],
        )


def build_replan_prompt(
    *,
    original_prompt: str,
    current_prompt: str,
    result: SupervisorResult,
    decision: IterationDecision,
) -> str:
    """Build compact planner context for the next bounded run."""

    gaps = "\n".join(f"- {gap}" for gap in decision.gaps) or "- (none supplied)"
    record_dir = str(result.record.directory) if result.record is not None else "(none)"
    output_preview = _render_output_preview(result)
    errors = _render_error_preview(result)

    return "\n".join(
        [
            "You are replanning a Dynamic Subgraphs run.",
            "",
            "Original user prompt:",
            original_prompt,
            "",
            "Previous planner prompt:",
            current_prompt,
            "",
            "Previous iteration result:",
            f"- run_id: {result.run_id}",
            f"- status: {result.status}",
            f"- response: {result.response}",
            f"- record_dir: {record_dir}",
            "",
            "Decision reason:",
            decision.reason,
            "",
            "Gaps to address:",
            gaps,
            "",
            "Previous output preview:",
            output_preview,
            "",
            "Previous errors:",
            errors,
            "",
            "Plan a revised bounded GraphSpec for the next attempt. "
            "Address the gaps above, reuse prior outputs only when they are "
            "relevant, and keep the graph small.",
        ]
    )


def _render_output_preview(result: SupervisorResult) -> str:
    if result.result is None:
        return "- (no execution result)"

    values = result.result.state.get("values", {})
    if not values:
        return "- (no values)"

    lines: list[str] = []
    for key in sorted(values.keys())[:8]:
        lines.append(f"- {key}: {_truncate(repr(values[key]))}")
    if len(values) > 8:
        lines.append(f"- ... {len(values) - 8} more value(s)")
    return "\n".join(lines)


def _render_error_preview(result: SupervisorResult) -> str:
    entries: list[dict] = []
    entries.extend(result.errors)
    if result.result is not None:
        entries.extend(result.result.state.get("errors", []) or [])

    if not entries:
        return "- (none)"

    lines: list[str] = []
    for entry in entries[:8]:
        stage = entry.get("stage") or entry.get("node_id") or "?"
        etype = entry.get("type", "Error")
        message = _truncate(str(entry.get("message", "")))
        lines.append(f"- {stage} ({etype}): {message}")
    if len(entries) > 8:
        lines.append(f"- ... {len(entries) - 8} more error(s)")
    return "\n".join(lines)


def _truncate(value: str, *, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


# ---------- LLM-backed decider ----------


JUDGE_SYSTEM_PROMPT = """\
You are the iteration judge for a Dynamic Subgraphs run. You evaluate
whether a bounded workflow's output addresses the user's original
prompt, and emit a structured `IterationDecision` telling the
supervisor what to do next.

Choose one of four actions:

- "stop": The output substantially addresses the user's prompt. The
  user could use it as-is, even if not perfect. Set
  `success_criteria_met=true`. Set `reason` to a one-sentence summary
  of what was accomplished.

- "replan": The output is missing essential content, has a factual
  error, or fails to address part of the prompt. Put concrete missing
  items in `gaps` — each gap a short specific phrase (e.g., "missing
  citations", "ignored the second part of the user's question",
  "explanation lacks concrete examples"). If you have a clear plan for
  the next attempt, supply `next_prompt`; otherwise the supervisor
  will build a context-rich one from your gaps.

- "ask_user": The user's prompt is GENUINELY ambiguous and you cannot
  proceed without clarification (multiple equally valid interpretations,
  not "I'm unsure how to improve"). Put the question in
  `question_to_user`.

- "fail": The run produced something unsalvageable AND retrying won't
  help (e.g., the user asked something impossible). Use sparingly.

Be strict but not nitpicky. The bar is "could the user use this?", not
"is this perfect?". Don't request replans for stylistic tweaks. If you
notice you're being asked to evaluate the LAST allowed iteration,
prefer `stop` with whatever was produced unless the run failed outright.

Note on truncation: long output values are truncated for display, but
the underlying value in `state.values` is what the user gets. Do NOT
treat "this excerpt looks truncated" as a gap — focus on what IS visible.
If the visible portion satisfies the criteria, choose `stop`. Only emit
gaps for substantive content issues you can actually see (missing
companies, wrong counts in what's shown, factual errors, missing
required elements that should appear early in the output).
"""


class _IterationDecisionPayload(BaseModel):
    """Closed-shape schema for structured-output LLM judge decisions.

    Mirrors the public `IterationDecision` dataclass. Kept private — the
    LlmIterationDecider converts to the public dataclass before returning.
    """

    action: Literal["stop", "replan", "ask_user", "fail"]
    reason: str = Field(min_length=1)
    success_criteria_met: bool = False
    gaps: list[str] = Field(default_factory=list)
    next_prompt: str | None = None
    question_to_user: str | None = None


class _StructuredJudgeModel(Protocol):
    """Anything that returns an `_IterationDecisionPayload` from `.invoke(messages)`."""

    def invoke(self, messages: list, /, **kwargs: Any) -> Any: ...


class LlmIterationDecider:
    """LLM-backed `IterationDecider` that judges output quality and decides next step.

    For obvious cases (paused / any *_failed status), defers to the fallback
    decider — no point spending tokens on decisions the framework already
    knows how to make. Real LLM judgement only fires on `ok` status and on
    `execution_failed` only if the caller explicitly asks for it.

    Catches every model failure mode (exception, wrong-shape return,
    invalid decision fields) into a closed-form `fail` decision so the
    meta-loop never crashes because of the judge.

    Args:
        structured_model: a chat model configured with
            `with_structured_output(_IterationDecisionPayload, method="function_calling")`.
            Must return an `_IterationDecisionPayload` from `.invoke(messages)`.
        success_criteria: optional explicit success rubric. When provided,
            included verbatim in the eval prompt. Otherwise the judge
            infers success from the user's original prompt.
        fallback: decider used for non-ok statuses. Defaults to
            `StatusIterationDecider`.
        judge_failed_runs: if True, the LLM also evaluates execution_failed
            runs (rare — usually the framework's classification is enough).
    """

    def __init__(
        self,
        structured_model: _StructuredJudgeModel,
        *,
        success_criteria: str | None = None,
        fallback: IterationDecider | None = None,
        judge_failed_runs: bool = False,
        value_render_limit: int = 4000,
    ) -> None:
        self._model = structured_model
        self._success_criteria = success_criteria
        self._fallback = fallback or StatusIterationDecider()
        self._judge_failed_runs = judge_failed_runs
        self._value_render_limit = value_render_limit

    def __call__(self, context: IterationContext) -> IterationDecision:
        status = context.result.status

        # Obvious cases: defer to the fallback decider. Don't spend tokens
        # on decisions the framework's classification already settled.
        if status in DEFER_STATUSES:
            return self._fallback(context)
        if status == RunStatus.EXECUTION_FAILED and not self._judge_failed_runs:
            return self._fallback(context)

        # Lazy import to avoid pulling langchain-core when the LLM decider
        # isn't used (matches the lazy-import pattern throughout runtime).
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=self._build_eval_prompt(context)),
        ]

        try:
            payload = self._model.invoke(messages)
        except Exception as exc:
            return IterationDecision(
                action="fail",
                reason=f"LlmIterationDecider model call failed: {exc}",
                gaps=[type(exc).__name__],
            )

        if not isinstance(payload, _IterationDecisionPayload):
            return IterationDecision(
                action="fail",
                reason=(
                    f"LlmIterationDecider expected _IterationDecisionPayload, "
                    f"got {type(payload).__name__}"
                ),
            )

        try:
            return IterationDecision(
                action=payload.action,
                reason=payload.reason,
                success_criteria_met=payload.success_criteria_met,
                gaps=tuple(payload.gaps),
                next_prompt=payload.next_prompt,
                question_to_user=payload.question_to_user,
            )
        except ValueError as exc:
            return IterationDecision(
                action="fail",
                reason=f"LlmIterationDecider produced invalid decision: {exc}",
            )

    def _build_eval_prompt(self, context: IterationContext) -> str:
        result = context.result
        parts: list[str] = [
            "User's original prompt:",
            context.original_prompt,
            "",
            f"Iteration: {context.iteration} of {context.max_iterations}",
        ]

        if self._success_criteria:
            parts.extend(
                [
                    "",
                    "Explicit success criteria:",
                    self._success_criteria,
                ]
            )

        parts.extend(
            [
                "",
                f"Run status: {result.status}",
                f"Run response: {result.response}",
            ]
        )

        values = (
            result.result.state.get("values", {}) if result.result is not None else {}
        )
        if values:
            parts.append("")
            parts.append("Outputs produced (state.values):")
            for key in sorted(values.keys()):
                # Use a generous limit per value — real LLM outputs are
                # routinely 1-4k chars per key, and judging "did the output
                # satisfy the criteria" requires seeing most of it.
                rendered = _truncate(repr(values[key]), limit=self._value_render_limit)
                parts.append(f"- {key}: {rendered}")
        else:
            parts.append("")
            parts.append("Outputs produced: (none)")

        error_entries: list[dict] = []
        error_entries.extend(result.errors)
        if result.result is not None:
            error_entries.extend(result.result.state.get("errors", []) or [])
        if error_entries:
            parts.append("")
            parts.append("Errors:")
            for entry in error_entries[:8]:
                stage = entry.get("stage") or entry.get("node_id") or "?"
                msg = _truncate(str(entry.get("message", "")))
                parts.append(f"- {stage}: {msg}")

        if context.previous_steps:
            parts.append("")
            parts.append(
                f"Previous iterations in this chain: {len(context.previous_steps)}"
            )
            for prior in context.previous_steps[-3:]:
                parts.append(
                    f"- iter {prior.iteration}: action={prior.decision.action}, "
                    f"reason={_truncate(prior.decision.reason, limit=200)}"
                )

        parts.extend(
            [
                "",
                "Emit your IterationDecision now. The bar is 'could the user "
                "use this?', not 'is this perfect?'.",
            ]
        )
        return "\n".join(parts)


def build_openai_iteration_decider(
    model: str = "gpt-5.4-nano",
    *,
    success_criteria: str | None = None,
    temperature: float | None = None,
    fallback: IterationDecider | None = None,
    judge_failed_runs: bool = False,
    value_render_limit: int = 4000,
) -> LlmIterationDecider:
    """Convenience factory: an `LlmIterationDecider` backed by ChatOpenAI.

    Requires `OPENAI_API_KEY`. Uses `function_calling` structured output for
    consistency with the planner factory (no strict-mode constraints needed
    since `_IterationDecisionPayload` is a closed shape, but keeping the
    method consistent makes the integration uniform).
    """

    from app.runtime.chat_models import build_openai_chat

    chat = build_openai_chat(model, temperature=temperature)
    structured = chat.with_structured_output(
        _IterationDecisionPayload, method="function_calling"
    )
    return LlmIterationDecider(
        structured,
        success_criteria=success_criteria,
        fallback=fallback,
        judge_failed_runs=judge_failed_runs,
        value_render_limit=value_render_limit,
    )


def build_provider_iteration_decider(
    model_provider: Any,
    model_ref: Any,
    *,
    success_criteria: str | None = None,
    fallback: IterationDecider | None = None,
    judge_failed_runs: bool = False,
    value_render_limit: int = 4000,
) -> LlmIterationDecider:
    """Build an LLM judge from a registered model provider."""

    structured = model_provider.build_structured_output(
        model_ref,
        _IterationDecisionPayload,
    )
    return LlmIterationDecider(
        structured,
        success_criteria=success_criteria,
        fallback=fallback,
        judge_failed_runs=judge_failed_runs,
        value_render_limit=value_render_limit,
    )
