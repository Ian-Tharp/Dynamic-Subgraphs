"""LLM-backed planner — emits a validated GraphSpec from a natural prompt.

Design choices:

- The planner depends on the *abstract* `Runnable` returned by
  `with_structured_output(GraphSpec)`. The concrete model (ChatOpenAI here)
  is wired by the `build_openai_planner` factory, so the rest of the system
  has no direct dependency on langchain-openai.
- Schema-constrained generation does most of the work; we still run the
  GraphSpec through `validate_graph_spec` because Pydantic alone does not
  enforce topology, budget, allowlists, or reachability.
- If validation fails, we feed the issues back to the model and ask for a
  repaired spec. Max 2 retries by default; raises `PlannerError` after that.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.compiler import COMPILER_HANDLED_KINDS
from app.models import GraphSpec, NodeKind
from app.registry import (
    DEFAULT_SUBAGENTS,
    DEFAULT_TOOLS,
    Registry,
    RegistryValidationError,
    validate_graph_spec,
)
from app.runtime.runners import default_runners


PLANNER_SYSTEM_PROMPT = """\
You are the graph planner for a dynamic subgraphs runtime. Given a user
prompt, you emit a `GraphSpec`: a small bounded workflow the runtime will
compile and execute.

Hard rules:
- Every node id must be unique and snake_case.
- Every node `kind` must be one of these — and ONLY these. Any other kind
  will be rejected at compile time:
    {executable_kinds}
- `tool_call` nodes may only reference tools from this allowlist:
    {tools}
- `spawn_subagent` nodes may only reference subagents from this allowlist:
    {subagents}
- The graph must have at least one START -> ... -> END path.
- Every node `inputs` key must be produced by an upstream node's `outputs`.
- Keep the graph small. Prefer fewer than 6 nodes unless the task clearly
  requires more.
- Provide a short `rationale` explaining why this shape fits the request.

Required params per executable kind (omitting these will fail validation):
- `llm_call`  -> {{ "instruction": "<non-empty string>" }}
- `tool_call` -> {{ "tool_name": "<allowlisted>", "args": {{...}} }}
- `spawn_subagent` -> {{ "agent_name": "<allowlisted>",
                        "task": "<what you want them to do>" }}
  Use a subagent when the task benefits from a role-specialized point of
  view rather than another generic llm_call. The subagent receives your
  `task` string AND the parent run's `state.values` as context, and
  returns a single string that lands in this node's `outputs[0]`.
  Pick `agent_name` from the subagent allowlist above based on the
  agent's role name (e.g., `document_specialist` for document analysis,
  `critic` for review/critique). Don't invent names — they must be in
  the allowlist.
- `reduce`    -> {{ "strategy": one of [{reduce_strategies}],
                  "input_keys": ["..."], "output_key": "...",
                  "instruction": "..." (REQUIRED when strategy is "llm_summarize"
                  — describes how to synthesize the inputs) }}
- `parallel_map` -> {{ "over": "<state_key_holding_a_list>",
                     "child_kind": "llm_call" | "tool_call",
                     "child_params": {{ ...child kind's required params... }} }}
  Convention for parallel_map children: the runtime exposes the current
  item to each worker under `state.values["item"]`. So an `llm_call` child's
  `instruction` should reference "the item in upstream state" (the model
  will see it as `- item: <value>` in the upstream-state block). A
  `tool_call` child receives the item automatically as `args.item`.
  The parallel_map node's `outputs` should declare one key — the final
  ordered list of per-item results lands there. The list at `over` must
  be produced by an upstream node's `outputs`.
- `branch` -> {{ "branches": ["<node_id_a>", "<node_id_b>", ...],
                "decision_key": "<state_key>" }}
  Rules for branch nodes (all enforced by the validator):
    1. Every name in `branches` must be the `id` of another node you
       declared in `nodes`. They are NOT free-form labels.
    2. Your `edges` must include exactly one edge from this branch to
       each name in `branches` — no more, no less.
    3. `decision_key` must be in this branch's `inputs`, and that key
       must be produced as an `outputs` of some upstream node.
    4. The branch declares no value `outputs` of its own (use `[]`).
    5. At runtime, the upstream node must write a string into
       `state.values[<decision_key>]` whose value exactly matches one of
       the names in `branches`. Anything else halts the workflow.
  Use `branch` when the workflow needs to choose one of several
  downstream paths based on prior output. Don't use it for parallel
  fan-out (that's `parallel_map`).
- `wait_for_event` -> {{ "event_type": "<name_of_event>",
                       "timeout_seconds": <int | null> }}
  Pauses the workflow durably. The caller (the human operator or an
  external system) must later resume the run with an event payload via
  `supervisor.resume(run_id, event=<payload>)`. That payload becomes the
  node's output (under `outputs[0]`).
  Use this ONLY when the user's request explicitly requires waiting for
  external input (a webhook, a human approval, a long external job, an
  async callback). Do NOT use it as a way to "think more" or to add a
  pause between LLM calls — the workflow stops until somebody manually
  resumes it.
  Common `event_type` names: "human_approval", "webhook_callback",
  "user_response", "external_job_result". Pick one that describes what
  the caller needs to supply.
- `emit_artifact` -> {{ "artifact_type": "file" | "message" | "report",
                      "name": "<short_artifact_name>",
                      "content_key": "<state_key_holding_the_content>" }}
  Persists a piece of the workflow's output to the configured sink (a
  file under the run directory, by default). The runner reads
  `state.values[<content_key>]` (must be in this node's `inputs`),
  hands it to the sink, and writes a reference dict (name + type + path)
  into `outputs[0]`. Use `emit_artifact` for explicit, named outputs
  the caller will want to find on disk: a final report, an emitted
  configuration, a generated message body. Don't use it for
  intermediate state — that's what `state.values` is for.
- `spawn_subgraph` -> {{ "sub_goal": "<goal for a bounded child workflow>",
                       "name": "<short_snake_case_id>",
                       "inputs_from": ["<parent_state_key>", ...] (optional) }}
  Delegates a self-contained sub-task to a freshly planned CHILD graph that
  runs to completion and hands its produced values back into this node's
  `outputs[0]` (as a dict of the child's outputs). The child is planned from
  `sub_goal` at runtime, runs one level deeper, and is budget-clamped to this
  graph's remaining LLM allowance. `inputs_from` seeds the listed parent
  `state.values` keys into the child.
  Use this ONLY when a sub-task genuinely deserves its own multi-step workflow
  (it would itself need several nodes / its own fan-out or routing). For a
  single step, just use `llm_call` or `tool_call` — do NOT wrap one call in a
  subgraph. Children may not use `wait_for_event` (no durable pause inside a
  child). Keep nesting shallow.

Edges connect node ids. `START` and `END` are literal string sentinels —
not node ids you declare in `nodes`. Every graph needs at least one edge
out of `"START"` and at least one edge into `"END"`.

CRITICAL — input/output keys must match exactly:
A downstream node's `inputs` list must contain only strings that appear in
some upstream node's `outputs` list. Match them character-for-character.
If a producer writes `outputs: ["draft"]`, the consumer must say
`inputs: ["draft"]`, not `inputs: ["summary"]` or `inputs: ["the_draft"]`.

Worked example — two producers feeding one reducer:
```json
{{
  "schema_version": 1,
  "graph_id": "example-002",
  "goal": "compare two sources",
  "rationale": "fan-out two llm_calls, then concat them",
  "nodes": [
    {{
      "id": "summarize_a",
      "kind": "llm_call",
      "outputs": ["summary_a"],
      "params": {{"instruction": "Summarize source A."}}
    }},
    {{
      "id": "summarize_b",
      "kind": "llm_call",
      "outputs": ["summary_b"],
      "params": {{"instruction": "Summarize source B."}}
    }},
    {{
      "id": "combine",
      "kind": "reduce",
      "inputs": ["summary_a", "summary_b"],
      "outputs": ["combined"],
      "params": {{
        "strategy": "concat",
        "input_keys": ["summary_a", "summary_b"],
        "output_key": "combined"
      }}
    }}
  ],
  "edges": [
    {{"from": "START", "to": "summarize_a"}},
    {{"from": "START", "to": "summarize_b"}},
    {{"from": "summarize_a", "to": "combine"}},
    {{"from": "summarize_b", "to": "combine"}},
    {{"from": "combine", "to": "END"}}
  ]
}}
```
Note how `summarize_a.outputs[0]` ("summary_a") is the same literal string
as `combine.inputs[0]` and `combine.params.input_keys[0]`.

If repairs are requested, fix the listed issues exactly and emit the full
corrected spec; do not narrate.
"""


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce a valid GraphSpec."""


class StructuredChatModel(Protocol):
    """The minimum surface a structured-output model must expose."""

    def invoke(self, messages: list, /, **kwargs: Any) -> Any: ...


class LLMPlanner:
    """Planner that asks an LLM for a `GraphSpec` and validates it.

    Args:
        structured_model: a chat model already configured with
            `with_structured_output(GraphSpec)`. Must return a GraphSpec
            instance from `.invoke(messages)`.
        registry: Registry used for validation. Defaults to the project default.
        max_retries: how many times to ask the model to repair an invalid spec.
        system_prompt: override only if you have a specific reason to.
    """

    def __init__(
        self,
        structured_model: StructuredChatModel,
        *,
        registry: Registry | None = None,
        max_retries: int = 2,
        system_prompt: str | None = None,
        executable_kinds: set[NodeKind] | None = None,
        executable_reduce_strategies: set[str] | None = None,
    ) -> None:
        self._model = structured_model
        self._registry = registry or Registry()
        self._max_retries = max_retries
        self._executable_kinds = (
            executable_kinds
            if executable_kinds is not None
            else set(default_runners().keys()) | set(COMPILER_HANDLED_KINDS)
        )
        self._executable_reduce_strategies = (
            executable_reduce_strategies
            if executable_reduce_strategies is not None
            else {"concat", "merge_dict"}
        )
        self._system_prompt = system_prompt or PLANNER_SYSTEM_PROMPT.format(
            tools=", ".join(sorted(self._registry.tools)) or "(none)",
            subagents=", ".join(sorted(self._registry.subagents)) or "(none)",
            executable_kinds=", ".join(sorted(k.value for k in self._executable_kinds))
            or "(none)",
            reduce_strategies=", ".join(
                f'"{s}"' for s in sorted(self._executable_reduce_strategies)
            )
            or '"concat"',
        )

    def __call__(self, prompt: str) -> GraphSpec:
        messages: list = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt),
        ]

        last_issues: list[str] = []
        for attempt in range(self._max_retries + 1):
            candidate = self._model.invoke(messages)
            if not isinstance(candidate, GraphSpec):
                raise PlannerError(
                    f"Model did not return a GraphSpec (got {type(candidate).__name__})"
                )

            try:
                return validate_graph_spec(candidate, registry=self._registry)
            except RegistryValidationError as exc:
                if attempt == self._max_retries:
                    raise PlannerError(
                        f"Planner failed after {self._max_retries + 1} attempt(s): {exc}"
                    ) from exc
                last_issues = [
                    f"- [{i.code}] {i.message}"
                    + (f" (node={i.node_id})" if i.node_id else "")
                    + (f" (field={i.field})" if i.field else "")
                    for i in exc.issues
                ]
                declared_outputs = sorted(
                    {key for node in candidate.nodes for key in node.outputs}
                )
                context = (
                    f"\nFor reference, the keys your previous spec declared as "
                    f"outputs across all nodes were: "
                    f"{', '.join(declared_outputs) if declared_outputs else '(none)'}.\n"
                    "If an input is reported missing, either (a) rename the "
                    "consumer's input to match a producer's existing output key "
                    "character-for-character, or (b) add the needed output to "
                    "the upstream node."
                )
                messages.append(
                    HumanMessage(
                        content=(
                            "The previous spec failed validation. Repair these "
                            "exact issues and emit the corrected full spec:\n"
                            + "\n".join(last_issues)
                            + context
                        )
                    )
                )

        # unreachable — the loop either returns or raises
        raise PlannerError("Planner exited without producing a spec")


def build_openai_planner(
    model: str = "gpt-5.4-nano",
    *,
    registry: Registry | None = None,
    max_retries: int = 2,
    temperature: float | None = None,
    executable_reduce_strategies: set[str] | None = None,
) -> LLMPlanner:
    """Convenience factory: build an LLMPlanner backed by ChatOpenAI.

    Requires `OPENAI_API_KEY` in the environment. The langchain-openai import
    is local so importing this module does not force the dependency on
    callers that only use `StaticPlanner`.
    """

    from app.runtime.chat_models import build_openai_chat

    chat = build_openai_chat(model, temperature=temperature)
    # Use function_calling rather than the default strict json_schema mode.
    # GraphSpec has `params: dict[str, Any]` (an open dict by design — each
    # node kind has its own typed params model that we validate post-hoc),
    # which OpenAI's strict mode rejects because it can't enforce
    # additionalProperties=false on it. Function-calling is more permissive
    # and works with the existing schema unchanged.
    structured = chat.with_structured_output(GraphSpec, method="function_calling")
    return LLMPlanner(
        structured,
        registry=registry,
        max_retries=max_retries,
        executable_reduce_strategies=executable_reduce_strategies,
    )


# Re-export the Callable form so it composes naturally with the Planner type alias
# used by the supervisor.
Planner = Callable[[str], GraphSpec]
