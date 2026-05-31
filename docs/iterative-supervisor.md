# Iterative Supervisor

The first adaptive orchestration slice is `Supervisor.run_iteratively(...)`.
It does not let a transient graph mutate while it is executing. Instead, it
runs a normal bounded supervisor pass, records it, asks a typed decider what to
do next, and optionally plans another bounded pass with compact prior-run
context.

```text
prompt
  -> supervisor.run(...)
  -> recorded run
  -> IterationDecision
  -> stop | replan | ask_user | fail
```

This keeps every attempt compatible with the existing guarantees:

- each iteration has a validated `GraphSpec`;
- each iteration has a normal run directory;
- failed iterations are still recorded;
- replanning happens between runs, not through arbitrary runtime mutation.

## API

```python
result = supervisor.run_iteratively(
    "Investigate the task and revise if the first pass is incomplete.",
    run_id="chain-001",
    max_iterations=3,
)
```

The default decider is intentionally conservative:

- `ok` -> `stop`;
- `paused` -> `ask_user`;
- anything else -> `fail`.

Pass a custom `IterationDecider` to enable adaptive behavior:

```python
from app.supervisor import IterationContext, IterationDecision


def decider(context: IterationContext) -> IterationDecision:
    if context.iteration == 1:
        return IterationDecision(
            action="replan",
            reason="The first pass found missing evidence.",
            gaps=["Need a grounded source check."],
        )
    return IterationDecision(
        action="stop",
        reason="The second pass is sufficient.",
        success_criteria_met=True,
    )
```

If a `replan` decision omits `next_prompt`, the supervisor builds a compact
planner prompt from the original prompt, prior status, prior output keys,
errors, record directory, and declared gaps.

## Strict Runners

`LangGraphExecutor(strict_runners=True)` disables placeholder runner defaults.
In that mode, every non-compiler-handled node kind used by a spec must have an
explicit runner. This prevents production runs from silently using echo
implementations for `llm_call`, `tool_call`, `spawn_subagent`, or
`emit_artifact`.

```python
executor = LangGraphExecutor(
    strict_runners=True,
    runners={
        NodeKind.LLM_CALL: real_llm_runner,
        NodeKind.TOOL_CALL: real_tool_runner,
        NodeKind.EMIT_ARTIFACT: artifact_runner,
    },
)
```

Use strict mode when evaluating adaptive loops. Otherwise the loop can appear
to succeed while only replaying placeholder behavior.

## Grounded Tool Runner

`build_grounded_tool_runner()` wires concrete implementations for the default
v1 tool allowlist:

- `web_search` queries DuckDuckGo's public instant-answer API and returns a
  compact evidence payload;
- `policy_lookup` returns the local registry allowlist and forbidden kinds;
- `mock_document_extract` extracts basic statistics from caller-provided text;
- `create_follow_up_task` returns a structured task object without external
  side effects.

`python -m app.main --llm ...` uses this runner with strict mode enabled, so
LLM-backed demo runs fail if they plan a `tool_call` that has no explicit
implementation.
