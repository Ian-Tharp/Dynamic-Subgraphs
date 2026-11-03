# Plan-level governance

Call-level guardrails (model-call limits, human approval on a single tool call,
PII filters) are now common in agent frameworks. They look at one call at a
time. Dynamic Subgraphs governs the **plan**: the whole synthesized `GraphSpec`
is checked against host policy **before it compiles**, its worst-case dollar
cost is bounded **before any node runs**, execution can be restricted to a
**catalog of pre-approved subgraphs**, and every run returns a **provenance
receipt** that can be verified against the recorded log.

Everything here lives in `app/governance/` (re-exported from
`dynamic_subgraphs.governance`). The checks are pure functions: no I/O, no
model calls, deterministic output. Every new policy field is **off by
default**, so an unconfigured run behaves exactly as before.

```text
plan ──► registry validation ──► PLAN GATE ──► compile ──► run ──► record ──► receipt
                                   │ structure · capabilities · data_flow
                                   │ catalog · cost
                                   └─► policy_denied (nothing compiled, nothing run)
```

## 1. Plan policy gate

`check_plan_policy(spec, policy, *, run_models=())` returns a `PlanGateReport`
with one `GateDecision` per gate. It never raises for a bad plan;
`enforce_plan_policy` is the raising variant (`PlanPolicyViolation`).

| Gate | Checks | Violation codes |
|------|--------|-----------------|
| `structure` | node count; longest START→END chain (`max_plan_length`); a `spawn_subgraph`/`gauntlet` needs `max_depth >= 2` | `max_nodes_exceeded`, `max_plan_length_exceeded`, `nesting_depth_exceeded` |
| `capabilities` | node kinds, tools (incl. `parallel_map` tool children), subagents, node `params.model`, and the run's role models | `node_kind_not_allowed`, `tool_not_allowed`, `subagent_not_allowed`, `model_not_allowed`, `run_model_not_allowed` |
| `data_flow` | forbidden edges and required gates (below) | `ungated_tool_to_state_change`, `missing_required_gate` |
| `catalog` | in catalog mode, the plan must be an approved entry | `plan_not_in_catalog` |
| `cost` | worst-case bound vs `max_cost_usd` / `max_tokens` | `cost_bound_exceeded`, `token_bound_exceeded`, `unpriced_model`, `no_cost_model` |

Each `PolicyViolation` carries `gate`, `code`, `message`, and, when it applies,
`node_id` and `field`.

**Fail closed.** If a gate raises, the plan is denied with `gate_error`. A
dollar ceiling with no cost model (`no_cost_model`), or a counted model with no
price (`unpriced_model`), is a deny, not a pass. A policy that names a tool class
missing from `tool_classes` raises `ValueError` when you construct it.

### Forbidden edges and required gates

Tools are grouped into named classes. Two rules use them:

- **State-changing classes** (`state_changing_classes`): the output of *any*
  tool node may not reach a tool in these classes unless every path between
  them passes a gate node. Node kinds in `state_changing_node_kinds` (for
  example `emit_artifact`) are protected the same way.
- **Gate-required classes** (`gate_required_classes`): a tool in these classes
  needs a gate node on every path from START.

A gate node is any node whose kind is in `gate_node_kinds` (default:
`wait_for_event`, the human-approval pause). The check is path-based: a side
edge that goes around the gate is caught.

```python
from app.models import NodeKind
from dynamic_subgraphs import ExecutionPolicy

policy = ExecutionPolicy(
    allowed_models=frozenset({"gpt-5.4-nano", "claude-haiku-4-5"}),
    max_plan_length=6,
    tool_classes={
        "state_changing": frozenset({"create_follow_up_task"}),
        "payments": frozenset({"issue_refund"}),
    },
    state_changing_classes=frozenset({"state_changing", "payments"}),
    gate_required_classes=frozenset({"payments"}),
    state_changing_node_kinds=frozenset({NodeKind.EMIT_ARTIFACT}),
)
```

### Where the gate runs

- **Root runs:** in the supervisor's validate step, after registry validation
  and before compile. A denied plan ends as status **`policy_denied`**. If
  `max_plan_attempts` allows a re-plan and some violation is fixable by the
  planner, the violations and the cost/length limits go back to the planner,
  the same way registry validation errors do. These are never repaired:
  `run_model_not_allowed`, `plan_not_in_catalog`, `no_cost_model`, `gate_error`.
- **Replays:** `Supervisor.replay` re-checks the recorded plan against the
  *current* policy, so a tightened policy stops an old plan from running again.
- **Nested children** (`spawn_subgraph`, `gauntlet` builder rounds): the
  `structure`, `capabilities`, `data_flow` and `catalog` gates run on every
  child plan. Cost is bounded at the root (below), and the shared budget ledger
  enforces it for children.

Every decision is also written to the run's trace as a `policy_decision` event
(at the head of `trace.jsonl`), so the recorder log shows what was checked.

## 2. Pre-flight cost bound

`estimate_plan_cost(spec, cost_model, *, max_llm_calls, max_fanout)` returns a
`CostBound` (worst-case LLM calls, tokens and USD) from a **static price table**
and **per-call token caps**. It makes no network calls and no model calls.

| Node | Counted calls |
|------|---------------|
| `llm_call`, `spawn_subagent`, `reduce` with `llm_summarize` | 1 |
| `parallel_map` with an `llm_call` child | effective `max_fanout` |
| `spawn_subgraph`, `gauntlet` | open-ended: slack calls at the most expensive model in play, up to `max_llm_calls` |

The runtime ledger already refuses LLM call *N+1* past the effective
`max_llm_calls`, so the bound keeps only the `max_llm_calls` most expensive
calls. Each call is priced at its node's `params.model`, else the cost model's
per-kind model, else its `default_model`, using the call's token cap.

```python
from dynamic_subgraphs import CostModel, ExecutionPolicy, ModelPrice, TokenCap

policy = ExecutionPolicy(
    max_cost_usd=0.05,
    max_tokens=60_000,
    cost_model=CostModel(
        prices={
            "gpt-5.4-nano": ModelPrice(input_per_1m=0.20, output_per_1m=1.25),
            "claude-haiku-4-5": {"input_per_1m": 1.0, "output_per_1m": 5.0},
        },
        default_cap=TokenCap(input_tokens=8_000, output_tokens=2_000),
    ),
)
```

With the SDK, unset routing is filled from the run's role models (worker by
default, reducer for `reduce`, subagent for `spawn_subagent`, and the planner
and judge models as candidates for open-ended slack). If the policy sets a
ceiling but has no `cost_model`, `EngineConfig(pricing=...)` is used as the
price table. Your `ExecutionPolicy` object is never rewritten: the receipt's
`policy_hash` is always of the policy as you configured it. The exact prices,
caps and routing used are recorded as `cost_model_hash` in the cost gate's
details.

What the bound does and doesn't cover:

- It's an upper bound **under the declared token caps**. You can enforce the
  output cap at the provider (`max_tokens`), but input size depends on upstream
  values. The receipt puts the bound next to the actual spend, and
  `within_bound` flags an overrun.
- The **root planner call** happens before a plan exists, so it can't be
  bounded in advance. In catalog mode with the default keyword selector,
  routing makes no model calls.

## 3. Approved-subgraph catalog mode

Set `ExecutionPolicy(approved_catalog=SubgraphCatalog(...))` to restrict
execution to a closed set of reviewed GraphSpecs. In this mode:

- the SDK plans with `CatalogPlanner`, which **selects** an entry for the
  prompt and fills it in. It never writes a new plan. Nested
  `spawn_subgraph` children also pick from the catalog.
- the `catalog` gate refuses any plan that isn't a filled entry. This includes
  a `fixed_spec`, a replayed spec, and a child plan.

Entries are templates. Any string in a node's params may contain `{prompt}`.
Matching treats `{prompt}` as a wildcard that must have the **same** value
everywhere it appears, so a plan matches only if it is the template filled with
one prompt. `budget` (the host clamps it anyway), `goal`, `rationale` and
`metadata` are not compared. Each entry has a content hash (`spec_hash`), which
the receipt records.

```python
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, ExecutionPolicy, SubgraphCatalog
from dynamic_subgraphs.governance import CatalogEntry, build_llm_catalog_selector

catalog = SubgraphCatalog(
    entries=(
        CatalogEntry("summarize", summarize_spec, "Summarize a document", ("summarize",)),
        CatalogEntry("claims_triage", triage_spec, "Triage a claim", ("claim", "triage")),
    ),
    # Optional. The default is a token-free keyword selector (first entry on no hit).
    # selector=build_llm_catalog_selector(chat_model),
)
engine = DynamicSubgraphs(EngineConfig(policy=ExecutionPolicy(approved_catalog=catalog), ...))
```

`build_llm_catalog_selector(chat_model)` lets a model choose. It can only
answer with a catalog id; any other answer fails the run as `plan_failed`.

## 4. Provenance receipt

Every `Supervisor.run` / `Supervisor.replay`, and every SDK `run()` (including
`policy_denied`, failed and `engine_failed` runs), returns a `ProvenanceReceipt`.
You get it as `RunResult.receipt` and `SupervisorResult.receipt`. When the run
is recorded, it's also written to `runs/<run_id>/receipt.json`
(`Artifact.RECEIPT`).

| Field | Meaning |
|-------|---------|
| `policy_hash` | sha256 of the `ExecutionPolicy` as configured (incl. its `cost_model` and catalog entry hashes) |
| `plan_hash` | sha256 of the validated plan (`spec.json`), or of the refused plan |
| `catalog_entry_id` / `catalog_entry_hash` | the approved entry the plan matched |
| `gate_decisions` | allow/deny per gate, with violations and details |
| `cost_bound` / `cost_actual` / `within_bound` | worst-case vs observed calls, tokens, USD |
| `model_ids` | configured role models, node-pinned models, models seen in usage |
| `started_at` / `finished_at` | UTC timestamps |
| `recorder_run_id` / `trace_digest` | the recorded run and a digest of its `trace.jsonl` |
| `receipt_hash` | sha256 over every other field |

All hashes are `sha256:` over canonical JSON (sorted keys, compact).

```python
from pathlib import Path
from dynamic_subgraphs import verify_receipt

check = verify_receipt(result.receipt, Path("runs") / result.receipt.recorder_run_id,
                       policy=engine.config.policy)
assert check.ok, check.problems
```

`verify_receipt` reports a problem if:

- the receipt was edited (`receipt_hash` mismatch);
- `spec.json` (or `rejected_spec.json` for a refused plan) doesn't hash to
  `plan_hash`;
- `trace.jsonl` doesn't digest to `trace_digest`, or its `policy_decision`
  events disagree with `gate_decisions`;
- `output.json` disagrees with `status`;
- `receipt.json` on disk isn't this receipt;
- the given policy doesn't hash to `policy_hash`.

Files the recording policy skipped are listed in `checked`; they aren't counted
as problems.

The receipt is integrity evidence, not a signature. Anyone who can rewrite both
the recorder files and the receipt can recompute the hashes. For
non-repudiation, store `receipt_hash` somewhere append-only, or sign it with
your own key.

### OpenTelemetry

The recorder has no span concept today (it writes a flat `TraceEvent` list), so
receipts don't emit OpenTelemetry GenAI attributes. If span export is added
later, the receipt fields map directly onto span attributes.

## Known limits

- Data flow is checked inside one graph. A parent's tool output passed to a
  child through `inputs_from` isn't traced across the boundary, but the child's
  own plan is still gated.
- `Supervisor.resume` continues a paused run under the plan the gate already
  approved. It doesn't re-run the gate and doesn't return a new receipt.
- Node `params.model` is checked by policy, but the built-in LLM runners use the
  run's role model. A pinned model only affects the gate and the cost bound.
