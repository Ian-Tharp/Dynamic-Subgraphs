> **Status:** Design — proposed, not yet implemented. Source of truth for the ExecutionPolicy work.
> **Provenance:** Produced by a 6-agent design workflow (3 lenses → synthesis →
> adversarial bypass red-team → reconciliation), each grounded in the actual source.
> The red-team's job was to make the planner exceed the host budget; it found 8 bypass
> vectors (fan-out×depth, nested re-mint, sibling TOCTOU, un-narrowed child registry,
> subagent/reduce LLM backdoors, wall-clock evasion) — all closed inline (§11).
> **Blocking:** Section 12 lists `DECISION:` points that need Ian before the dependent PRs.

# DS ExecutionPolicy: Host-Owned Governance — Final Implementable Spec

This is the final, implementation-ready design. It integrates every red-team fix so a governed planner cannot exceed the host budget through any path I could verify in the source: static self-grant, fan-out × depth multiplication, nested re-mint, sibling TOCTOU, un-narrowed child registry, the `spawn_subagent`/`reduce` LLM backdoors, or wall-clock evasion. Every line below is grounded in code I read; file:line anchors are exact as of the current tree.

---

## 0. Source facts that drive (and constrain) the design — re-verified

- **`GraphBudget`** (`app/models/graph_spec.py:15-19`): `max_nodes=12 (ge=1)`, `max_depth=2 (ge=1)`, `max_wall_seconds=90 (ge=1)`, `max_llm_calls=8 (ge=0)`. No `max_fanout`.
- **Validator** (`app/registry/validator.py`): node count vs `spec.budget.max_nodes` (line 76), llm count vs `spec.budget.max_llm_calls` (line 85), `spec.budget.max_depth` vs `MAX_DEPTH_CEILING=3` (lines 93-103, ceiling defined line 21). Signature `validate_graph_spec(spec, registry=None)`; returns `spec.model_copy(update={"nodes": normalized_nodes})` (line 113).
- **Executor** (`app/runtime/executor.py`): `_recursion_limit_for(spec)` = `max(25, spec.budget.max_nodes * 4)` (lines 268-276), called from `_invoke_config(spec, run_id)` (line 222). `execute()` seeds metadata with `budget_max_llm_calls` + `run_id` *after* the caller dict (lines 173-177), so a caller can't override them. `_run_graph_isolated` (lines 25-68) runs `graph.invoke` in a **non-daemon** thread joined with **no timeout**. `max_wall_seconds` referenced nowhere.
- **Counters ledger** (`app/models/run_state.py:31-44`): `counters` is `Annotated[dict, add_counters]` — **summed** across concurrent branches. `nodes_executed`/`llm_calls_consumed` are written by `node_counter_delta` (`wrappers.py:120-130`) and PM workers (`parallel_map.py:199,239`) but **never compared to any ceiling** at runtime. The only consumer today is the spawn clamp (`subgraph.py:115-121`). `metadata` is `Annotated[dict, merge_dicts]` — **last-writer-wins** (used by `run_id`/`graph_depth`).
- **`parallel_map`** (`app/runtime/parallel_map.py`): dispatcher decodes a JSON-string `over` into a list at lines 91-104, type-checks at 106, empty-list shortcut at 123-131, then emits one `Send` per item at 133-146 (each Send copies `metadata` at line 142). No fan-out ceiling. `_halt_with_error(...)` (line 312) is the fail-closed exit. `child_kind` is `Literal["tool_call","llm_call"]` (`app/registry/params.py:27`) — **`spawn_subgraph` is not expressible as a PM child** (structurally true today; keep it that way).
- **`spawn_subgraph`** (`app/runtime/subgraph.py`): runner clamps **only** `max_llm_calls = max(0, budget_max_llm_calls - llm_calls_consumed)` (lines 115-121), passes `max_llm_calls=` to launcher (line 135). Depth ceiling = hard-coded `MAX_DEPTH_CEILING` (line 97). `make_child_launcher.launch` re-`min()`s into `spec.budget` (lines 196-203) and calls `validate_graph_spec(spec, registry)` (line 204). Child metadata is hand-built as an allowlist `{"graph_depth","parent_run_id"}` (+ optional `replay_of`) at lines 212-218 — **wall deadline / effective budget would be dropped by omission today.** Spend rolls up via reserved `__spend__` key in `make_node_wrapper` (`wrappers.py:99-115`) **after** the child returns.
- **Registry** (`app/registry/registry.py`): constructor takes `tools=`, `subagents=`, `kinds=` (lines 16-25). Tool gate `tool_not_allowlisted` for both `tool_call` and PM child tools (lines 106-141); subagent gate `subagent_not_allowlisted` (lines 153-167). `count_llm_calls` adds **+1** per PM-over-`llm_call` (lines 207-210, "upper bound per fan-out handled at runtime" — but it is **not** handled at runtime). `counts_as_llm_call_for_node` (171-193) covers `reduce/llm_summarize` and excludes `spawn_subgraph`/`parallel_map`. `allowed_kinds()` returns `frozenset(self._kinds.keys())`. `DEFAULT_TOOLS` = `{policy_lookup, create_follow_up_task, web_search}`; `DEFAULT_SUBAGENTS` = `{document_specialist, critic}` (`allowlists.py`).
- **Wiring (the critical gap):** `build_supervisor` (`app/assembly.py:212-256`) constructs `LangGraphExecutor(...)` (lines 243-247) and `make_child_launcher(planner=, executor=, recorder=)` (line 254) **both with no `registry=` argument** — so every validate (root and child) runs against a default `Registry()` today. The root validate happens in the *static supervisor graph*: `_make_validate_node` calls `validate_graph_spec(state["spec"])` (`app/supervisor/graph.py:111`, **no registry, no policy**); `_make_execute_node` calls `executor.execute(compiled, run_id=...)` (line 154, **no initial_metadata**). `build_supervisor_graph(planner, executor, recorder)` takes no policy (lines 33-38). `Supervisor.__init__` builds and compiles that graph (`supervisor.py:65-69`).
- **Facade:** `EngineConfig` (frozen, `dynamic_subgraphs/engine.py:343-401`) → `DynamicSubgraphs.__init__` (443) → `build_supervisor(...)` (541-548). `RunConfig` (`assembly.py:45-124`) is the model-role seam from `ModelSelection.to_run_config`. `Model = ModelRef` (engine.py:65) and `from app.runtime import ModelRef` in the facade `__init__` is the established "app type re-exported as public name" pattern. `RunResult._from_supervisor` reads `result.validated_spec` (engine.py:335) — the `SupervisorResult.validated_spec` is the SDK's window into the run.
- **Existing `max_wall_seconds` readers:** `app/api/serialize.py:21` and `app/api/routers/runs.py:59` already read `spec.budget.max_wall_seconds` for job metadata. **`GraphBudget` must stay** and the recorded `spec.budget` should reflect the *granted* wall so those readers report the real cap.
- **`RUN_STATUSES`** live in `dynamic_subgraphs/types.py:17-30` (and mirror `app/supervisor/state.py:21-35`). Adding a status requires editing both.

---

## 1. Goal & non-goals

**Goal.** Move every budget ceiling out of the planner-produced `GraphSpec` into a host-owned `ExecutionPolicy` on `EngineConfig`. The planner may still *request* a `GraphBudget`, but the **effective** limit for each numeric field is `min(planner_request, host_ceiling, parent_remaining)`, and the tool / subagent / node-kind vocabulary is the **intersection** of the host allow-set with the registry. Make node-count, llm-calls, depth, fan-out, and wall-clock **host-enforced at the point where they can actually be violated** (runtime, not only validation). Hold the policy **across `spawn_subgraph` nesting** (a child can never escape the parent's *remaining* budget). Surface effective limits for inspection. Change nothing for callers who set no policy.

**Non-goals.**
- Not removing `GraphBudget` from `GraphSpec` (kept as a non-binding *request* — preserves replay-on-disk, planner self-restraint, recorded intent, and the existing API `max_wall_seconds` readers).
- Not hard-killing in-flight LLM/tool calls. Wall-clock is a **between-node + result-acceptance wall plus a bounded `join`**, not a CPU kill (CPython can't kill a thread). Per-provider request timeouts remain the real latency control and are out of scope (documented).
- Not making the registry vocabulary runtime-mutable. The registry is the outer bound; policy only **narrows**.
- Not adding new planner-facing node kinds. Exactly one optional planner-facing schema field is added: `GraphBudget.max_fanout` (see §3, DECISION).

---

## 2. ExecutionPolicy interface + where it lives + EngineConfig wiring

**Module home.** One host-facing dataclass `ExecutionPolicy` and one resolved `EffectiveBudget`, both in a **new `app/policy.py`** (app owns them; the facade re-exports `ExecutionPolicy`). This keeps the architecture rule intact (facade imports app; app never imports facade) with no duplicate facade-side type to drift. `EngineConfig.policy` is directly `app.policy.ExecutionPolicy`, re-exported as a public name — the same pattern as `Model = ModelRef`.

`MAX_DEPTH_CEILING` moves to `app/policy.py` as the single source of truth; `app/registry/validator.py` imports and **re-exports** it (so `subgraph.py`'s `from app.registry.validator import MAX_DEPTH_CEILING` keeps working). Import direction: `policy.py` imports only `app.models.*` (no import of `registry`/`validator`), so no cycle.

```python
# app/policy.py  (NEW — in `app`, importable by the facade; imports only app.models.*)
from __future__ import annotations
from dataclasses import dataclass
from app.models.node_kinds import NodeKind
from app.models.graph_spec import GraphBudget

MAX_DEPTH_CEILING = 3  # absolute nesting rail; validator.py re-exports this.

# Sentinel contract for the allow-sets (tools/subagents/kinds):
#   None        => host imposes NO narrowing  -> effective set = full registry set
#   frozenset() => host forbids ALL of that category (fail-closed, ban-all)
# These are DISTINCT. Never treat falsy as "no narrowing" (that is fail-open).

@dataclass(frozen=True)
class ExecutionPolicy:
    """Host-owned ceilings. Numeric fields are hard upper bounds. The
    collections, when set, are the host's allow-set; the planner may only
    NARROW further via per-node refs, never widen. Immutable; safe to share."""
    max_nodes: int = 12
    max_llm_calls: int = 8
    max_depth: int = 2                 # per-run request cap; clamped to ceiling below
    max_depth_ceiling: int = MAX_DEPTH_CEILING   # absolute nesting rail
    max_fanout: int = 64               # host cap on items per parallel_map dispatch
    max_wall_seconds: int = 90         # now actually enforced
    allowed_tools: frozenset[str] | None = None
    allowed_subagents: frozenset[str] | None = None      # RED-TEAM fix (Bypass 5)
    allowed_node_kinds: frozenset[NodeKind] | None = None

    def __post_init__(self) -> None:
        # Coerce any iterable allow-set to frozenset so equality/identity/as_dict
        # are stable even if a caller passes a plain set (RED-TEAM Bypass 6).
        for name in ("allowed_tools", "allowed_subagents", "allowed_node_kinds"):
            val = getattr(self, name)
            if val is not None and not isinstance(val, frozenset):
                object.__setattr__(self, name, frozenset(val))
        for name in ("max_nodes", "max_depth", "max_fanout",
                     "max_wall_seconds", "max_depth_ceiling"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"ExecutionPolicy.{name} must be >= 1")
        if self.max_llm_calls < 0:
            raise ValueError("ExecutionPolicy.max_llm_calls must be >= 0")
        if self.max_depth > self.max_depth_ceiling:
            object.__setattr__(self, "max_depth", self.max_depth_ceiling)

@dataclass(frozen=True)
class EffectiveBudget:
    """Resolved limits for ONE graph execution. Single source of truth consumed
    by validator, executor, parallel_map, spawn_subgraph; surfaced on RunResult
    and capabilities()['policy']. JSON-safe via as_dict()."""
    max_nodes: int
    max_llm_calls: int
    max_depth: int
    max_fanout: int
    max_wall_seconds: int
    allowed_tools: frozenset[str]
    allowed_subagents: frozenset[str]
    allowed_node_kinds: frozenset[NodeKind]

    def as_dict(self) -> dict[str, object]:
        return {
            "max_nodes": self.max_nodes,
            "max_llm_calls": self.max_llm_calls,
            "max_depth": self.max_depth,
            "max_fanout": self.max_fanout,
            "max_wall_seconds": self.max_wall_seconds,
            "allowed_tools": sorted(self.allowed_tools),
            "allowed_subagents": sorted(self.allowed_subagents),
            "allowed_node_kinds": sorted(k.value for k in self.allowed_node_kinds),
        }
```

**EngineConfig wiring** (`dynamic_subgraphs/engine.py`):
- Add to frozen `EngineConfig`: `policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)`.
- `DynamicSubgraphs.__init__`: `self._policy = config.policy`.
- `run()` passes `policy=self._policy` into `build_supervisor(...)`.
- Re-export `ExecutionPolicy` (and, DECISION-pending, `EffectiveBudget`) from `dynamic_subgraphs/engine.py.__all__` and `dynamic_subgraphs/__init__.py` via `from app.policy import ExecutionPolicy`.

**RunConfig stays model-only.** Do **not** add policy to `RunConfig` (it is the `ModelSelection.to_run_config` seam). Pass `policy` as an explicit keyword of `build_supervisor` — mirrors `recorder`/`checkpointer`/`artifact_sink` already being separate kwargs. Default `policy=ExecutionPolicy()` keeps `build_supervisor` back-compat.

---

## 3. The min(planner, host, remaining) + intersection semantics

`resolve_effective_budget` is the single pure function every enforcement site routes through. **Children are identified structurally and cannot fall back to the full host budget** (RED-TEAM Bypass 3 fix):

```python
# app/policy.py
def resolve_effective_budget(
    policy: ExecutionPolicy,
    request: GraphBudget,
    *,
    registry_tools: frozenset[str],
    registry_subagents: frozenset[str],
    registry_kinds: frozenset[NodeKind],
    remaining: "RemainingBudget | None" = None,   # REQUIRED for any child (depth>0)
    is_child: bool = False,
) -> EffectiveBudget:
    # A child MUST compose against the parent's remaining. No safe default exists:
    # falling back to the full host policy lets an N-level nest re-mint N budgets.
    if is_child and remaining is None:
        raise ValueError(
            "resolve_effective_budget: a child resolution (is_child=True) "
            "requires `remaining`; refusing to grant a child the full host budget."
        )

    host_tools = registry_tools if policy.allowed_tools is None \
        else (policy.allowed_tools & registry_tools)
    host_subs = registry_subagents if policy.allowed_subagents is None \
        else (policy.allowed_subagents & registry_subagents)
    host_kinds = registry_kinds if policy.allowed_node_kinds is None \
        else (policy.allowed_node_kinds & registry_kinds)

    node_cap  = policy.max_nodes      if remaining is None else min(policy.max_nodes, remaining.nodes)
    llm_cap   = policy.max_llm_calls  if remaining is None else min(policy.max_llm_calls, remaining.llm_calls)
    depth_cap = policy.max_depth      if remaining is None else min(policy.max_depth, remaining.depth)

    request_fanout = getattr(request, "max_fanout", policy.max_fanout)  # optional new field
    return EffectiveBudget(
        max_nodes=min(node_cap, request.max_nodes),
        max_llm_calls=min(llm_cap, request.max_llm_calls),
        max_depth=min(depth_cap, request.max_depth, policy.max_depth_ceiling),
        max_fanout=min(policy.max_fanout, request_fanout),
        max_wall_seconds=min(policy.max_wall_seconds, request.max_wall_seconds),
        allowed_tools=host_tools,
        allowed_subagents=host_subs,
        allowed_node_kinds=host_kinds,
    )
```

`RemainingBudget` is a tiny frozen dataclass `(nodes: int, llm_calls: int, depth: int)` in `app/policy.py`, carried in **`counters`/explicit args, never in last-writer-wins `metadata`** (RED-TEAM Bypass 3.3).

Rules:
1. **Per numeric field: EFFECTIVE = min(host, planner request[, parent remaining])** for `max_nodes`, `max_llm_calls`, `max_wall_seconds`. Planner may self-restrain lower; never raise.
2. **Depth: triple min** — `min(host.max_depth, request.max_depth, host.max_depth_ceiling)` (and parent remaining for children). `__post_init__` already clamps `policy.max_depth ≤ ceiling`, so a misconfigured host can't raise the rail.
3. **allowed_tools / allowed_subagents / allowed_node_kinds = INTERSECTION** with the registry, never replacement. `None` ⇒ full registry set (back-compat). `frozenset()` ⇒ ban-all. The planner narrows further implicitly: any node naming a tool/subagent/kind outside the effective set is rejected at validation.
4. **`max_fanout`** uses `getattr(request, "max_fanout", policy.max_fanout)` so old recorded specs without the field resolve (and the new Pydantic default fills it on load anyway).
5. **`GraphBudget` stays** on `GraphSpec`. Nothing downstream reads `spec.budget.*` for *enforcement* anymore — everything reads the resolved `EffectiveBudget`. The validator stamps the resolved budget back onto the returned/recorded `spec.budget` so on-disk specs and the existing API `max_wall_seconds` readers reflect *granted* limits.

---

## 4. Enforcement matrix (each limit → static and/or RUNTIME → exact site)

The headline red-team finding: **static-only enforcement of node/llm budgets is bypassable at runtime** (fan-out × depth, multi-kind LLM backdoors). The fix is a **fail-closed runtime spend-ledger check** that is *authoritative*; static checks remain as fast-fail. Both layers read `EffectiveBudget`, never `spec.budget`.

| Limit | Layer | Exact site | Change |
|---|---|---|---|
| `max_nodes` | **Static** | `validator.py:76` | Compare `len(normalized_nodes) > effective.max_nodes`. |
| `max_nodes` | **RUNTIME (authoritative)** | `wrappers.py:make_node_wrapper._run` (top) and `parallel_map.py:dispatch` (before Send loop) | Halt fail-closed when `counters.nodes_executed (+ projected) > effective.max_nodes`. |
| `max_llm_calls` | **Static (floor)** | `validator.py:85` | Compare `llm_count > effective.max_llm_calls`. |
| `max_llm_calls` | **RUNTIME (authoritative)** | `wrappers.py` (before a `counts_as_llm_call` node runs), `parallel_map.py:dispatch` (project `len(items)` if child counts as llm), `parallel_map.py:worker` (before child_runner) | Halt fail-closed when `consumed + projected > effective.max_llm_calls`. **This is the single fix that closes fan-out×depth (Bypass 1) and the `spawn_subagent`/`reduce` backdoors (Bypass 5).** |
| `allowed_tools` | **Static** | validator via **narrowed Registry** | Build the validating Registry with `tools=effective.allowed_tools`; existing `tool_not_allowlisted` (registry.py:106-141) enforces for `tool_call` and PM child tools — zero new code path. |
| `allowed_subagents` | **Static** | validator via **narrowed Registry** | Registry `subagents=effective.allowed_subagents`; existing `subagent_not_allowlisted` (registry.py:153-167) enforces. **RED-TEAM Bypass 5.** |
| `allowed_node_kinds` | **Static** | `validate_graph_spec` (new loop) | Per node, `node.kind not in effective.allowed_node_kinds` → new issue `node_kind_not_in_policy`. (Dedicated check, not registry `kinds=` removal, which would emit a misleading `unknown_kind`.) |
| recursion rail | **RUNTIME** | `executor.py:_recursion_limit_for` + `_invoke_config` | Derive from `effective.max_nodes`, never `spec.budget`. See §5. |
| `max_fanout` | **RUNTIME** | `parallel_map.py:dispatch`, **after** the JSON decode (line ~106), **before** the Send loop (133) | `if len(items) > effective.max_fanout: _halt_with_error("FanoutExceeded")`. Fail-closed; never truncate. Must be **post-decode** (RED-TEAM Bypass 7). |
| `max_wall_seconds` | **RUNTIME** | `wrappers.py` + `parallel_map` dispatcher & worker (cooperative) + `_run_graph_isolated` (hard backstop) | See §5. |
| depth across nesting | **RUNTIME** | `subgraph.py` runner | Ceiling = `min(effective.max_depth, MAX_DEPTH_CEILING)`; refuse `depth+1 > ceiling`. |
| nodes/llm/depth across nesting | **RUNTIME remaining → child static + child runtime** | `subgraph.py` runner + launcher | `resolve_effective_budget(..., is_child=True, remaining=...)` with **pessimistic reservation** (§6). |

**RUNTIME ledger access.** The wrapper and PM functions already receive `state` (with `counters` and `metadata`). The `EffectiveBudget` is read from `state["metadata"]["effective_budget"]` (seeded by the executor, §5). On breach they emit the same `Command(goto=END)` + `errors` + `NODE_ERROR` path the wrappers/PM already use (`wrappers.py:75-91`, `parallel_map.py:_halt_with_error`), with `type="BudgetExceeded"`. Because `counters` is summed by `add_counters`, the projection in the PM dispatcher (`consumed + len(items)`) is the correct pre-fan-out guard.

> Note on the runtime node check vs LangGraph recursion: the per-node `nodes_executed` check is defense-in-depth; the recursion rail (now scaled off `effective.max_nodes`) remains the structural step ceiling.

---

## 5. Threading the policy + EffectiveBudget + deadline through the seams

`build_supervisor` becomes the single construction chokepoint. It resolves the host allow-sets into **one narrowed Registry** and threads `policy` + that registry into **every** validate and execute site — root and child (closing RED-TEAM Bypass 2 and 5.2):

```python
# app/assembly.py : build_supervisor(..., policy: ExecutionPolicy = ExecutionPolicy())
base = Registry()  # outer bound (DEFAULT_TOOLS / DEFAULT_SUBAGENTS / all kinds)
narrowed = Registry(
    tools     = base.tools     if policy.allowed_tools     is None else (policy.allowed_tools     & base.tools),
    subagents = base.subagents if policy.allowed_subagents is None else (policy.allowed_subagents & base.subagents),
    # kinds NOT removed from the registry (would misreport unknown_kind); the
    # node_kind_not_in_policy check handles kinds. Pass kinds=None (full set).
)
executor = LangGraphExecutor(runners=runners, checkpointer=checkpointer,
                             strict_runners=config.strict_runners,
                             registry=narrowed, policy=policy)          # registry now wired
runners[NodeKind.SPAWN_SUBGRAPH] = build_spawn_subgraph_runner(
    make_child_launcher(planner=planner, executor=executor,
                        recorder=recorder, registry=narrowed, policy=policy))  # registry+policy wired
supervisor = Supervisor(planner=planner, executor=executor, recorder=recorder,
                        registry=narrowed, policy=policy)
```

`Supervisor.__init__` forwards `registry`/`policy` into `build_supervisor_graph(planner, executor, recorder, registry, policy)`; `_make_validate_node` calls `validate_graph_spec(state["spec"], registry=narrowed, policy=policy)`. **Root validate now uses the policy** — closing the gap where `graph.py:111` validated against a default registry with no policy.

**Executor becomes policy-aware** (single place that seeds metadata + builds the recursion rail):

```python
class LangGraphExecutor:
    def __init__(self, *, registry=None, runners=None, checkpointer=None,
                 strict_runners=False, policy: ExecutionPolicy | None = None):
        ...
        self._policy = policy or ExecutionPolicy()

    def execute(self, compiled, *, inputs=None, run_id, initial_metadata=None):
        concrete = _coerce_compiled_graph(compiled)
        md_in = initial_metadata or {}
        # Reuse if a child path already resolved (its budget+deadline span the tree);
        # else mint at the root. SINGLE chokepoint -> one deadline/budget per spawn tree.
        eff_dict = md_in.get("effective_budget")
        if eff_dict is None:
            eff = self._resolve_root(concrete.spec)   # is_child=False, remaining=None
        else:
            eff = EffectiveBudget.from_dict(eff_dict)
        deadline = md_in.get("wall_deadline_monotonic")
        if deadline is None:
            deadline = time.monotonic() + eff.max_wall_seconds
        metadata = {
            **md_in,
            "effective_budget": eff.as_dict(),          # read by wrappers/PM/spawn at RUNTIME
            "budget_max_llm_calls": eff.max_llm_calls,   # back-compat readers
            "wall_deadline_monotonic": deadline,         # absolute monotonic instant
            "run_id": run_id,                            # authoritative (after md_in)
        }
        config = self._invoke_config(eff, run_id)        # recursion off EFFECTIVE
        hard_timeout = max(0.0, deadline - time.monotonic())
        state, paused, payloads = _run_graph_isolated(
            concrete.graph, make_initial_state(inputs=inputs, metadata=metadata),
            config=config, inspect_config=..., hard_timeout_s=hard_timeout)
        ...
```

- `_recursion_limit_for(eff)` / `_invoke_config(eff, run_id)` take the `EffectiveBudget` (its `max_nodes`) — **never `spec.budget`** (RED-TEAM Bypass 2).
- The PM dispatcher reads `effective_budget` and `wall_deadline_monotonic` from `state["metadata"]` (already copied per Send at `parallel_map.py:142`) — no compiler signature change needed.

### Wall-clock mechanism (cooperative primary + bounded-join backstop)

1. **Absolute monotonic deadline** computed once at the root `execute()` (`time.monotonic() + max_wall_seconds`), written to `metadata["wall_deadline_monotonic"]`. An *instant*, so it propagates into children unchanged and means the same moment on every thread.
2. **Cooperative checks at every node-entry point** — not just `make_node_wrapper`. Add `check_deadline(state.metadata)` at the top of `make_node_wrapper._run`, **in the PM dispatcher before the Send loop, and in each PM worker before `child_runner`** (RED-TEAM Bypass 4: fan-out branches are already in flight after dispatch, so the worker check is required). On expiry → `Command(goto=END)` with a `NODE_ERROR` of type `DeadlineExceeded` + an `errors` entry.
3. **Hard backstop:** `_run_graph_isolated` gains `hard_timeout_s` and joins with `thread.join(hard_timeout_s)`; the invoke thread is made `daemon=True` so an abandoned hung node dies with the process. If `thread.is_alive()` after join, raise `DeadlineExceeded` on the caller thread; `execute()` converts it to `ok=False`, status `wall_timeout`. The isolated-thread design is *why* the backstop works — the caller regains control even if `invoke` is wedged in one slow node. **Documented limitation:** the daemon thread may keep *billing* until the slow call returns; wall-clock bounds *latency / supervisor wait*, not spend. The **runtime llm-call ledger (§4) is the spend control.**
4. **`resume()` deadline:** the *original* deadline is carried in checkpointed state; `resume()` does **not** reset to a fresh `max_wall_seconds`. It may extend only by an explicit, host-policy-bounded grant (RED-TEAM Bypass 4). `wait_for_event` is banned in children, so a parked interrupt never trips the tree deadline; the root may pause/resume but cannot mint unbounded fresh windows. **DECISION:** exact resume-extension policy (carry-original vs. bounded re-grant) — default to carry-original.
5. New status **`wall_timeout`** added to `RUN_STATUSES` (`dynamic_subgraphs/types.py`) **and** the `SupervisorState` docstring/allowed set (`app/supervisor/state.py`); maps to `ok=False`.

---

## 6. Nested spawn_subgraph accounting — composes for EVERY field, mandatorily and pessimistically

Today only `max_llm_calls` composes, optionally. The fix makes **every numeric field compose, makes the child resolution mandatory (no full-budget fallback), and reserves the child's granted budget up front to kill the sibling/parallel TOCTOU** (RED-TEAM Bypass 3).

**`build_spawn_subgraph_runner._runner`** (`subgraph.py:94-148`):
- Read `effective_budget` + `wall_deadline_monotonic` from parent `state.metadata`; read live `counters`.
- Depth ceiling = `min(eff.max_depth, MAX_DEPTH_CEILING)`; refuse `depth+1 > ceiling` (fail-closed, as today).
- Compute remaining from the summed ledger:
  - `remaining_llm = max(0, eff.max_llm_calls - counters.get("llm_calls_consumed", 0))`
  - `remaining_nodes = max(0, eff.max_nodes - counters.get("nodes_executed", 0))`
  - `remaining_depth = eff.max_depth - 1`
- **Pessimistic reservation (TOCTOU fix).** The roll-up via `__spend__` only happens *after* the child returns (`wrappers.py:99-115`), so two sibling spawn nodes scheduled in the same superstep would each read the same `consumed`. To close this without forbidding sibling spawns: the spawn runner **emits a `counters` delta reserving the child's *granted* budget at dispatch** (e.g. `{"llm_calls_consumed": granted_llm, "nodes_executed": granted_nodes}` *before* the child runs), then on return rolls up *actual* spend and **refunds** the unused remainder (`granted - actual`) via the same `__spend__`/counters mechanism. Net ledger after a child = actual spend; but concurrent siblings each see the other's reservation, so they cannot both claim the full remaining. Because `counters` is summed (`add_counters`), reservations from concurrent branches aggregate correctly; `metadata` is never used for remaining (last-writer-wins would lose a sibling). **DECISION:** reserve-and-refund (capability-preserving, chosen) vs. the simpler "serialize sibling spawns / forbid concurrent spawn nodes." Reserve-and-refund is the default; flag for Ian if the refund bookkeeping is judged too heavy for v1, in which case fall back to forbidding concurrent sibling spawns.
- Pass a `RemainingBudget(nodes, llm_calls, depth)` + the deadline to the launcher (not the bare `max_llm_calls`).

**`ChildLauncher.__call__`** signature: replace `max_llm_calls: int | None` with `remaining: RemainingBudget | None` and `wall_deadline_monotonic: float | None` (kept optional **only** for structural/test callers; the production runner always passes them, and `resolve_effective_budget(is_child=True)` raises if `remaining` is missing — so a forgetful caller fails closed, never opens the budget).

**`make_child_launcher.launch`** (`subgraph.py:175-236`):
- Closes over the host `policy` and the **narrowed registry** (both now threaded from `build_supervisor`).
- After obtaining `spec` (planned or replay-loaded), call `resolve_effective_budget(policy, spec.budget, registry_tools=narrowed.tools, registry_subagents=narrowed.subagents, registry_kinds=narrowed.allowed_kinds(), remaining=remaining, is_child=True)`.
- **Clamp-below-floor:** `GraphBudget` is `ge=1` for nodes/depth. If `remaining.nodes == 0` or `remaining.depth < 1`, do **not** build an invalid budget — raise `SubgraphError("nest out of budget")` and fail closed.
- Stamp the resolved budget onto `spec.budget` via `model_copy` (recorded child shows granted limits), then `validate_graph_spec(spec, registry=narrowed, policy=policy)`. An oversized child now fails closed at validation.
- **Child metadata must propagate the whole parent envelope, not an allowlist** (RED-TEAM Bypass 4): build `child_metadata = {**parent_metadata, "graph_depth": depth+1, "parent_run_id": parent_run_id, "effective_budget": child_eff.as_dict(), "wall_deadline_monotonic": parent_deadline}` (plus `replay_of` when present). Passing the parent metadata through wholesale means `wall_deadline_monotonic` and lineage **cannot be dropped by a future one-line edit**. The runner must hand the parent metadata to the launcher for this (add it to the `ChildLauncher` call, or have the runner read it from `state` and pass it).

**`spawn_subgraph` under `parallel_map`:** structurally impossible (`child_kind` is `Literal["tool_call","llm_call"]`). **Keep the `Literal` closed** and add a regression test asserting a PM with `child_kind="spawn_subgraph"` fails Pydantic validation — do not add code, do not loosen the type.

Result: the child's `EffectiveBudget` is monotonically non-increasing down the nest; an N-level nest's total `nodes_executed`/`llm_calls_consumed` ≤ root effective budget; one deadline instant spans the tree; concurrent siblings cannot double-spend.

---

## 7. Default policy + back-compat

`ExecutionPolicy()` defaults mirror **today's effective behavior** so no-policy runs are governed identically:
- `max_nodes=12`, `max_llm_calls=8`, `max_depth=2`, `max_depth_ceiling=3` (== current `GraphBudget` defaults + `MAX_DEPTH_CEILING`).
- `allowed_tools=None`, `allowed_subagents=None`, `allowed_node_kinds=None` → intersection = full registry (`DEFAULT_TOOLS`, `DEFAULT_SUBAGENTS`, all `NodeKind`) — identical to today.
- New ceilings: `max_wall_seconds=90` (matches the never-enforced `GraphBudget` default — only bites pathological runs; turning on the dead field is the *intended*, documented behavior change), `max_fanout=64` (generous; caps a runaway 1000-item fan-out). **DECISION:** `max_fanout` default 64 vs 32 — default 64, flag for Ian.

Back-compat guarantees:
- `validate_graph_spec(spec)` keeps working: `policy` and `registry` default to `None`; when `None`, resolve against `ExecutionPolicy()` + the spec's own request + a default `Registry()` (root, `is_child=False`). Existing CLI/tests pass, now governed by host defaults (which equal the old self-granted defaults).
- `EngineConfig.policy` / `build_supervisor(policy=...)` / `LangGraphExecutor(policy=...)` default to `ExecutionPolicy()`.
- `GraphBudget` gains only optional `max_fanout` (Pydantic default 64); recorded specs round-trip; `recorder.load_validated_spec` tolerates old files (verified by test). The existing API readers of `spec.budget.max_wall_seconds` keep working and now see the *granted* value.
- `MAX_DEPTH_CEILING` re-exported from `validator.py` (sourced from `policy.py`) — `subgraph.py`'s import unchanged.

**Behavior-change to flag (DECISION):** stamping the resolved budget onto the returned/recorded `spec.budget` means any test asserting the planner's *original* budget survived validation will see clamped values. Intended (single source of truth); call it out and update affected tests.

---

## 8. Effective-limits introspection (requirement 4)

- `engine.config.policy` is directly inspectable.
- New `DynamicSubgraphs.effective_limits()` → `EffectiveBudget` resolved against an **empty/default request at root** (`is_child=False`, `remaining=None`), as a dict — the host ceilings a planning agent should self-limit to.
- `capabilities()` (engine.py:415, a `@classmethod`) — **DECISION:** `capabilities()` is a classmethod with no policy in scope. Either (a) make it also available as an instance method that includes `"policy": self._policy`-resolved ceilings, or (b) add an optional `policy=` param. Default: add an instance-level `effective_limits()` (above) and include `"policy"` only in the instance path; keep the classmethod unchanged to avoid signature churn. (Flag: surfacing host ceilings reveals governance config — acceptable for most threat models; confirm.)
- **Surface granted vs requested on results.** `SupervisorResult`/`SupervisorState` gain `effective_budget: dict | None` populated from the root graph's `state.metadata["effective_budget"]`; `RunResult` gains `effective_budget: dict | None` (from `result.effective_budget`), included in `to_dict()`. Both *requested* (`plan.budget`) and *granted* (`effective_budget`) are visible, so a planner can learn it over-asked. (Wire through `RunResult._from_supervisor`, which already reads `result.validated_spec`.)

---

## 9. Phased PR checklist (re-ordered so no window is open between PRs)

The red-team showed PR ordering matters: if executor rails (old PR-3) land before root-validate wiring (old PR-7), the root path trusts `spec.budget` in the interim (Bypass 2). New order keeps every intermediate state safe.

1. **PR-1 — policy object (no behavior change).** Add `app/policy.py` (`ExecutionPolicy`, `EffectiveBudget`, `RemainingBudget`, `resolve_effective_budget`, `MAX_DEPTH_CEILING`, `from_dict`/`as_dict`). `validator.py` imports + re-exports `MAX_DEPTH_CEILING`. Unit tests: resolution math, `is_child` raises without `remaining`, frozenset coercion, `None` vs `frozenset()`.
2. **PR-2 — static enforcement + narrowed registry + full wiring chokepoint.** `validate_graph_spec(spec, registry=None, *, policy=None)`: resolve, compare node/llm/depth vs effective, add `node_kind_not_in_policy`, stamp resolved budget onto returned spec. Thread `policy` + narrowed registry (tools ∩, subagents ∩) through `build_supervisor` → `build_supervisor_graph` (root validate) → `LangGraphExecutor(registry=, policy=)` → `make_child_launcher(registry=, policy=)`. **Land root + child validate wiring together** so neither path validates against the full vocabulary. Tests: 1000-node request rejected at 12; tool/subagent/kind intersection; `frozenset()` ban-all; child validates against narrowed registry.
3. **PR-3 — executor rails (off EffectiveBudget) + metadata seed.** `_recursion_limit_for(eff)`; `execute()` resolve-or-reuse; seed `effective_budget` + `wall_deadline_monotonic` into metadata. Tests: recursion tracks effective not spec.budget; child reuses parent's seeded budget.
4. **PR-4 — RUNTIME spend ledger (authoritative).** Fail-closed checks in `make_node_wrapper`, PM dispatcher (pre-Send projection), PM worker. Tests: PM over 60 items with `max_llm_calls=8` halts `BudgetExceeded`; N PM nodes can't multiply past effective; per-node node-count breach halts.
5. **PR-5 — fan-out cap.** Post-decode `len(items) > effective.max_fanout` → `FanoutExceeded` in dispatcher. Tests: list of N>cap halts fail-closed (incl. JSON-string `over`); N≤cap unaffected; empty list unaffected.
6. **PR-6 — wall-clock.** `check_deadline` in wrapper + PM dispatcher + PM worker; daemon invoke thread + `hard_timeout_s` join; `DeadlineExceeded` → `wall_timeout` (add to both status sets); resume carries original deadline. Tests: slow node trips between-node check; wedged node trips hard backstop; fan-out worker trips; resume doesn't reset deadline; daemon thread doesn't block exit.
7. **PR-7 — nested accounting (compose + reserve).** `resolve_effective_budget(is_child=True, remaining=)` in spawn runner + launcher; pessimistic reserve-and-refund via `counters`; child metadata propagated wholesale; clamp-below-floor fails closed; keep PM `child_kind` Literal closed. Tests: 3-level nest total spend ≤ root; concurrent sibling spawns can't double-spend; child requesting 1000 clamped to remaining; child can't regain a host-removed tool/subagent; host `max_depth=1` refuses spawn even though ceiling 3; remaining=0 fails closed; one shared deadline across levels.
8. **PR-8 — facade + introspection.** `EngineConfig.policy`; `run()` passes policy; re-export `ExecutionPolicy`; `effective_limits()`; `capabilities()` instance path; `RunResult.effective_budget` + `SupervisorResult`/state field; `GraphBudget.max_fanout` optional field + planner-prompt note + docs (wall-clock-as-latency-wall limitation, new defaults, granted-vs-requested).

---

## 10. pytest plan (including bypass-attempt tests)

Extend: `tests/test_validator.py`, `tests/test_executor.py`, `tests/test_parallel_map.py`, `tests/test_subgraph.py`, `tests/test_assembly.py`, `tests/test_sdk.py`, `tests/test_wrappers.py`. New: `tests/test_policy.py`.

**Resolution math (`test_policy.py`)** — `min()` per field; planner-asks-less honored, asks-more clamped; depth triple-min; `__post_init__` clamps host max_depth to ceiling and coerces sets to frozenset; intersection (`None`=full, subset=intersection, `frozenset()`=ban-all, host-named non-registry tool not granted); `is_child=True` without `remaining` raises; `remaining` drives child caps.

**Static bypass attempts** — 1000-node + 13 nodes rejected at 12; `max_llm_calls=1000` validated at 8; tool outside `allowed_tools` → `tool_not_allowlisted`; subagent outside `allowed_subagents` → `subagent_not_allowlisted`; kind outside `allowed_node_kinds` → `node_kind_not_in_policy`; **grep-guard test**: assert no enforcement path reads `spec.budget.*` for node/llm/depth (regression: huge `spec.budget` + small effective ⇒ rejection).

**RUNTIME ledger bypass attempts (the core fix)** — PM over 60-item list, `max_llm_calls=8` → `BudgetExceeded`, no downstream nodes run; many PM nodes can't collectively exceed effective; `spawn_subagent`-only plan with `max_llm_calls` small halts at runtime even though it passes the static +1 floor; `reduce/llm_summarize` likewise; per-node node-count ceiling halts a wide graph.

**Runtime rails** — fan-out 100 items, cap 64 → `FanoutExceeded` (incl. JSON-string `over` decoded to >cap); recursion limit derives from effective not spec.budget; slow node → cooperative `wall_timeout`; wedged node → hard `join` backstop `ok=False`; PM worker honors deadline; daemon thread doesn't block process exit; `resume()` doesn't inherit/reset to a fresh deadline.

**Nested bypass attempts** — 3-level nest: total `nodes_executed` ≤ root effective `max_nodes`, total `llm_calls_consumed` ≤ root effective `max_llm_calls`; **two concurrent sibling spawn nodes cannot both claim full remaining** (reserve-and-refund); child requesting `max_nodes=1000` clamped to parent remaining and fails closed; child naming a host-removed tool/subagent rejected; host `max_depth=1` refuses spawn at depth 1; remaining=0 fails closed (no `ge=1` Pydantic crash); one shared deadline instant across all levels (assert metadata propagation); child launcher built without `remaining` fails closed (not full-budget).

**Back-compat** — no-policy run reproduces today's behavior (`test_e2e_pipeline`, `test_sdk` unchanged); `validate_graph_spec(spec)` with no policy/registry passes a valid default spec; old recorded spec without `max_fanout` round-trips through `recorder.load_validated_spec` and replays; API `max_wall_seconds` readers (`serialize.py`, `routers/runs.py`) see the granted value.

---

## 11. Bypass vectors and exactly how each is closed

1. **Static-only node/llm budget (fan-out × depth multiplication; Bypass 1 CRITICAL).** Closed by the **authoritative runtime spend-ledger check** (§4) in the wrapper, PM dispatcher (pre-Send projection of `len(items)`), and PM worker. The summed `counters` make the projection correct; static `count_llm_calls` is demoted to a fast-fail floor.
2. **`recursion_limit` from `spec.budget` in root/child path (Bypass 2 HIGH).** Closed by `_recursion_limit_for(EffectiveBudget)` + executor resolve-or-reuse, and by landing root-validate policy wiring **in the same PR** as static enforcement (PR-2), so no interim trusts `spec.budget`.
3. **Nested non-composition / re-mint / sibling TOCTOU (Bypass 3 CRITICAL).** Closed by mandatory child resolution (`is_child=True` raises without `remaining` — no full-budget fallback), composing all fields against parent remaining, **pessimistic reserve-and-refund via summed `counters`** (never `metadata`), and clamp-below-floor failing closed.
4. **Wall-clock can't preempt workers / nested reset (Bypass 4 HIGH).** Cooperative checks at **every** node-entry point including PM dispatcher and worker; daemon thread + bounded `join`; parent metadata propagated wholesale so the deadline can't be dropped; `resume()` carries the original deadline. Documented: wall bounds latency, the **runtime ledger** is the spend bound.
5. **`allowed_tools` evaded via subagents / un-narrowed child registry (Bypass 5 HIGH).** Closed by adding `allowed_subagents` (intersected into the narrowed registry's `subagents=`), threading the narrowed registry into **both** `LangGraphExecutor(registry=)` and `make_child_launcher(registry=)` (today both default to `None`), and enforcing the llm budget by the **runtime ledger** (so `reduce/llm_summarize` and `spawn_subagent` LLM calls count regardless of `allowed_node_kinds`).
6. **`None` vs `frozenset()` / plain-set storage (Bypass 6 MED).** `None`≠`frozenset()` contract enforced; `__post_init__` coerces allow-sets to `frozenset`; unit-tested.
7. **Fan-out check ordering / JSON-string `over` (Bypass 7 LOW/MED).** Fan-out check placed **after** the JSON decode (line ~106), before the Send loop; `>` comparison leaves the empty-list shortcut intact; regression test with a JSON-string `over` of >cap items.
8. **Deadline-not-inherited (Bypass 4/8).** Single executor "reuse-if-present else mint" chokepoint + wholesale parent-metadata propagation; tested for one shared instant across the nest.

---

## 12. DECISIONS still needing Ian

- **DECISION (max_fanout planner schema):** add optional `GraphBudget.max_fanout` (planner-facing + prompt note) — spec assumes yes; alternative is host-only. 
- **DECISION (sibling-spawn TOCTOU):** reserve-and-refund (chosen, capability-preserving) vs. forbid/serialize concurrent sibling spawn nodes (simpler). Fall back to forbid if refund bookkeeping is too heavy for v1.
- **DECISION (resume deadline):** carry-original (chosen) vs. bounded re-grant of a fresh window.
- **DECISION (capabilities exposure):** include resolved host ceilings in an instance-level `capabilities()`/`effective_limits()` — confirm acceptable for the threat model (reveals governance config).
- **DECISION (clamp note):** whether the validator emits a non-fatal note when `min()` actually clamped a request, so the planner learns it over-asked.
- **DECISION (defaults):** `max_fanout=64` vs `32`; `max_wall_seconds=90` is the documented turn-on of the dead field.
- **DECISION (stamped budget):** confirm stamping the resolved budget onto the recorded `spec.budget` is desired (single source of truth + correct API `max_wall_seconds` reporting) despite changing any test that asserts the planner's original budget survived.

---

### Relevant files (all absolute)
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\policy.py` (NEW)
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\models\graph_spec.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\registry\validator.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\registry\registry.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\runtime\executor.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\runtime\parallel_map.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\runtime\subgraph.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\runtime\wrappers.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\models\run_state.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\assembly.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\supervisor\graph.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\supervisor\supervisor.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\supervisor\state.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\app\api\serialize.py` / `app\api\routers\runs.py` (existing `max_wall_seconds` readers)
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\dynamic_subgraphs\engine.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\dynamic_subgraphs\__init__.py`
- `C:\Users\praht\OneDrive\Desktop\Projects\Dynamic-Subgraphs\dynamic_subgraphs\types.py`
