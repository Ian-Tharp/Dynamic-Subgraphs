> **Status:** Design — proposed, not yet implemented. Source of truth for Slice 7.
> **Provenance:** Produced by a 7-agent design workflow (4 design lenses → synthesis →
> adversarial red-team → reconciliation), each grounded in the actual source. The
> red-team's three required fixes are applied inline (see the end of the doc).
> **Blocking:** Section 10 lists `DECISION:` points that need Ian before the tagged PRs.

# Slice 7 — Eval / Value Layer for Dynamic Subgraphs (FINAL)

An **opt-in scorer** that grades a completed run into a persisted, queryable `EvalResult`, turning the write-only `runs/` corpus into comparable memory and enabling the one benchmark that decides the dynamic-topology thesis: **does an LLM-invented topology beat a hand-authored baseline on cost-adjusted quality across a heterogeneous batch?**

This is the post-red-team revision. Every fatal finding is resolved inline and tagged **RESOLVED**. Decisions still requiring Ian are tagged **DECISION:** (the script/Notion reader should treat those as blocking before implementation of the affected PR).

## 0. Decisions & scope (locked 2026-06-03)

Ratified with Ian after the red-team. These override the body where they conflict.

**Locked decisions (of the 8 in §10):**
- **Baseline = fixed-library router (Arm B) is the verdict baseline**; single linear graph (Arm C) is sanity-only. ✓
- **Pre-registration thresholds accepted**: `quality_floor 0.6`, `judge_cost_floor $0.005`, `judge_budget = max(floor, 0.15·run_cost)`, rank on `value_per_ktok`, `value_per_usd` diagnostic-only. ✓
- **Scope = SDK-first**: score the `DynamicSubgraphs` engine path only this slice. **Defer** app-layer (CLI/HTTP) scoring → **drops PR2.5 and PR9** from Slice 7; document that `runs/` produced via the API are unscored until a later slice. ✓
- **Decision rule (§7) intentionally NOT yet locked** — it gates only PR8 (the benchmark, the last PR). We pre-register it right before the benchmark, once real eval scores exist. Better pre-registration discipline.

**Layer boundary — DS vs CORE (new, from the stack architecture).** Dependencies point downward (GPW → CORE → **DS**); DS is the substrate CORE *uses*, so DS cannot depend upward on CORE for evaluation. Therefore:
- **DS owns structural / operational governance eval** — *is this a valid, parsimonious, on-budget, successfully-executed, tool-grounded graph/trajectory?* This is intrinsic to being a governed substrate ("are we building correct graphs") and must live in DS. The deterministic gate **is** this.
- **CORE owns semantic / cognitive eval** — *was this a good answer; did the C→O→R→E loop produce value?* Higher layer.
- **Implication:** DS's eval is **deterministic/structural-first**. The LLM-judge `goal_completion` is a **bridge** (useful for the benchmark now), explicitly *not* a permanent DS responsibility — its long-term home is CORE. Mark it as such; don't let DS grow a semantic-quality mandate it shouldn't own.

**Scope adjustment — module placement & a layering fix (supersedes the body's `app/eval/`).** Because scope is SDK-first, the eval layer lives in **`dynamic_subgraphs/eval/`** (the facade), not `app/eval/`. The facade may import from `app`; `app` must not import the facade. **Caveat for PR1:** `TokenUsage` currently lives in `dynamic_subgraphs/engine.py`, and `engine.py` will import the eval module (for `EngineConfig.eval_gate`) → a cycle if the eval module imports `TokenUsage` from `engine.py`. **PR1 must first relocate `TokenUsage`** into a leaf module (`dynamic_subgraphs/usage.py`) and re-export it from `engine.py` for back-compat, so both `engine` and `eval` import it without a cycle.

**PR decomposition correction (learned during PR1 impl, 2026-06-03).** The repo enforces an **`Artifact` ⇔ recorder-producer ⇔ `docs/recipes.md`** invariant via guard tests (`tests/test_sdk.py::test_artifact_values_match_recorder_filenames` / `test_every_artifact_is_documented_in_recipes`). So the recording vocab (`Artifact.EVAL`, `Recording.evaluated()`, the `"evaluated"` capabilities preset) **cannot** ship in a types-only PR1 — it must land *atomically with its producer and docs*. Revised split: **PR1 = eval types + `TokenUsage` relocation only** (shipped); the vocab moves into the **recorder-integration PR** alongside `_EvalProducer` + the `recipes.md` entry. Also: because the eval types live in the facade (`dynamic_subgraphs/eval`) and `app/recording` must not import the facade, that producer takes the score as a **pre-serialized mapping** (or a duck-typed `.model_dump()`), not a hard `EvalResult` import.

**Adjacent feature — developer prompt overrides (separate slice, captured here so it isn't lost).** Devs should be able to customize the prompts that drive *spec generation* (and the eval rubric). The hook already exists internally: `LLMPlanner.__init__(system_prompt=...)` (`app/supervisor/llm_planner.py:223`) defaults to `PLANNER_SYSTEM_PROMPT.format(...)`. The work is to **surface it through `EngineConfig`** without letting a raw override break the GraphSpec contract:
- The planner prompt is a **contract + guidance** template: a non-negotiable structural part (emit a valid `GraphSpec` over the frozen vocabulary, schema, budgets — fills the `.format()` slots) plus a customizable guidance part (domain steering, e.g. "prefer `parallel_map` over deep recursion for worldbuilding").
- Expose **guidance-level overrides** (appended/templated) by default; allow a full-replacement escape hatch that is clearly marked as "you now own keeping plans valid."
- The **eval LLM-judge rubric is the same mechanism** — its prompt should be overridable too, so prompt-customization and the eval layer share one `PromptOverrides`/registry seam.
- Optional: version overrides via the **LangSmith prompt hub** (`get_prompt_by_name`/`push_prompt`) for teams that manage prompts there.
- **Does it violate governance? No.** Overriding *prompts* doesn't touch "plans not code" or the frozen vocabulary — the planner still emits a validated `GraphSpec`. It's on-thesis (developer-friendly, model-agnostic SDK). Spec it as its own slice after the eval foundation.

**Added PR — LangSmith eval-set / regression (new, from Ian's idea).** After the corpus reader (PR6), add a PR that promotes scored runs into a **LangSmith dataset** (`create_dataset`/`list_examples`) keyed on `(prompt → graphspec shape / trajectory / reference / score)`, and uses `run_experiment` as a **graph-correctness regression suite** over live model calls: change the planner/registry, re-run the set, detect graphspec/trajectory/score drift. Local `eval.json` is the score; LangSmith is the versioned eval set + regression guard. Cross-check local cost vs LangSmith cost (same method as `model-comparison-2026-06.md`).

---

### Code grounding (verified by reading source, not the draft)

- `app/models/node_kinds.py` — 9 `NodeKind`s (`llm_call, tool_call, spawn_subagent, parallel_map, reduce, branch, wait_for_event, emit_artifact, spawn_subgraph`). `app/models/graph_spec.py` — `GraphSpec/NodeSpec/EdgeSpec/GraphBudget` (budget: `max_nodes=12, max_depth=2, max_wall_seconds=90, max_llm_calls=8`; `NodeSpec.params` is an open `dict[str, Any]` and the real behavior lives in `params["instruction"]`). `app/models/trace.py:17` — `TraceEventKind.EVAL_RESULT` exists and is unused.
- `app/supervisor/iteration.py` — `LlmIterationDecider` (acceptance gate; `_IterationDecisionPayload` closed shape; `build_provider_iteration_decider(model_provider, model_ref, ...)` calls `model_provider.build_structured_output(model_ref, _IterationDecisionPayload)`; defers on `{paused, plan_failed, validation_failed, compile_failed, record_failed, resume_failed, replay_failed}`; catches every model failure into a safe default). The eval layer mirrors this plumbing but **must not import or call it**.
- `app/recording/recorder.py` — `ArtifactProducer` protocol (filename-selected via `FileRecorder._selects`), `ArtifactContext` (frozen: `run_id, directory, spec, result, prompt`), `DEFAULT_PRODUCERS` (spec, trace, output, mermaid, prompt, summary), `FileRecorder` (`record`, `load_validated_spec`, `load_output`, `list_runs`), `NullRecorder`. `_extract_output` deliberately strips events from `output.json`.
- `app/runtime/tools.py` — **only `web_search` is genuinely grounded**. `policy_lookup` (introspects allowlists), `mock_document_extract` (string stats), `create_follow_up_task` (`created=False`) are echo/no-op. The model-comparison doc confirms `mock_document_extract` was removed from `DEFAULT_TOOLS`; the other two remain echo placeholders. **This is the root of red-team A1/B3.**
- `app/runtime/model_providers.py` — `ModelRef` carries a `temperature: float | None` field and `structured_method`. `build_structured_output(ref, schema)` takes **only ref + schema** (no temperature/sampling kwargs) — temperature must be set on the `ModelRef`. `OpenAIModelProvider` forces `method="function_calling"` by default (which on the OpenAI path forces `tool_choice` and ignores temperature for some models).
- `dynamic_subgraphs/engine.py` — `EngineConfig` (frozen), `RunResult` (frozen-ish dataclass), `TokenUsage.from_handler` (exact, with `by_model`), `_compute_cost` (LiteLLM/manual pricing → `float | None`), `capabilities()`. **`usage`/`cost` are computed at engine.py:551-552, AFTER `supervisor.run()`** — the supervisor graph never sees them. `RunResult._from_supervisor` reads `result.validated_spec` (None on plan/validation failure), `result.result` (ExecutionResult or None).
- `dynamic_subgraphs/recording.py` — `Artifact` StrEnum (values = filenames), `RECORDER_ARTIFACTS = frozenset(Artifact) - {EMITTED}`, **`Recording.all() == frozenset(Artifact)` (so a new enum member is auto-included by `all()`/`record=True`)**, `Recording.replayable() == {SPEC, OUTPUT}` (NOT `{SPEC, TRACE, OUTPUT}` as the draft claimed). `capabilities()["recording_presets"]` is a **hardcoded list literal** in engine.py — adding a preset requires editing it.
- `app/assembly.py` — `RunConfig` already carries `judge_model`/`judge_ref`. `_build_planner` **hardwires `StaticPlanner(build_demo_spec())` and ignores any caller spec** (the benchmark fixed-arm gap). `_attach_callbacks(ref, callbacks)` merges callbacks into `ModelRef.extra_kwargs` — the seam by which the usage handler reaches the executor's worker thread. The usage handler is attached to **planner, worker, reducer, subagent** refs, so the dynamic arm's `RunResult.cost` already includes planner tokens (relevant to A4).
- `app/supervisor/supervisor.py` — `SupervisorResult(run_id, status, response, validated_spec, result, record, errors)` has **no usage/cost field** (relevant to C1). `app/supervisor/graph.py` — topology `START -> receive -> plan -> validate -> execute -> record -> respond -> END`; `_make_record_node` no-ops when `validated_spec`/`result` is None and catches record failures into `record_failed`.
- `app/runtime/executor.py` — `ExecutionResult(state, trace, ok, error, paused, interrupt_payloads)`.
- `app/registry/validator.py` — `validate_graph_spec(spec, registry=None) -> GraphSpec` (returns normalized copy; raises `RegistryValidationError`). `MAX_DEPTH_CEILING = 3`.
- `docs/evals/model-comparison-2026-06.md` — real costs: compare/nano **$0.0010**, summarize/nano **$0.0011**, research/haiku **$0.0786**. nano 3-node vs haiku 6-node = ~23× on the same task. Confirms both the near-zero-cost regime (A2/B2) and the planner-driven topology variance (A5).

---

## 1. Goal & non-goals

**Goal.** Score a *completed* run `(GraphSpec, ExecutionResult, TokenUsage, cost)` into a typed `EvalResult`, persist it to `runs/<id>/eval.json`, surface it on `RunResult.eval`, and make the corpus groupable by task/topology/origin so a **paired, cost-adjusted** quality comparison is computable. OFF by default; zero cost and zero behavior change when unused.

**Non-goals (explicit):**
- **Not an acceptance gate.** Never returns stop/replan/ask_user/fail; never alters control flow. `EvalResult.overall_verdict` is advisory metadata only. The gate module **must not import `app/supervisor/iteration.py`** (hard architectural boundary — red-team C4).
- **Not inside the meta-loop.** The score does not feed `run_iteratively` in Slice 7. Re-importing the score into the loop re-imports the circularity/cost problems; deferred.
- **Not a new model role.** Reuse the existing `judge_model`/`judge_ref` slot.
- **Not a circular LLM-grades-LLM default.** The default gate is fully deterministic and token-free. The LLM judge is opt-in, reference-anchored, and budget-floored.
- **Not a free-form preference judge for the verdict.** Where an LLM judge contributes to the *thesis verdict*, it scores against a reference checklist with verbosity controls (red-team B4), never "is this good?".

---

## 2. EvalGate interface + EvalResult

**Where scoring runs (resolved).** `usage`/`cost` exist only in the engine (engine.py:551-552). So the gate is invoked in `DynamicSubgraphs.run()` after `supervisor.run()` and after cost is computed; persistence happens via a recorder helper that re-opens the run dir.

**RESOLVED (C1 — app-layer corpus gap).** The shipped CLI/HTTP corpus is generated through `build_supervisor`/`Supervisor.run`, not the SDK engine, and `SupervisorResult` carries no economics. To avoid an SDK-only memory layer, **PR2.5 surfaces aggregated usage on `SupervisorResult`** by threading the `UsageMetadataCallbackHandler` (already attached in `app/assembly.py`) up to the supervisor result. This is a small additive change (a `usage: TokenUsage | None` field, populated by the engine/assembly that owns the handler). The supervisor-node hook (Section 6) then scores with **real token economics**; cost is still `None` unless a price book is threaded, so `value_per_ktok` is the app-layer-stable axis (Section 3). **DECISION:** confirm we want the app-layer (CLI/API) corpus scored in Slice 7 vs. deferring app-layer scoring to a later slice. If deferred, document that `runs/` from the API is unscored until then.

**RESOLVED (C2 — re-open races / `record=True` semantics).** Two concrete rules:
1. `record_eval` is only ever called when the active recorder is a `FileRecorder` **and** `Artifact.EVAL` is in the recorder selection. With a `NullRecorder` (the SDK default `recording=False`) the gate still runs and the result is returned on `RunResult.eval`, but **no `eval.json` is written** — documented, not silent.
2. We **accept** that `Recording.all()` (`== record=True`) will include `Artifact.EVAL` once the member is added to the enum. This is harmless when no gate is configured (the producer's `applies()` returns False → no file). The draft's "keep EVAL out of all()" claim is **dropped** as impossible without special-casing the enum. The §9 test asserts this actual behavior.

New module `app/eval/` (slim, mypy-strict, lazy LLM imports — mirrors `iteration.py`). Pydantic closed shapes for the LLM payload; frozen dataclasses for context/result seams.

```python
# app/eval/types.py
EVAL_SCHEMA_VERSION = 1

@dataclass(frozen=True)
class EvalContext:
    run_id: str
    spec: GraphSpec                 # result.validated_spec
    result: ExecutionResult         # .state/.trace/.ok/.error/.paused
    prompt: str | None
    status: RunStatus               # outer status at score time
    usage: TokenUsage               # exact tokens, by_model
    cost: float | None              # USD or None
    origin: Literal["invented", "authored", "routed"] = "invented"  # benchmark arm
    task_id: str | None = None      # logical task; groups arms of the same task
    reference: EvalReference | None = None   # per-task gold/checklist (mandatory for verdict)
    planner_model_name: str | None = None    # to split planner cost out of by_model (A4)

class EvalGate(Protocol):
    name: str
    def evaluate(self, context: EvalContext) -> EvalResult: ...
```

`EvalResult` is the persisted unit: a four-dimension rubric + a flat `quality` aggregate + a denormalized `RunFingerprint` + economics. **RESOLVED (C3/C5):** `eval.json` persists only `EvalResult` (scores + fingerprint + economics) — it **never** re-embeds `state.values` or the trace. `signals` is constrained to `dict[str, float]` (B-finding C5: avoid the `float|int|str|bool` union that breaks round-trip type-stability); non-numeric provenance goes in typed string fields, not `signals`.

```python
# app/eval/types.py  (pydantic for clean JSON round-trip)
class ScoreComponent(BaseModel):
    dimension: Literal["plan_validity","grounding","goal_completion","cost_efficiency"]
    score: float = Field(ge=0.0, le=1.0)
    verdict: Literal["pass","weak","fail","skipped","neutral"]   # "neutral" added (A1)
    method: Literal["deterministic","heuristic","llm_judge","reference"]
    applicable: bool = True            # False => contributed as a shared neutral (A1)
    rationale: str = ""
    signals: dict[str, float] = Field(default_factory=dict)   # numeric only (C5)

class RunFingerprint(BaseModel):
    graph_id: str
    goal: str
    prompt_sha256: str
    node_count: int
    edge_count: int
    max_depth: int
    node_kind_histogram: dict[str, int]
    has_grounded_tool: bool            # True iff a web_search tool_call ran (A1 pivot)
    planner_model: str | None
    worker_model: str | None
    topology_signature: str            # see §3 (shape-aware, not multiset-only) (B5)
    instruction_sha256: str            # separate hash over params["instruction"] (B5)
    origin: Literal["invented","authored","routed"]

class EvalResult(BaseModel):
    schema_version: Literal[1] = EVAL_SCHEMA_VERSION
    run_id: str
    scored_at: datetime
    gate: str                          # e.g. "deterministic@v1" | "rubric-llm:claude-...@v1"
    rubric_id: str = "ds.default.v1"   # pins comparability across runs
    weights: dict[str, float]          # persisted so re-weighting is reproducible
    applicable_dimensions: list[str]   # the exact dims that counted (A1 — pairing key)
    quality: float = Field(ge=0.0, le=1.0)
    components: list[ScoreComponent]
    overall_verdict: Literal["pass","weak","fail","skipped"]
    # economics (copied from context — exact, no recompute):
    cost_usd: float | None
    planner_cost_usd: float | None     # split out of by_model (A4)
    worker_cost_usd: float | None      # cost_usd - planner_cost_usd when both known
    total_tokens: int
    # value axes (see §3 for which one decides the verdict):
    value_per_ktok: float | None       # quality / (total_tokens/1000) — provider-stable PRIMARY
    value_per_usd: float | None        # quality / cost_usd — DIAGNOSTIC ONLY, never the verdict
    quality_floor_met: bool            # quality >= rubric quality floor (gate for Pareto) (A2)
    fingerprint: RunFingerprint
    # determinism / circularity guards:
    deterministic: bool                # True iff no LLM judge fired
    judge_model: str | None = None
    judge_cost: float | None = None
    judge_fired: bool = False          # for judge-coverage reporting (B2)
    errors: list[dict[str, Any]] = Field(default_factory=list)
```

**No overlap with `LlmIterationDecider`:** the decider runs *between* runs in `run_iteratively`, returns an action, mutates the next prompt, is never persisted. The gate runs *once after* a single run, returns a persisted number, is inert to control flow. They share only the `judge_ref` role and the structured-output discipline.

---

## 3. Scoring rubric & gate types (circularity, pairing, and metric fixes)

Four dimensions. **Three are deterministic and free**; `goal_completion` needs a reference or a judge. The judge is gated *behind* the deterministic dimensions.

| Dimension | Method | What it measures | Source (verified present) |
|---|---|---|---|
| `plan_validity` | deterministic | re-run `validate_graph_spec(spec)` + **parsimony** (node/edge/depth vs goal) + budget adherence vs `GraphBudget` | `spec`, `result.trace.events` |
| `grounding` | deterministic / **shared-neutral** | claims/numbers in `state.values` supported by `web_search` outputs in trace | `result.state.values`, trace tool events |
| `goal_completion` | **reference** or `llm_judge` (never bare keyword for the verdict) | did the output address the goal | `prompt`, `state.values`, `reference` |
| `cost_efficiency` | deterministic (derived) | quality-per-token / quality-per-cost | `usage`, `cost` |

### RESOLVED A1 — grounding must be *paired-comparable*, not per-run-renormalized
Skip-and-renormalize is correct for **single-run display** but a **confound for paired comparison**: the always-`web_search` fixed arm is always grounded, while pure-`llm_call` dynamic runs are not, so the two arms get different denominators on the same task. Fix:
- The **applicable-dimension set is computed per `task_id` and shared across all arms of that task**. The benchmark harness pre-computes it (a task that requires grounding marks `grounding` applicable for *every* arm).
- When grounding is inapplicable to a run but applicable to the task, it is scored as a **shared neutral constant** (`verdict="neutral"`, `applicable=True`, fixed `score` from the rubric, identical for both arms) — it does **not** drop out and does **not** renormalize the denominator.
- `EvalResult.applicable_dimensions` is persisted; the corpus reader **asserts both arms of a paired comparison share the identical `applicable_dimensions` set, or marks the pair `incomparable`** and excludes it from the verdict.
- For standalone single-run scoring (non-benchmark), the old skip-and-renormalize remains available and is the default; pairing-mode is a flag the benchmark harness sets. **DECISION:** confirm the shared-neutral constant value (proposal: `0.5`, i.e. "unknown, neither rewarded nor punished") and that grounding-required is a per-task property authored in the task pack.

### RESOLVED A2/B2/Q2 — the verdict is NOT decided on raw `value_per_usd`
Real runs hit **$0.001**, so `quality/cost` is unbounded as cost→0 and a cheap-mediocre run (quality 0.3 @ $0.0003 → 1000) beats a genuinely better one (0.9 @ $0.01 → 90). That **inverts** the thesis. Resolution:
- **`value_per_usd` is DIAGNOSTIC ONLY** — persisted and reported, never the decision variable.
- The headline value axis is **`value_per_ktok = quality / (total_tokens/1000)`** (provider-pricing-stable, never near-zero-divide because every real run spends tokens). This is also the only axis available app-layer when cost is `None`.
- The **thesis verdict uses quality-at-a-cost-ceiling (Pareto with a quality floor)**, not any ratio (red-team's required fix #1):
  - A run is **eligible to "win" only if `quality >= quality_floor`** (`quality_floor_met=True`). Below the floor it cannot win regardless of cheapness.
  - Among eligible runs, the paired comparison is: for each task, **does the dynamic arm Pareto-dominate or match the baseline** — i.e. `quality_dyn >= quality_base AND cost_dyn <= cost_base`, OR strictly better on one without being worse on the other.
- A pre-registered **penalty form** `adjusted = quality - λ·(cost_usd or tokens-normalized)` is reported as a secondary cut for sensitivity. **DECISION:** ratify `quality_floor` (proposal `0.6`), the value axis used for ranking ties (`value_per_ktok`), and λ for the penalty cut, **before** running (pre-registration prevents post-hoc rationalizing).

### RESOLVED B3/B4/Q4 — goal_completion cannot be both circularity-free AND headline-weighted on a keyword proxy
Deterministic keyword/substring coverage is gameable and doesn't measure goal completion; the planner emits node instructions and can parrot goal keywords into outputs. And the "judge bias cancels in the paired delta" claim is **false for arm-correlated bias** (verbosity preference correlates with arm shape). Resolution:
- The **deterministic gate explicitly cannot adjudicate the thesis on `goal_completion`.** It scores `plan_validity`, `grounding`, `cost_efficiency` credibly; for `goal_completion` it emits a `reference`-method component **only when an `EvalReference` exists** (checklist coverage with synonym/embedding matching), else a clearly-labeled low-confidence `heuristic` component that is **excluded from the thesis headline weight**.
- **Reference packs are MANDATORY for the thesis verdict** (Open Q4 = yes). Each benchmark task ships an `EvalReference` (`checklist`, `must_include`, `must_not_include`, optional `gold_answer`). Goal-completion is scored as **coverage of required reference points minus unsupported-claim penalties**, not free-form preference.
- When an LLM judge contributes to the verdict it must: (a) score against the reference checklist, (b) run with **output length normalized/capped before judging**, (c) report a **verbosity-vs-score correlation** as a bias diagnostic in `signals`, (d) use a **cross-model judge** (`judge_model != worker_model`, warn if equal).
- **Two headline numbers are reported separately**, never blended into one ambiguous claim: (1) **structural/cost quality** (deterministic: plan_validity + grounding + cost_efficiency) as the circularity-free headline; (2) **reference-anchored goal_completion** as the goal headline, with judge-coverage stated. **DECISION:** confirm the default weight split. Proposal for `ds.default.v1`: `goal_completion 0.40, grounding 0.25, plan_validity 0.20, cost_efficiency 0.15`, BUT the thesis verdict is reported per-headline (structural vs goal) rather than on the blended `quality`, so the 0.40 weight never silently rides on a keyword proxy.

### RESOLVED B1 — determinism/ensemble must match the real provider seam
`build_structured_output(ref, schema)` exposes no temperature/sampling args. Resolution:
- Temperature is set by constructing the judge `ModelRef(temperature=0.0, ...)` (the field exists). Documented as **best-effort**: OpenAI's `function_calling` path ignores temperature for some models, so LLM-judged rows are **NOT bit-reproducible** regardless. Only `deterministic=True` rows are byte-stable.
- **`ensemble=N` is DROPPED from the default** (N separate `.invoke()` calls collide with the budget floor and add nondeterminism it can't fully remove). It may be exposed as an opt-in advanced flag, off by default, never used for the headline.

### RESOLVED B2 — judge budget is a floor + fraction, with coverage reporting
`judge_budget_fraction=0.15` of a $0.001 run is $0.00015 — smaller than any real structured call, so the judge is *always* skipped on the cheap runs that decide the verdict, biasing the LLM-judged sample toward expensive topologies. Resolution:
- Cap is **`judge_budget = max(judge_cost_floor, judge_budget_fraction * run_cost)`** with `judge_cost_floor` default `$0.005` (enough for one structured call on a multi-thousand-token output). **DECISION:** ratify `judge_cost_floor` and `judge_budget_fraction`.
- The benchmark **reports judge coverage** (fraction of runs actually LLM-judged vs heuristic). If coverage `< 100%`, the LLM-judged cut is flagged non-comparable across arms and the verdict falls back to the deterministic + reference headlines.

### RESOLVED B5 — `topology_signature` must be shape-aware, with a separate instruction hash
Multiset-of-kinds + kind-pair edges **over-merges** (strips `params["instruction"]`, so "summarize X" and "write adversarial critique of Y" collide) and **fragments unpredictably** (a `branch` DAG vs a linear chain of the same kinds). Resolution:
- `topology_signature` = sha256 over a **canonical topological shape**: the directed edge structure (kind-labeled, with in/out-degree per node and a stable topological ordering), **not** just a sorted multiset. `params`/ids stripped.
- `instruction_sha256` (separate field) = sha256 over the concatenated, normalized `params["instruction"]` values in topological order.
- Promotion (the eventual template library) requires **both** `topology_signature` match **and** `instruction_sha256` similarity — never topology alone. **DECISION:** confirm the equivalence class for promotion (shape-only merge vs shape+coarse-tool-bucket vs shape+instruction-similarity threshold). Proposal: shape match for corpus grouping; shape + instruction-embedding-similarity ≥ threshold for promotion.

### Gate types (factories mirror `build_provider_iteration_decider`)
- **`DeterministicEvalGate`** — `name="deterministic@v1"`, `deterministic=True`, zero tokens. The default and the gate the thesis headline uses to stay credible. Emits `plan_validity`, `grounding`, `cost_efficiency` credibly, and `goal_completion` only via `reference` (or a labeled low-confidence heuristic excluded from the verdict headline).
- **`LlmEvalGate`** — opt-in. Only `goal_completion` calls the judge, via `model_provider.build_structured_output(judge_ref, _GoalScorePayload)`. `_GoalScorePayload(covered: list[str], missing: list[str], unsupported_claims: list[str], score: float[0..1], rationale: str)` — reference-anchored, closed shape. Catches every model failure into a `ScoreComponent(verdict="fail"/"skipped", method="llm_judge")` so a flaky judge never crashes a recorded run (copied from `LlmIterationDecider`).

**Failed/paused runs.** Non-ok statuses short-circuit to a deterministic `fail`/`skipped` `EvalResult` with no judge call (mirrors the decider deferring on non-ok statuses). A dynamic-arm `validation_failed`/`plan_failed` counts as `quality=0`, `quality_floor_met=False` — robustness is part of the thesis.

---

## 4. eval.json data model + recorder integration

**Artifact vocab (`dynamic_subgraphs/recording.py`):**
- Add `EVAL = "eval.json"` to `Artifact`. It auto-joins `RECORDER_ARTIFACTS` and (per C2) is **included by `all()`/`record=True`** — accepted and tested.
- Add preset `Recording.evaluated()` = `{SPEC, TRACE, OUTPUT, EVAL}`.
- Add `"evaluated"` to the **hardcoded** `capabilities()["recording_presets"]` list in engine.py (explicit edit — it is not derived).

**Producer (`app/recording/recorder.py`):**
- Add `eval_result: EvalResult | None = None` to `ArtifactContext` (frozen, defaulted — additive; existing call sites still compile).
- Add `_EvalProducer` (`key="eval"`, `filename="eval.json"`), `applies()` True only when `ctx.eval_result is not None`. Append to `DEFAULT_PRODUCERS` after `_SummaryProducer`.
- Add `FileRecorder.record_eval(run_id, eval_result)` — re-opens `runs/<id>/`, writes `eval.json` via the producer (scoring happens after `record`). Only called when recorder is a `FileRecorder` and `Artifact.EVAL` is selected (C2).
- Add `FileRecorder.load_eval(run_id) -> EvalResult | None` (mirrors `load_output`; returns None for old dirs — backward compatible).
- Extend `list_runs()` summaries with `quality`/`value_per_ktok`/`origin`/`deterministic`/`quality_floor_met` pulled from `eval.json` when present; old dirs omit the fields.

**Why denormalized.** The embedded `RunFingerprint` answers "invented vs authored on cost-adjusted quality" purely from `runs/*/eval.json`. `prompt_sha256` groups same-task runs; `topology_signature` is the motif key; `origin` is the headline pivot; `has_grounded_tool` makes A1's pairing auditable.

**`EVAL_RESULT` trace event.** The reserved `TraceEventKind.EVAL_RESULT` is emitted as a `TraceEvent` carrying the composite + per-dimension verdicts, **embedded inside `eval.json`** (and optionally appended to a sibling `trace.eval.jsonl`), since the original `trace.jsonl` is already written at score time. It carries **no raw state/trace** (C3).

**Corpus reader (`dynamic_subgraphs/corpus.py`, new):**
```python
class EvalCorpus:
    def __init__(self, root_dir: Path = Path("runs")): ...
    def load(self, run_id: str) -> EvalResult | None: ...
    def all(self) -> Iterator[EvalResult]: ...                 # globs runs/*/eval.json, skips Nones
    def by_task(self) -> dict[str, list[EvalResult]]: ...      # group on prompt_sha256
    def by_topology(self) -> dict[str, list[EvalResult]]: ...  # group on topology_signature
    def compare_origins(self, task_id: str) -> OriginComparison:
        # Pareto/quality-floor comparison per §3; refuses mismatched rubric_id
        # AND mismatched applicable_dimensions (A1); marks incomparable pairs.
```

---

## 5. EngineConfig / RunResult / capabilities() changes

`dynamic_subgraphs/engine.py`:
- `EngineConfig`: add `eval_gate: EvalGate | None = None` (default None == OFF). Add `eval_tags: EvalTags | None = None` carrying `task_id`/`origin`/`reference` for benchmark tagging; expose as `run(..., task_id=..., origin=..., reference=...)` kwargs.
- In `run()`, after `usage`/`cost`: if `eval_gate` is set, build `EvalContext` (with `planner_model_name` resolved from the planner ref so planner cost can be split — A4), call `eval_gate.evaluate(ctx)`, persist via `recorder.record_eval(run_id, result)` **only when recorder is `FileRecorder` and `Artifact.EVAL` is selected** (C2). Wrap in try/except so a gate failure never fails an otherwise-ok run (mirrors `_make_record_node`).
- `RunResult`: add `eval: EvalResult | None = None`; populate in `run()`; include `"eval": self.eval.model_dump(mode="json") if self.eval else None` in `to_dict()`.
- `__all__` / package export: add `EvalGate`, `EvalResult`, `EvalCorpus`, `DeterministicEvalGate`, `LlmEvalGate`, `EvalReference`, `EvalTags`, builders.
- `capabilities()`: add `"eval_gates": ["deterministic", "rubric-llm"]`, add `"evaluated"` to the hardcoded `recording_presets` list, `"eval.json"` flows into `artifacts` automatically.

**RESOLVED A4 — planner cost split + reported both ways.** The dynamic arm's `RunResult.cost` already includes planner tokens (the usage handler is attached to the planner ref in `_attach_callbacks`). We **split planner-model tokens out of `usage.by_model`** into `planner_cost_usd` and `worker_cost_usd` on `EvalResult`. The benchmark reports dynamic results **both with and without planner cost**, so a KILL verdict is actionable: if dynamic loses on full cost but wins on worker-only cost, a promoted/cached template library (Slice 7's endgame, which eliminates per-run planning) would recover it.

**RESOLVED Q1 — judge token accounting.** The judge fires through `build_structured_output(judge_ref, ...)` with its **own** `UsageMetadataCallbackHandler` (attached to the judge `ModelRef` via `_attach_callbacks`), so judge tokens do **not** inflate `RunResult.usage`/`cost` (the benchmark denominator). `judge_cost` is recorded separately on `EvalResult`. This keeps every value axis honest. **DECISION:** confirm separate-handler (recommended) vs counted-in-run-cost. Recommended: separate.

---

## 6. Supervisor wiring

Primary path is the engine layer (Section 5). For the **shipped CLI/HTTP corpus** (C1):
- **PR2.5** surfaces `usage` on `SupervisorResult` (additive `usage: TokenUsage | None`), populated from the handler that `app/assembly.py` already attaches. This lets the app-layer score with **real token economics** (cost stays `None` unless a price book is threaded → `value_per_ktok` is the app-layer axis).
- `build_supervisor(..., eval_gate: EvalGate | None = None)` threaded to `build_supervisor_graph`, inserting an `evaluate` pass-through node on the `record -> respond` edge, skipped when `eval_gate is None` (topology unchanged otherwise). `SupervisorState` gains `eval: EvalResult | None`; `SupervisorResult` gains `eval: EvalResult | None`.
- **Caveat (documented):** the SDK path remains the fully-featured integration (it has cost); the supervisor node is the app-layer integration (tokens, no cost by default). No fragile cost plumbing is invented into the graph.

---

## 7. The A/B benchmark + decision rule

The thesis is worth keeping only if invented graphs beat a **realistic** baseline on **quality-at-a-cost-ceiling** across a heterogeneous batch. Every task runs through identical machinery (same recorder, same usage handler, same LiteLLM cost).

### RESOLVED A3 — the baseline is a fixed-library ROUTER, not one linear graph
A single linear `plan -> web_search -> synthesize -> emit` is a guaranteed strawman on a heterogeneous batch (it wastes search on "summarize a paragraph" and underplans "research nuclear vs solar"). A KEEP verdict against it proves only "one fixed shape can't fit all tasks." Resolution — **three arms, the router is PRIMARY:**
- **Arm A (dynamic, `origin="invented"`):** current `LLMPlanner`.
- **Arm B (fixed-library router, `origin="routed"`) — the PRIMARY baseline.** A small set (3–5) of hand-authored, reviewed `GraphSpec`s + a cheap deterministic-or-tiny-LLM router that picks one per task. This is the actual production alternative DS pivots to on a KILL verdict, so the verdict is actionable either way.
- **Arm C (single fixed graph, `origin="authored"`) — sanity floor only.** The frozen linear `fixed-host-graph-v1.json`. Reported but never the headline.

The verdict is **A vs B**. A vs C is a sanity check (if A can't beat even C, the thesis is in serious trouble; if A beats C but not B, routing wins).

### Wiring gap to fix (`app/assembly.py`)
`_build_planner` hardwires `StaticPlanner(build_demo_spec())` with **mock** runners and ignores any caller spec. Add `fixed_spec: GraphSpec | None` to `RunConfig`; when set, `_build_planner` returns `StaticPlanner(config.fixed_spec)` while `_build_runners` still builds **real LLM workers** (`strict_runners=True`). For the router arm, the router selects the `fixed_spec` per task before the run. Stamp `origin` into each arm's `EvalContext`.

### Controls (each a mitigation)
- **Pin one worker model across all arms** (`run_benchmark(engine_model, ...)`) — else the result measures model strength, not topology.
- **Reviewed, frozen baselines.** The router library + `fixed-host-graph-v1.json` are checked-in validated `GraphSpec`s (plans-as-data, diffable, attackable in review). They must be genuinely capable.
- **Same gate+rubric+reference for all arms**, with `applicable_dimensions` asserted identical per task (A1) or the pair is dropped. Headline uses `DeterministicEvalGate` + mandatory references (B3); LLM-judged goal_completion is a secondary cut with reported coverage (B2).
- **Reference packs mandatory** (B3/Q4).
- **`repeats` derived, not fixed at 3** (A5). The dynamic arm re-plans each run, so topology is itself a random variable (the doc shows 23× cost swings driven by node-count choices). Run a **6-task pilot at `repeats>=5`**, compute per-task coefficient of variation of the value axis in the dynamic arm, and **derive the repeats needed to detect the target effect at the observed CV.** Report **paired confidence intervals / a sign test**, not a bare `win_rate`.
- **Heterogeneous, hard tasks** (compare/research/summarize/extract/plan/qa) with quality headroom so the comparison doesn't saturate at ~1.0.
- **Single-shot only** in the dynamic arm (no `run_iteratively`) — keep the topology test clean of the acceptance loop.

### Metrics & pre-registered decision rule (commit before running)
- **Decision variable: quality-at-a-cost-ceiling (Pareto with quality floor)** — NOT raw `value_per_usd` (A2). Report `value_per_ktok` for ranking and `value_per_usd` as diagnostic only.
- Eligibility: a run counts toward a "win" only if `quality_floor_met` (proposed floor `0.6`).
- Paired, per-task (A vs B): `pareto_win_dyn` = dynamic Pareto-dominates-or-matches the router on (quality, cost). Report **with-planner-cost and worker-only-cost** (A4).
- **KEEP topology** if, among floor-eligible tasks, dynamic Pareto-wins in ≥60% of tasks **with a paired CI excluding "no effect"**, on full cost.
- **KILL topology** (pivot to governance/routing/recursion) if the router Pareto-dominates dynamic on ≥ the same margin, even after removing planner cost (templating wouldn't save it).
- **INCONCLUSIVE** otherwise → expand the batch.
- Batch sizing: 6-task pilot (repeats≥5) to estimate CV and **derive** the main-run repeats and N; main run 24–40 heterogeneous tasks (4–6 per task type).
- **DECISION:** ratify the full pre-registered rule — `quality_floor` (0.6), the win threshold (60%), the CI/sign-test method, whether KEEP requires winning on *full* cost or *worker-only* cost, and the pilot CV→repeats derivation. Pre-registration is mandatory before any spend.

### Harness
**Module `dynamic_subgraphs/bench.py` (new, behind a `bench` extra)** — a thin loop over `BenchTask × Arm × repeat` producing `ArmRunRow`/`BenchReport`, persisting `runs/bench/<bench_id>/bench.jsonl` + `report.json`. Dataset versioned as `docs/evals/bench-tasks-v1.jsonl` (each task carries its rubric/reference/grounding-required flag inline). It **consumes** the eval layer; it never re-implements scoring. LangSmith: tag runs with `arm`/`task_id`/`origin`, register the task set via `create_dataset`, cross-check local cost against LangSmith cost (same method as `model-comparison-2026-06.md`). Results doc: `docs/evals/bench-dynamic-vs-fixed-2026-06.md`.

---

## 8. Phased implementation checklist (small PRs)

1. **PR1 — types & vocab (no behavior).** `app/eval/types.py` (`EvalContext`, `EvalResult`, `ScoreComponent`, `RunFingerprint`, `EvalReference`, `EvalTags`, `EvalGate` Protocol; `signals: dict[str, float]`). Add `Artifact.EVAL` + `Recording.evaluated()`. Pure additive; mypy-strict.
2. **PR2 — deterministic gate.** `app/eval/gates.py`: `DeterministicEvalGate` + the three deterministic scorers + reference-anchored `goal_completion` + shape-aware `topology_signature`/`instruction_sha256` + fingerprint builder + paired-mode shared-neutral grounding (A1) + Pareto/floor economics (A2/A4). Free, fully offline-testable.
3. **PR2.5 — surface usage on `SupervisorResult`** (C1) so the app-layer can score with real token economics.
4. **PR3 — recorder integration.** `ArtifactContext.eval_result`, `_EvalProducer`, `FileRecorder.record_eval`/`load_eval`, `list_runs()` eval fields; assert `record=True` includes EVAL semantics (C2).
5. **PR4 — engine wiring.** `EngineConfig.eval_gate` + `eval_tags`/`run(...)` kwargs, score-after-cost call, `RunResult.eval` + `to_dict()`, separate judge usage handler (Q1), planner-cost split (A4), `capabilities()` updates, package exports.
6. **PR5 — LLM gate.** `LlmEvalGate` + `_GoalScorePayload` (reference-anchored, length-normalized) + `build_provider_eval_gate` (reuses `judge_ref`), reference-pack loader, budget floor+fraction (B2), cross-model warning (B4), temperature-via-ModelRef (B1), judge-coverage signal.
7. **PR6 — corpus reader.** `dynamic_subgraphs/corpus.py` (`EvalCorpus`, Pareto `compare_origins` refusing mismatched `rubric_id`/`applicable_dimensions`), export.
8. **PR7 — benchmark fixed-arm seam.** `RunConfig.fixed_spec` + `_build_planner`/`_build_runners` real-worker static path; router library + `docs/evals/fixed-host-graph-v1.json`.
9. **PR8 — benchmark harness.** `dynamic_subgraphs/bench.py`, `bench-tasks-v1.jsonl` (with references + grounding-required flags), three arms (dynamic / router-primary / single-fixed-sanity), pre-registered Pareto decision rule, results doc skeleton.
10. **PR9 — optional supervisor node hook** (app-layer scoring; only if CLI/API callers need it now).

Each PR ships with tests and keeps the layer OFF by default.

---

## 9. Test plan (pytest, mirrors existing `tests/`)

- **`test_eval_types.py`** — `EvalResult` JSON round-trip; **`signals` numeric-only type-stability across dump/load** (C5); `value_per_ktok`/`value_per_usd` math, **None when cost is None / tokens always present**; `quality_floor_met` logic; `quality` bounds; `applicable_dimensions` persisted.
- **`test_topology_signature.py`** — golden signatures; **invariance** to node ids/params; **sensitivity** to edge structure and in/out-degree (B5 canary: a `branch` DAG ≠ a linear chain of the same kinds); `instruction_sha256` separates same-shape-different-instruction graphs.
- **`test_deterministic_gate.py`** — plan_validity parsimony/budget signals; **grounding paired-mode shared-neutral** (A1: inapplicable run scores the shared constant with `applicable=True`, denominator unchanged; assert two arms get identical `applicable_dimensions`); reference-anchored goal_completion; failed/paused → deterministic fail/skip, no judge; bit-stable re-run (`deterministic=True`).
- **`test_llm_gate.py`** — fake `_StructuredJudgeModel`: happy path; exception / wrong-shape / out-of-range score caught into errors-bearing result, never raises; **budget floor+fraction** (B2: cheap run still judged because of the floor; coverage signal set); cross-model warning when `judge==worker`; length-normalization applied before judging.
- **`test_recorder_eval.py`** — `_EvalProducer.applies()` gating; `eval.json` written only when selected; `record_eval`/`load_eval` round-trip; `load_eval` None for old dir; `list_runs()` eval fields; **`record=True`/`Recording.all()` includes EVAL but writes nothing without a gate** (C2); `ArtifactContext` default keeps existing call sites green; **`eval.json` never contains raw `state.values`/trace** (C3).
- **`test_sdk_eval.py`** — `EngineConfig(eval_gate=DeterministicEvalGate())` e2e (mock planner, token-free): `RunResult.eval` populated, `to_dict()` JSON-safe; **OFF by default → `RunResult.eval is None`, no `eval.json`**; gate exception does not change `RunResult.status`; **judge tokens excluded from `RunResult.usage`** (separate handler, Q1); planner cost split present (A4).
- **`test_supervisor_usage.py`** (PR2.5) — `SupervisorResult.usage` populated from the assembly handler; app-layer node hook scores with tokens, `value_per_ktok` set, cost `None`.
- **`test_corpus.py`** — `all()` skips dirs without eval.json; `by_task`/`by_topology` grouping; `compare_origins` Pareto math, refuses mismatched `rubric_id` **and** mismatched `applicable_dimensions` (A1), excludes incomparable pairs.
- **`test_bench.py`** — fixed-arm `StaticPlanner(fixed_spec)` runs **real-runner** path (not mock); router selects per task; all arms share gate/rubric/reference and identical per-task `applicable_dimensions`; **Pareto + quality-floor verdict** (A2) not raw ratio; with/without-planner-cost rows (A4); plan-failure in dynamic arm → `quality=0`, `quality_floor_met=False`; derived-repeats pilot path produces a CV and a recommended N (A5).

---

## 10. Open decisions for Ian (consolidated — blocking the tagged PRs)

- **DECISION (Q1, PR4):** judge tokens in a separate handler (recommended) vs counted in `RunResult.cost`.
- **DECISION (A1, PR2/PR8):** shared-neutral grounding constant value (proposal 0.5); grounding-required authored per-task.
- **DECISION (A2/B2, PR2/PR5/PR8):** ratify `quality_floor` (0.6), `judge_cost_floor` (0.005), `judge_budget_fraction`, the value axis for ranking (`value_per_ktok`), and λ for the penalty cut — **before running**.
- **DECISION (B3/B4, PR2/PR5):** confirm `ds.default.v1` weights and that the thesis verdict is reported per-headline (structural vs reference-goal), never on the blended `quality` alone.
- **DECISION (B5, PR2/promotion):** equivalence class for promotion (shape-only vs shape+instruction-similarity threshold).
- **DECISION (A3, PR7/PR8):** confirm the **fixed-library router is the primary baseline (Arm B)** and the single linear graph is sanity-only (Arm C).
- **DECISION (A5/§7, PR8):** ratify the full pre-registered decision rule (floor, 60% win threshold, CI/sign-test, full-cost vs worker-only-cost for KEEP, pilot CV→repeats derivation).
- **DECISION (C1, PR2.5/PR9):** score the shipped CLI/HTTP corpus in Slice 7, or defer app-layer scoring (and document `runs/` from the API as unscored until then).

### What changed from the draft (red-team's three required fixes, all applied)
1. **Verdict no longer rides on raw `value_per_usd`** (demoted to diagnostic; headline is `value_per_ktok` + Pareto-with-quality-floor), and **the baseline is a fixed-library router**, not one linear graph (A2 + A3).
2. **Rubric is paired-comparable** (per-task shared `applicable_dimensions`, grounding as a shared-neutral constant, asserted-equal-or-incomparable) and **goal_completion is reference-anchored + reported per-headline**, never a bare keyword proxy in the verdict (A1 + B3 + B4).
3. **Judge economics and determinism match the real seam** — temperature via `ModelRef`, ensemble dropped from default, budget `max(floor, fraction)` with reported coverage, planner cost split out of `by_model` and reported both ways (B1 + B2 + A4).
