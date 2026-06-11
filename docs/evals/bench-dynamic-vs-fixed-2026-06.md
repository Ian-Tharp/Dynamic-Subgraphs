# Benchmark: Dynamic (LLM-invented) vs Hand-Authored Topologies — 2026-06

> **Status:** PILOT — results pending; this document carries NO verdict.
> The KEEP/KILL decision rule is **deliberately not yet registered**: per the
> design spec (§0/§7, `docs/specs/2026-06-03-eval-value-layer-design.md`), the
> full rule (win threshold, CI method, full-vs-worker-cost basis, penalty λ)
> is pre-registered in this document — with the pilot CV table in hand but
> **before** any main-run spend — and then never edited. The benchmark harness
> (`dynamic_subgraphs/bench.py`) intentionally contains no decision logic.

## 1. Question

Does an **LLM-invented topology** (the live planner) beat a **hand-authored
baseline** on cost-adjusted quality across a heterogeneous task batch?
This is the experiment Ian's adversarial panel (2026-06-03) named as the one
move that decides the dynamic-topology thesis.

## 2. Method

### Arms
| Arm | origin | What runs |
|---|---|---|
| A — dynamic | `invented` | live `LLMPlanner` plans per task, per repeat (topology is a random variable by design) |
| B — router (**PRIMARY baseline**) | `routed` | a deterministic router picks one of 4 reviewed, frozen `GraphSpec`s by task_type (`docs/evals/router-library-v1/`); `{prompt}` substituted into instructions + tool args (`fill_spec`); real LLM workers |
| C — single fixed (sanity floor) | `authored` | one frozen linear web_search→synthesize graph for every task (`docs/evals/fixed-host-graph-v1.json`) |

The verdict question is **A vs B**. A vs C is a sanity check only.

### Controls (all enforced in code, most test-pinned)
- **One pinned worker model across all arms:** `gpt-5.4-nano` (planner runs on the same model in the dynamic arm; planner cost is split out per-row so results are reported with and without it).
- **Same gate + rubric + references for every arm:** `DeterministicEvalGate(grounding_applicability="reference_only")` — grounding applicability is a task property, so `applicable_dimensions` are identical across arms by construction (test-pinned).
- **Locked scoring decisions (2026-06-09):** required-dimension zero for ungrounded runs on grounding-required tasks (deliberately harsher on the dynamic arm than the neutral-0.5 alternative); verdicts reported per-headline (structural vs reference-anchored goal), never on blended quality alone; `value_per_ktok` is the ranking axis, `value_per_usd` diagnostic only; quality floor 0.6.
- **Reference-anchored goal scoring:** lowercase-substring checklist coverage minus must_not_include penalties; checklists de-echoed (≤2 deliberate subject anchors for question-tasks; zero router-suffix collisions — test-pinned) so prompt-parroting can't score.
- **Integrity aborts:** a successful run that produces no EvalResult raises `BenchIntegrityError` (an infrastructure failure must never be booked as quality-0 evidence); bench_ids are single-use.
- **Single-shot runs** (no iterative meta-loop); sequential execution; LangSmith tracing disabled for the run (quota exhausted — would add retry latency for dropped data).

### Task pack
`docs/evals/bench-tasks-pilot-v1.jsonl` — 6 tasks, 5 types (compare, research×2, summarize, extract, plan). Research tasks (and only research tasks) carry `grounding_required: true`. Summarize/extract material is inline (self-contained). The plan-type task deliberately exercises the router's fallback path (routes to the summarize spec) — an honest stress on the router arm.

### Pilot purpose (NOT a verdict)
6 tasks × 3 arms × 5 repeats = 90 runs. The pilot exists to measure the
**per-task coefficient of variation** of `value_per_ktok` in the dynamic arm
(its topology re-rolls every repeat → it is the variance-dominant arm), from
which the main run's required repeats are derived
(`max(3, ceil((1.96·CV/0.20)²)+1)` — normal approx., 95%, pre-registered 20%
relative effect), plus failure/cost telemetry.

## 3. Pilot results (2026-06-11)

`bench_id=pilot-2026-06` — 90 rows (6×3×5), wall 681s, **total cost $0.1420**,
309,659 tokens. Raw data: `runs/bench/pilot-2026-06/{bench.jsonl,report.json}`;
per-run artifacts under `runs/pilot-2026-06-*/`. Worker pinned: gpt-5.4-nano.

### Run health per arm

| arm | n | ok | failures |
|---|---|---|---|
| invented | 30 | 25 | 4× `plan_failed` (2 summarize, 2 extract — the long-inline-material tasks), 1× `execution_failed` (research-1 r0) |
| routed | 30 | **30** | none |
| authored | 30 | 20 | **10× `execution_failed` — ALL 5 repeats of BOTH material tasks** |

**Authored-arm failure cause (verified from output.json):** the single fixed
graph substitutes the entire ~2,800-char prompt into its Tavily `web_search`
query → HTTP 400. This is an *honest shape failure*, not an infrastructure
artifact: a one-size-fits-all search-then-synthesize graph genuinely cannot
serve "summarize/extract this inline material" — which is precisely what a
sanity floor exists to expose. (Contrast: the router sent those tasks to its
single-`llm_call` specs and scored 0.964 / 0.800.)

**Invented-arm `plan_failed` pattern:** 4 of 5 plan failures occur on the two
tasks whose prompts embed large inline material — the nano planner's
structured-output emission degrades with very long prompts. Per the locked
rules these score quality 0 (robustness is part of the thesis).

### Per-arm aggregates (ok rows for means; quality counts failures as 0)

| arm | mean quality (all 30) | structural | goal | value_per_ktok | mean cost/run | mean nodes |
|---|---|---|---|---|---|---|
| invented | 0.643 | 0.897 | 0.644 | 0.168 | $0.0028 | **1.6** |
| routed | **0.870** | 0.987 | **0.741** | **0.779** | $0.0011 | 2.8 |
| authored | 0.552 | 0.998 | 0.621 | 0.456 | $0.0009 | 2.0 |

### Per-task mean quality (failures as 0)

| task | invented | routed | authored |
|---|---|---|---|
| compare-1 | 0.909 | 0.935 | 0.756 |
| research-1 | 0.459 | 0.895 | 0.828 |
| research-2 | 0.802 | 0.777 | 0.794 |
| summarize-1 | 0.486 | 0.964 | 0.000 |
| extract-1 | 0.408 | 0.800 | 0.000 |
| plan-1 | 0.791 | 0.848 | 0.936 |

### Pilot CV table (invented arm, value_per_ktok) → derived main-run repeats

| task | CV | samples | dropped_none | derived repeats |
|---|---|---|---|---|
| compare-1 | 0.498 | 5 | 0 | 25 |
| research-1 | 0.594 | 5 | 0 | 35 |
| research-2 | 0.238 | 5 | 0 | 7 |
| summarize-1 | 0.123 | 3 | 2 | 3 |
| extract-1 | 0.221 | 3 | 2 | 6 |
| plan-1 | 0.070 | 5 | 0 | 3 |

### Observations (directional only — n is small and the rule is not yet registered)

1. **The router arm dominated this pilot:** 30/30 clean, highest mean quality,
   highest goal headline, ~4.6× the dynamic arm's value_per_ktok, at ~40% of
   its per-run cost. The dynamic arm won no task decisively (research-2's
   0.802-vs-0.777 is within repeat noise).
2. **The dynamic arm's mean plan size was 1.6 nodes** — consistent with the
   2026-06-03 panel's finding that these task shapes don't elicit topology
   invention; the planner pays its planning round-trip mostly to emit a single
   llm_call.
3. **Planner-cost asymmetry is material at this scale:** the dynamic arm's
   per-run cost (~$0.0028) is ~2.5× the router's (~$0.0011); worker-only cost
   comparisons (per the locked A4 split) will matter for the main run's
   KEEP-on-which-cost-basis decision.
4. **Robustness differences are real signal:** 5 invented failures vs 0 routed.
   The authored arm's 10 failures are the sanity floor working as designed.
5. **Variance is task-type dependent:** derived repeats range 3 (plan,
   summarize) to 35 (research-1) — the main run should size repeats per task
   type, not uniformly.

## 4. Pre-registered decision rule — REGISTERED 2026-06-11 (Ian + Claude, with §3 pilot data in hand, before any main-run spend; this section is now frozen)

**Arms (main run).** Two dynamic sub-arms, both judged against the router:
- **A1** = nano planner + nano workers ("as deployed cheaply" — pilot continuity)
- **A2** = stronger planner (model pinned at main-run launch from the SDK's
  recommended tier) + nano workers (the SDK's own documented recipe — the
  steelman). Each sub-arm gets its own §5 verdict line; the headline thesis
  verdict is **A2 vs B** (the thesis's best case), with A1 reported alongside
  (the deployment-reality case).

**Rule (per dynamic sub-arm vs Arm B):**
- Eligibility: `quality_floor_met` (floor **0.6**, locked 2026-06-03).
- Decision variable: per-task paired Pareto on (quality, **full cost** —
  planner included). Worker-only cost is the registered sensitivity cut: a
  dynamic arm that loses on full cost but wins worker-only is recorded as
  **INCONCLUSIVE-leaning-template-cache**, never KEEP.
- **KEEP** iff the dynamic arm Pareto-wins **≥60%** of floor-eligible tasks
  AND a paired sign test rejects no-effect at **p<0.05** (bootstrap CI on the
  per-task win indicator reported, 10,000 resamples).
- **KILL** iff the router Pareto-dominates the dynamic arm at the same bar
  (≥60%, p<0.05) **even on worker-only cost** (templating wouldn't save it).
- **INCONCLUSIVE** otherwise → expand the batch (same rule, no re-registration).
- Penalty-form sensitivity cut (secondary, never decisive):
  `adjusted = quality − 0.05·(total_tokens/10,000)` (λ = 0.05 per 10k tokens).
- Verdicts reported **per headline** (structural; reference-anchored goal) and
  per stratum (see pack) — never on blended quality alone (locked 2026-06-09).

**Main-run pack & sizing:** 24 tasks, stratified — 4 per type across 6 types,
including a **topology-eliciting stratum** (multi-source synthesis /
data-dependent fan-out tasks the pilot pack lacked), authored under the same
de-echo + reference rules as the pilot pack (echo-guard test extended to the
new pack). Per-task repeats from pilot-CV derivation (§3 table; re-derived per
new task type at pilot rates), **capped at 20**. Per-stratum verdict lines so
"simple tasks favor routers, complex tasks favor planning" is visible if true.
Estimated ~1,800–2,600 runs ≈ $4–8 at pilot rates (4 arms incl. A2).

## 5. Main run

Not started. Pack authoring + A2 planner pinning are the remaining
prerequisites; the rule above is frozen.
