# Main run v1 — runbook (dynamic vs router)

> **Status:** READY, validated offline. Only the paid live run is left, and
> it needs the owner's sign-off on §6 first.
> The decision rule is the frozen §4 of
> [`bench-dynamic-vs-fixed-2026-06.md`](./bench-dynamic-vs-fixed-2026-06.md);
> this runbook implements it and does not change it. §4 is applied by
> `dynamic_subgraphs/_bench_verdict.py`, committed before any paid main-run
> data exists, so the analysis can't be chosen after seeing results (§6).

## 1. What is pinned

| Item | Value | Where |
|---|---|---|
| Pack | 24 tasks, 4 per type × 6 types | [`bench-tasks-main-v1.jsonl`](./bench-tasks-main-v1.jsonl) |
| Worker (every arm) | `openai/gpt-5.4-nano-2026-03-17` (alias `gpt-5.4-nano`, as in the pilot) | [`bench-main-v1.config.json`](./bench-main-v1.config.json) |
| **A1** planner (`invented-a1`) | `openai/gpt-5.4-nano-2026-03-17` | config |
| **A2** planner (`invented-a2`, headline) | `openai/gpt-5.4-mini-2026-03-17` (alias `gpt-5.4-mini`) | config |
| **B** router (`routed`) | `router-library-v1/`, unchanged | config |
| **C** floor (`authored`) | `fixed-host-graph-v1.json`, unchanged | config |
| Gate | `DeterministicEvalGate(grounding_applicability="reference_only")` | `bench.py` |

The models are dated snapshots, so an alias can't move to a new model in the
middle of a multi-hour run. Each row records its planner in
`ArmRunRow.planner_model`.

**Why gpt-5.4-mini for A2.** §4 asks for a "stronger planner … from the SDK's
recommended tier", meaning the SDK's documented recipe of a capable planner
with cheap workers (README "Use a capable model for the planner";
`docs/recipes.md` §3). gpt-5.4-mini is the next tier up in the worker's own
family. It uses the same provider and key, so the provider isn't a confound,
and LiteLLM prices it, so planner cost is booked for the full-cost Pareto
check. Full `gpt-5.4` is the stronger option. It costs about 3.3× mini per
token, and the planner cost counts toward A2's full cost. To use it, change one
line in the config before launch. §4 says the planner is pinned at launch, so
this is the last moment it can change.

### Strata (task_type)

| Stratum | Tasks | Router (Arm B) sends it to | Notes |
|---|---|---|---|
| compare | 4 | `compare.json` (2-way fan-out → reduce → synthesize) | question-style, ≤2 echoed anchors |
| research | 4 | `research.json` (3 parallel web searches) | only grounded stratum (`grounding_required`) |
| summarize | 4 | `summarize.json` (1 llm_call) | inline material, 1.2k–2.4k chars |
| extract | 4 | `extract.json` (1 llm_call) | inline record; answers verbatim in material |
| plan | 4 | summarize fallback (1 llm_call) | as in the pilot: an honest stress on the router |
| **synthesize** | 4 | summarize fallback (1 llm_call) | **topology-eliciting stratum** (below) |

**Topology-eliciting stratum (`synthesize`).** These are self-contained
multi-item inputs (vendor bids, regional reports, a SKU reorder policy, a task
dependency graph). Each needs per-item work (a natural `parallel_map`
fan-out) followed by a reduce or decision stage, sometimes in two stages
(synthesize-4 re-plans after a slip). That is where an invented topology
could plausibly beat the router's single `llm_call`. The checklists hold
**computed results** that are not in the prompt (e.g. `15240`, `1735`,
`t1>t2>t4>t6>t7`), so restating the input can't score. Prompts ask for plain
digits and fixed formats so substring scoring stays fair.

### Pilot failure modes avoided (all test-pinned in `tests/test_bench_tasks.py`)

- **Checklist echo:** compare/research/plan/synthesize prompts contain at
  most 2 checklist items and at least 3 non-echoed ones. Research items don't
  collide with the router's query suffixes.
- **Tavily query breakage:** Tavily rejects queries over 400 characters. Every
  routed `web_search` query fits, and so does the authored arm's query for
  every question-style task. The authored arm still fails on the
  summarize/extract/synthesize tasks, by design: it's the sanity floor's
  honest shape failure from pilot §3, and it fails at the search node before
  any LLM spend.
- **Long-prompt planner degradation:** material prompts are capped at 2,400
  characters. The pilot's material prompts were about 2,800, and the nano
  planner hit `plan_failed` 4 times on them.
- No overlap with the pilot pack (ids, prompts, material). Grounding applies
  to research only.

### Repeats (§4: pilot-CV derivation, cap 20)

| Stratum | Repeats | Derivation (§3 table) |
|---|---|---|
| compare | 20 | pilot-compare-1 → 25, capped |
| research | 20 | max(research-1 35, research-2 7), capped |
| summarize | 3 | pilot-summarize-1 → 3 |
| extract | 6 | pilot-extract-1 → 6 |
| plan | 3 | pilot-plan-1 → 3 |
| synthesize | 20 | new type with no pilot CV, so it gets the cap. That is the most the rule allows, so a 5-repeat re-derivation pilot could only lower it. |

Each task uses the same repeat count in every arm (paired design).

## 2. Size and cost estimate

`4 tasks × 72 repeat-sum × 4 arms` = **1,152 runs**
(compare 320, research 320, synthesize 320, extract 96, summarize 48, plan 48).

| Arm | $/run (basis) | Runs | Est. |
|---|---|---|---|
| invented-a1 | 0.0028 (pilot invented mean) | 288 | $0.81 |
| invented-a2 | 0.0090 (planner share × ~3.7 mini/nano list price, 2× worker for larger plans) | 288 | $2.59 |
| routed | 0.0011 (pilot) | 288 | $0.32 |
| authored | 0.0009 (pilot; upper bound, since material tasks fail fast) | 288 | $0.26 |
| **Total** | | **1,152** | **≈ $3.97**, **≈ $6 with a 1.5× margin** |

This is inside §4's registered envelope of $4–8. Wall time is about **2.4 h**
sequential at the pilot's 7.6 s/run; A2 will likely run slower. Tavily usage
is about 288 routed research runs × 3 searches, plus up to 288 authored
searches, so roughly 1,150 search calls. Check the Tavily plan quota before
launch.

## 3. Commands

All commands run from the repo root after `uv sync --all-extras`.

```bash
# 1. Plan + estimate only (no engine, no network)
uv run python -m dynamic_subgraphs._bench_main \
  --config docs/evals/bench-main-v1.config.json --estimate

# 2. Full offline dry-run: mock planner + mock runners, the same 1,152-run
#    schedule, zero tokens (~40 s). Writes to a temp dir unless --runs-dir.
uv run python -m dynamic_subgraphs._bench_main \
  --config docs/evals/bench-main-v1.config.json --dry-run

# 3. THE LIVE RUN (paid). Needs both keys; refuses before any spend if either
#    is missing. Forces LangSmith tracing off (pilot parity). Runs under the
#    spend kill-switch (below).
export OPENAI_API_KEY=...   TAVILY_API_KEY=...
uv run python -m dynamic_subgraphs._bench_main \
  --config docs/evals/bench-main-v1.config.json --live

# 4. Apply the frozen §4 rule. Writes verdict.json + verdict.md next to the
#    rows and prints the markdown table.
uv run python -m dynamic_subgraphs._bench_main \
  --config docs/evals/bench-main-v1.config.json \
  --analyze runs/bench/main-v1/bench.jsonl
```

### Spend kill-switch

Every `--dry-run` and `--live` run is guarded by a `SpendGuard`
(`dynamic_subgraphs/bench.py`):

- **Cap.** `spend_cap_usd` in the config is **$8.00**, the top of §4's
  registered $4–8 envelope (the estimate with margin is about $6).
  `--max-spend-usd X` overrides it for one launch. Before each task run, the
  guard stops the benchmark if spend so far plus that run's estimate
  (`cost_estimate.usd_per_run`) would pass the cap. It can overshoot by at
  most one run's cost minus its estimate. A live run refuses to start with
  no cap.
- **Manual stop.** Create `runs/bench/<bench_id>/STOP` (e.g.
  `touch runs/bench/main-v1/STOP`) and the run stops before its next task
  run. The in-flight run finishes and is recorded.
- **Untracked spend.** Spend is each row's `cost_usd`. A failed run that
  used tokens but has no price is charged its arm's estimate. In a live run
  a *successful* run with tokens and no price aborts, because spend can't
  be tracked (install the `cost` extra: `uv sync --all-extras`).
- **On abort** the CLI exits with code 3. Every completed row is in
  `bench.jsonl`, `spend.json` holds the spend and the reason, and
  `report.json` is not written. Treat it like any other partial run (below).

PowerShell for step 3: `$env:OPENAI_API_KEY="..."; $env:TAVILY_API_KEY="..."`,
then the same `uv run ...` line.

Outputs go to `runs/bench/main-v1/{bench.jsonl,report.json}`, with per-run
artifacts under `runs/main-v1-*`. Rows are appended to `bench.jsonl` as each
run finishes, so a crash keeps every completed row. `report.json` is written
only at the end.

**If the live run aborts partway** (provider outage, `BenchIntegrityError`):
bench ids are single-use, so don't reuse `main-v1`. Keep the partial
`bench.jsonl` as evidence, and relaunch the whole run with
`--bench-id main-v1b` so the paired design stays intact. Record the abort in
the results doc.

`--max-repeats N` exists for smoke runs only (e.g. `--live --max-repeats 1`
is a 96-run, roughly $0.33 live smoke at the estimate rates). It is **not**
the registered sizing, so never report verdicts from a capped run.

## 4. Offline validation done

- `tests/test_bench_tasks.py` covers pack integrity: count, strata balance,
  unique ids, no pilot overlap, no checklist echo, the Tavily limit, grounding,
  material presence, and the topology-stratum shape.
- `tests/test_bench_main.py` covers the config (A1/A2 pins, repeats against
  the §3 derivation and cap), the run plan and estimate, an end-to-end CLI
  `--dry-run` (96 rows across all 4 arms with planner ids recorded), the live
  run refusing without keys before any spend, and `--estimate` spending
  nothing.
- `tests/test_bench.py` covers the harness: sub-arms, per-type repeats, rows
  surviving a mid-run crash, and refusing duplicate arm names.
- The full 1,152-run `--dry-run` was run locally: 1,152 rows, all `ok`, 288
  per arm, in about 36 s. `--analyze` on those rows runs cleanly in about
  3 s (the mock rows spend no tokens, so the verdicts it prints are
  meaningless and the report says so).
- `tests/test_bench_verdict.py` holds golden synthetic fixtures that land on
  each of the four verdicts (with ties, floor-ineligible tasks and failed
  runs), plus every number in §6. `tests/test_bench_killswitch.py` covers
  the cap, the STOP file, untracked spend and the mid-run abort.

## 5. After the run

Run step 4 (`--analyze`) on the run's `bench.jsonl`. It applies §4 per
dynamic sub-arm against `routed` and prints one verdict line per sub-arm ×
headline (structural, goal) × scope (all tasks, then each stratum). The
headline is **A2 vs B**, with A1 alongside. Copy `verdict.md` into §5 of the
results doc and commit `verdict.json` with it. If the report says
`registered_sizing: false` or lists unpaired tasks, it is not a
decision-grade run.

## 6. Reading the verdict (owner sign-off before the paid run)

### How §4 is turned into code

§4 leaves several choices open. Each is fixed in `_bench_verdict.py` now,
before any paid data, so none can be picked after seeing results.
Changing any of them after the run is a re-registration and must be written
up as one.

| Choice | Fixed as |
|---|---|
| Pairing | By `ArmRunRow.arm`, not origin (A1 and A2 are both `invented`; `corpus.compare_origins` can't tell them apart). Each sub-arm vs `routed`. |
| Per-task value | Mean over repeats of the headline score, blended quality, full cost, worker-only cost and tokens. Failed runs score 0. |
| Floor | An arm meets it on a task iff its mean blended quality ≥ 0.6. Task is floor-eligible iff at least one arm meets it. If only one arm does, that arm wins the task. If neither does, the task is dropped. |
| Pareto | Win = ≥ on headline score and ≤ on cost, strictly better on one. Anything else is a tie, including better quality at higher cost. |
| Cost | Full = `cost_usd` (planner included). Worker-only = `worker_cost_usd`, or `cost_usd` when there's no planner split (the router has no planner). If any row on a task is unpriced, that task uses tokens and both bases are equal (flagged). |
| Win share | Wins ÷ floor-eligible tasks. Ties stay in the denominator ("≥60% of floor-eligible tasks"). |
| Sign test | Exact one-sided binomial (`math.comb`) over decisive tasks, ties excluded. |
| Bar | Share ≥ 60% **and** p < 0.05. |
| Bootstrap | 95% percentile CI of the win share; resample eligible tasks; 10,000 resamples; `random.Random(20260611)`. Reported, not part of the bar. |
| Verdict | KEEP: dynamic clears the bar on full cost. KILL: router clears it on full **and** worker-only cost. INCONCLUSIVE-leaning-template-cache: dynamic clears it on worker-only cost only. Otherwise INCONCLUSIVE. |
| λ cut | Per run `headline − 0.05 × tokens / 10,000`; per task the higher mean wins. Reported, never decides. |
| Strata | Same computation, marked descriptive. |

### The arithmetic

These numbers come from the code and are pinned in
`tests/test_bench_verdict.py`.

- **No stratum can reach a verdict.** A stratum has 4 tasks, so the best
  one-sided p is 4/4 wins = 1/16 = **0.0625**. Per-stratum lines are
  descriptive only. They can show "complex tasks favor planning" as a
  pattern, but they can't KEEP or KILL anything.
- **The headline needs ≥17 of 24 wins (71%), not 60%.** With no ties,
  16/24 gives p = 0.076 and 17/24 gives p = 0.032. The sign test binds
  harder than the 60% share.
- **Ties move the bar.** Ties leave the sign test's n but stay in the
  share's denominator, so both bars move:

  | Ties (of 24 eligible) | Decisive n | Wins for p<0.05 | Wins for ≥60% share | Wins needed |
  |---|---|---|---|---|
  | 0 | 24 | 17 | 15 | **17** |
  | 2 | 22 | 16 | 15 | **16** |
  | 4 | 20 | 15 | 15 | **15** |
  | 6 | 18 | 13 | 15 | **15** |
  | 8 | 16 | 12 | 15 | **15** |
  | ≥10 | ≤14 | – | 15 | **impossible** |

  With 8 ties, 12/16 decisive passes the sign test (p = 0.038), but 12/24 is
  only a 50% share, so it is INCONCLUSIVE. With 10 or more ties no KEEP or
  KILL is possible. Floor-ineligible tasks shrink the 24 the same way.
- **Repeats don't add decision power.** Repeats are averaged into one value
  per task before the Pareto check, so they reduce noise within a task.
  The sign test's n is the number of tasks (at most 24), whatever the repeat
  count. A larger batch of tasks is the only way to more power, which is
  §4's INCONCLUSIVE path ("expand the batch").

Sign-off: the owner has read this section and accepts these numbers before
launching `--live`.
