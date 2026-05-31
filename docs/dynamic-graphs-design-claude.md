# Dynamic Subgraphs — Design Notes (Claude)

## What we're trying to build

A LangGraph-based runtime where an LLM, given a prompt, can **synthesize a graph
(N nodes, M edges) at runtime**, execute it, throw it away, and keep a record
of what it built and what it produced. The motivating use cases the operator
cited:

- Spin up a sub-agent on demand.
- Kick off an async background process and come back for the answer later
  (wait for a webhook, poll a job, etc.).
- Do a one-off computation, gather data from N sources in parallel, summarize.
- Generally: let the agent *decide its own topology* per problem, instead of
  routing every problem through one frozen graph.

The key constraint is that today, `graph.compile()` is treated like a build
step — it happens once before the program starts. We want it to be something
the agent itself does mid-conversation, as often as it likes, with parameters
the LLM filled in.

## How this differs from what questforge already does

QuestForge uses `deepagents.create_deep_agent(...)`, which under the hood is a
**fixed StateGraph** (an agent loop) plus a registry of named subagents the
director picks among via a `task` tool. The director isn't building a graph —
it's calling pre-defined specialists by name. The topology (director ↔ tools ↔
named subagents) is hardcoded at module-import time.

Dynamic subgraphs are a different animal: the *shape* of the computation is an
**output of the LLM**, not a constant of the program. The closest cousin in
the questforge world would be: imagine the director, instead of calling
`task("lore-keeper", ...)`, instead emitting `{nodes: [...], edges: [...]}`,
having that compiled into a transient StateGraph, running it, and folding the
result back into its own state.

Both styles are legitimate. Deepagents trades flexibility for predictability;
dynamic graphs trade predictability for the ability to fit weird-shaped
problems without pre-enumerating them.

## What LangGraph actually supports here

LangGraph is friendlier to runtime construction than the "compile is a build
step" framing suggests. The relevant primitives:

1. **`StateGraph(...).compile()` is just a function call.** Nothing stops you
   from calling it inside a node. You can build, compile, and `.invoke()` a
   fresh graph mid-run. The compiled artifact is a plain Python object you
   can drop on the floor when done.

2. **Subgraphs as nodes.** A compiled graph can be added as a node in another
   graph. So an outer "host" graph can contain a "run dynamic subgraph" node
   that compiles + invokes whatever the planner emitted, then returns its
   final state as the node's output.

3. **`Send` API.** Inside an edge function you can return `[Send("node", payload), ...]`
   to fan out N parallel invocations of `"node"` with N different payloads,
   where N is decided at runtime. This is the "map" half of map-reduce and is
   the lightest-weight form of "dynamic topology" — you don't have to build
   a new graph, you just decide how many copies of an existing node to launch.

4. **`Command(goto=..., update=...)`.** A node can return a `Command` that
   directs the runtime to jump to a named node and update state in one shot.
   Combined with conditional edges, this lets a single fixed graph behave
   like a much larger one — the LLM's output controls routing instead of
   topology.

5. **`interrupt()` + `Checkpointer`.** Native pause/resume. A node can
   `interrupt()` to wait for a human (or for a background job to finish);
   the checkpointer persists state; a later `.invoke(Command(resume=...))`
   continues. This is how you wait for an async process without holding a
   coroutine open.

6. **Functional API (`@entrypoint`, `@task`).** A more flexible alternative
   to `StateGraph` where you write what looks like normal Python and the
   framework records the dependency graph implicitly. Useful when the graph
   you'd build is mostly "do A, then B, then C in parallel" and the
   StateGraph ceremony adds nothing.

The implication: there are at least **three legitimate ways** to deliver
"dynamic graphs," and they sit on a spectrum from "the LLM emits a literal
graph spec we compile" to "the LLM just makes routing decisions inside a
fixed graph that's permissive enough to feel dynamic."

## Three architectural approaches, in increasing rigidity

### A. **Literal dynamic graph (LLM emits topology)**

The agent outputs a JSON spec like:

```json
{
  "nodes": [
    {"id": "fetch_weather", "kind": "http_get", "params": {"url": "..."}},
    {"id": "fetch_news",    "kind": "http_get", "params": {"url": "..."}},
    {"id": "summarize",     "kind": "llm_call", "params": {"prompt": "..."}}
  ],
  "edges": [
    {"from": "START", "to": "fetch_weather"},
    {"from": "START", "to": "fetch_news"},
    {"from": "fetch_weather", "to": "summarize"},
    {"from": "fetch_news",    "to": "summarize"},
    {"from": "summarize", "to": "END"}
  ]
}
```

A compiler function walks the spec, instantiates each node from a **registry
of node kinds** (`{"http_get": http_get_node, "llm_call": llm_call_node, ...}`),
wires the edges, calls `.compile()`, and `.invoke()`s the result. The output
plus the spec get persisted, then the graph is discarded.

**Strengths:**
- Truly arbitrary topology; the LLM has the most leverage.
- The spec is *inspectable and replayable* — you can re-run the same graph
  later for debugging or eval.
- Easy to visualize: dump the spec to Graphviz/Mermaid.

**Weaknesses:**
- You have to design the node-kind registry carefully. Every "kind" is
  effectively a sandboxed capability with a typed parameter schema. Too few
  kinds and the LLM is stuck; too many (or one that takes arbitrary Python)
  and you've reinvented an unsandboxed eval.
- The LLM has to produce a valid DAG (or known cycle pattern). Schema-
  validate hard, then *also* check for unreachable nodes, dangling edges,
  type mismatches between producer and consumer.
- Debugging a graph the LLM made is harder than debugging one you wrote —
  stack traces happen inside synthesized nodes whose existence you didn't
  plan for. Logging and a good "graph spec → mermaid" renderer pay for
  themselves immediately.

### B. **Dynamic routing inside a permissive host graph (`Send` + `Command`)**

Don't synthesize topology. Build one host graph with a small number of
generic nodes: `planner`, `worker`, `reducer`, `wait_for_event`, `done`.
The planner's output is a list of `Send("worker", payload)` calls. The
worker handles a wide range of payloads (it dispatches internally based
on `payload.kind`). The reducer folds results.

This is what most production "agentic" systems actually look like under
the hood. It's "dynamic" in behavior but "static" in topology.

**Strengths:**
- No graph-construction code at runtime. Cheap, fast, easy to reason about.
- Existing LangGraph tracing/checkpointing just works.
- The worker registry doubles as your node-kind registry from approach A,
  but you never have to compile anything new.

**Weaknesses:**
- "What did the agent decide to do" is encoded in messages and routing
  decisions, not in a graph object. Visualization is post-hoc reconstruction.
- The topology is implicitly capped by the host graph's edges. If the LLM
  wants a 3-stage pipeline with a feedback loop between stages 2 and 3, you
  have to have anticipated that shape.

### C. **Hybrid: dynamic subgraph compiled inside a static outer**

This is the approach I'd actually start with.

```
                            ┌────────────────────┐
   user prompt  ──►  planner ──► spec ──►  compile_and_run  ──►  recorder ──► END
                            └────────────────────┘
                                      │
                          ┌───────────┴────────────┐
                          │  transient StateGraph  │
                          │  built from spec,      │
                          │  compiled, invoked,    │
                          │  discarded             │
                          └────────────────────────┘
```

The **outer graph is static and small**: `planner` → `compile_and_run` →
`recorder` → `END`. The planner is an LLM call that emits a graph spec
constrained by the node-kind registry. `compile_and_run` is a normal Python
node that takes the spec, builds a `StateGraph`, compiles it, invokes it
with the planner's seed state, and returns the final state. `recorder`
persists `{prompt, spec, mermaid, final_state, telemetry}` to disk (or a
db) and returns a summary to the user.

The transient subgraph is real LangGraph — it gets checkpointing, tracing,
streaming, interrupts, the works. The outer graph stays trivially debuggable.

**Why this is the right starting point:**

- You get the inspectability of approach A (the spec is a real artifact)
  *and* the operational simplicity of approach B (the outer graph is fixed
  and tooled).
- The planner can be swapped out independently of the compiler. Start with
  a "planner" that's just a function returning a hardcoded spec for one
  use case; once the compile/run/record path is solid, replace it with an
  LLM.
- If you later decide some shapes are common enough to bake into the host
  graph (approach B), you can — it's an optimization, not a rewrite.

## The node-kind registry is the actual hard part

Whichever approach you pick, the LLM is choosing from a vocabulary of node
kinds. That vocabulary is the contract between "what the model can imagine"
and "what your runtime can actually do." Some initial candidates worth
thinking about:

| Kind                  | Params                         | Notes |
|-----------------------|--------------------------------|-------|
| `llm_call`            | `prompt`, `model?`, `tools?`   | The escape hatch — punts a sub-problem back to an LLM. |
| `http_get` / `http_post` | `url`, `headers?`, `body?`  | Data gathering. Needs an allowlist or a sandbox. |
| `tool_call`           | `tool_name`, `args`            | Calls a registered Python tool. Cheaper than wrapping every tool as its own kind. |
| `python_eval`         | `code`, `inputs`               | **Don't ship this until sandboxed** (e.g., subprocess + resource limits, or a wasm runtime). |
| `branch`              | `condition`, `if_true`, `if_false` | Lets the LLM express conditional edges as nodes. |
| `wait_for_event`      | `event_id`, `timeout`          | Maps to `interrupt()` under the hood — this is your "async background process" primitive. |
| `parallel_map`        | `over`, `node`                 | Maps to `Send`. |
| `summarize` / `reduce`| `inputs`, `instruction`        | LLM call specialized for fold steps. |

**Design heuristic:** start with the smallest registry that lets you build
the demo use case end-to-end, and grow it only when you hit a real wall. The
temptation to ship a `python_eval` kind on day one is strong and should be
resisted — once it's there the LLM will reach for it and you'll be debugging
synthesized Python forever.

## Recording, replay, and eval

The "throw the graph away, keep the record" requirement is what makes this
project actually tractable. The record should at minimum include:

- The triggering prompt and any seeded context.
- The full graph spec (JSON) and a rendered Mermaid diagram.
- Per-node IO: inputs, outputs, errors, wall time, token spend.
- The final assembled output.
- A `replay_id` you can pass back to re-execute the same spec deterministically
  (modulo LLM non-determinism — pin temperature/seed where possible).

The minimum-viable persistence layer is a directory of JSON files
(`runs/<uuid>/spec.json`, `runs/<uuid>/trace.jsonl`, `runs/<uuid>/output.md`).
The maximum-viable version is a small SQLite db with `runs`, `nodes`, `edges`,
`messages` tables. Either works; pick the one that doesn't slow you down.

Once you have records, you can build:
- A *graph gallery* — "show me every graph the agent has ever built; cluster
  by shape." Likely tells you which shapes recur and could be promoted to
  named templates.
- A *replay UI* — re-run a recorded spec with a different model or with one
  node mocked, to debug regressions.
- An *eval set* — `(prompt, expected output)` pairs you can run nightly to
  catch planner drift when you swap models or change prompts.

## Risks I'd flag up front

- **Cost explosion.** A planner that emits 50-node graphs three layers deep
  will spend serious money on tokens. Put a hard cap on `len(nodes)` in the
  spec validator and a global per-run wallclock + token budget. The graph
  runner should `raise` and the recorder should still get to log a failed run.
- **Infinite loops / runaway recursion.** If a node-kind is allowed to be
  "spawn another dynamic subgraph," you need a depth counter passed through
  state. Default cap to ~3. LangGraph's `recursion_limit` helps on a per-graph
  basis but doesn't catch nested-graph stack depth.
- **State schema fragility.** Each transient subgraph needs a `State` typeddict.
  Either generate it from the spec (one key per node id) or use a generic
  `dict`-typed channel with a permissive reducer. The first is safer at the
  cost of complexity; the second is fine for v1.
- **Debuggability.** When something goes wrong inside a synthesized subgraph,
  the stack trace points at your `compile_and_run` node, not at the conceptual
  node the LLM intended. Mitigate by (a) per-node logging with the LLM's node
  id, (b) catching exceptions in each node wrapper and attaching them to the
  recorded trace rather than letting them bubble silently.
- **Determinism for replay.** LLM nodes are non-deterministic. Decide early
  whether replay means "same spec, same inputs, possibly different LLM output"
  (likely fine) or "byte-identical re-execution" (much harder; needs cached
  LLM responses).

## A 1-week MVP

1. **Day 1.** Hardcoded spec → compile → invoke → print result. No LLM
   planner yet. Registry: `{"llm_call", "http_get", "summarize"}`. Outer
   graph: `runner` → `END`.
2. **Day 2.** Add the recorder. Persist spec + per-node IO to
   `runs/<uuid>/`. Render a Mermaid diagram of the spec.
3. **Day 3.** Add the LLM planner node. Constrain output via JSON schema
   (use `with_structured_output` on the model). Validate the spec is a DAG,
   nodes reference real kinds, edges reference real nodes.
4. **Day 4.** Add `parallel_map` (built on `Send`) and `wait_for_event`
   (built on `interrupt`). The async-background-process use case becomes
   real here.
5. **Day 5.** Eval harness: 5-10 hand-written prompts with rough expected
   shapes. Run nightly. Track planner drift across model versions.
6. **Day 6-7.** Build whichever piece broke first. Almost certainly the
   planner's spec quality or the registry surface area.

## Where this could go

If the core loop works, the natural extensions are:

- **Graph templates / macros.** Recurring shapes the planner emits get
  promoted to named templates the LLM can reference instead of re-deriving.
  This is how you keep token spend down as the system matures.
- **Composable subgraphs.** A node kind that *is* a recorded prior graph,
  invoked by id. The agent starts building libraries of its own solutions.
- **Cross-run memory.** The recorder feeds a retrieval index ("have I solved
  something like this before?") that the planner consults before emitting a
  new spec.
- **Human-in-the-loop checkpoints.** A node kind `await_human(question)` that
  interrupts and waits for operator approval before continuing. Trivially
  built on `interrupt()`; very valuable for the "I'm about to spend money or
  send an email" cases.

## Open questions worth answering before writing more code

1. Is the goal *one* synthesized graph per user prompt, or can the agent
   build multiple in sequence within one conversation? (Affects how
   conversational state relates to graph state.)
2. Are dynamic subgraphs allowed to spawn dynamic subgraphs? If yes, what's
   the depth cap and how does cost accounting roll up?
3. Where does the planner live? A standalone "planner" agent that only emits
   specs, or is it the user-facing agent itself emitting a spec as one of
   several tool-shaped outputs?
4. Replay semantics — "re-run the spec" or "re-run with cached LLM responses"?
5. Persistence — flat files, SQLite, or piggyback on LangGraph's checkpointer
   tables? The first is fastest to ship; the third is most consistent with
   the rest of the framework.

None of these block starting; all of them want answers before the system
calcifies.

---

## Appendix — Comparison against the GPT-5.5 / Codex draft

After this doc was written, a parallel version of the design (`dynamic-graphs-design.md`)
was produced by a different model (GPT-5.5 via Codex). Both docs land in
roughly the same place architecturally, but they emphasize different things
and each catches some details the other missed. This appendix is a candid
read of where they agree, where they diverge, and where each is stronger.

### Where we agree (good signal — these are the load-bearing claims)

- **Hybrid architecture is the right starting point.** Both docs converge on
  the same shape: a small static outer/supervisor graph (`plan → compile → run
  → record → respond`) wrapping a transient inner graph the LLM synthesizes
  per problem. Neither doc recommends starting with full dynamic-everywhere.
- **The node-kind registry is the load-bearing decision.** Both docs call
  this out as more important than the graph machinery itself. Same logic:
  the registry is the *language* the LLM composes over, and its surface
  area determines both expressiveness and safety.
- **`python_eval` and arbitrary-shell node kinds are v2+, not v1.** Both
  warn explicitly against this and for the same reason: once it exists, the
  LLM will route around your typed operators and you've quietly become a
  code-gen runtime with no sandbox story.
- **Recording is non-optional, and replay is the research instrument.**
  Both docs treat the persisted spec+trace as a first-class artifact (not
  a debugging convenience), and both anticipate that recorded graphs
  eventually become a retrieval substrate for future plans.
- **Two failure-mode clusters dominate:** cost/recursion blowup, and
  debuggability of synthesized topology. Both docs spend real space on each.
- **The long arc is template emergence.** Both predict that recurring graph
  shapes will surface in the recorder, get promoted to named templates the
  planner can reference by id, and gradually shift the system from
  "improvising from scratch each time" to "composing from a learned library."

When two independently written designs agree this completely on the core
architectural claims, that's worth taking seriously. The disagreements
below are about emphasis and framing, not direction.

### Where the Codex draft is meaningfully stronger

1. **The "four kinds of dynamic" distinction.** Codex separates *dynamic
   parameters*, *dynamic fan-out*, *dynamic routing*, and *dynamic topology*
   as four orthogonal axes. This doc collapses them into "approach A/B/C"
   which mixes the axes together. The Codex framing is sharper for thinking
   about which axes you actually need — most production systems live happily
   on the first three and never need genuine topology synthesis. I should
   have separated these.

2. **`GraphSpec` schema versioning + an inline `budget` envelope.** Codex's
   `GraphSpec` example includes `schema_version: 1` and a `budget` block
   (`max_nodes`, `max_depth`, `max_wall_seconds`, `max_llm_calls`) baked
   into the spec itself. This doc treats budgets as a runtime concern rather
   than a spec-level concern. Codex's framing is better: the budget travels
   with the spec, which means it survives serialization, gets recorded with
   the run, and the planner can be made aware of it in-context.

3. **Three-tier classification of "async."** Codex separates (a) short
   concurrent work (use `Send`), (b) durable waiting (use `interrupt()` +
   checkpointer), and (c) true detached background work (needs a separate
   job manager — control plane vs. execution plane). This doc lumped these
   together under `wait_for_event`. Codex is right that case (c) is not a
   graph problem at all; trying to make LangGraph the job manager is a
   common failure mode.

4. **Explicit maturity curve (phases 1-4).** Codex sketches a four-phase
   arc (transient experiments → motif emergence → reusable library →
   adaptive orchestration that picks among fixed/template/novel per
   problem). This is more useful than my "where this could go" bullet list
   because it implies *ordering* and *what success looks like at each
   phase*. Phase 4 in particular is a genuinely interesting target — the
   system choosing when to be novel vs. when to reuse.

5. **Exhaustive pre/post validator checklist.** Codex enumerates roughly
   a dozen checks the spec validator should run (unique ids, registry
   membership, reachability, declared cycles only, budgets, input
   provenance, etc.). This doc handwaves "schema-validate hard." Codex's
   list is the kind of thing you can hand to whoever builds the validator
   and have them ship it.

6. **The generic `DynamicRunState` envelope recommendation.** Codex
   recommends a single `TypedDict` with `inputs`/`values`/`artifacts`/
   `errors`/`events`/`metadata` keys, and graduating to per-graph typed
   schemas only after specific graph families stabilize. This doc offered
   the choice ("generate per-spec or use generic dict") without a
   recommendation. Codex's is cleaner and lets you ship v1 without a
   spec-to-TypedDict codegen step.

7. **Final-take framing.** "A system that learns how to assemble temporary
   organizations of cognition" is a better articulation of the long-term
   thesis than anything in this doc. It also reframes the project from
   "dynamic graphs" (a mechanism) to its actual purpose (organizational
   learning over time).

### Where this doc is meaningfully stronger

1. **More concrete enumeration of LangGraph primitives.** This doc names
   `Send`, `Command(goto=...)`, `interrupt()` + `Checkpointer`, subgraphs-
   as-nodes, and the Functional API individually, with what each is good
   for. Codex mentions these but more abstractly. If someone is going to
   pick up this doc and start writing code, they need to know which API
   surface corresponds to which capability.

2. **Day-by-day MVP plan.** A 1-week schedule with concrete daily deliverables
   is more actionable than a milestone list. Both have their place, but
   for a side-project that needs momentum, "what do I build on day 1" is
   the more useful framing.

3. **The stack-trace debuggability failure mode is named explicitly.** This
   doc calls out that errors will appear to originate from `compile_and_run`
   rather than from the conceptual node the LLM intended, and proposes the
   specific mitigation (per-node wrapper that catches and attaches the
   LLM's node id to the trace). Codex's debuggability section is generic.

4. **The hybrid section's promotion-path claim.** This doc explicitly notes
   that approach C is not a commitment — if some shapes recur enough you
   can later bake them into a permissive host graph (approach B), and it's
   an optimization, not a rewrite. That matters when deciding to start
   with C: you want to know the exit ramps.

5. **Comparison to QuestForge as the lead framing.** This doc grounds the
   "what's actually new here" claim against a concrete prior system
   (deepagents = fixed loop + named subagents). Codex mentions QuestForge
   but doesn't use it as a foil to clarify what "dynamic topology" is
   adding beyond what the operator already has.

### Where we disagree (small, but real)

- **State schema for v1.** Codex recommends the generic `DynamicRunState`
  envelope and graduating later; this doc offered both options without
  picking. I now think Codex is right — ship the envelope, defer codegen.
- **Whether the planner should emit rationale alongside topology.** Codex
  raises this as an open question; this doc doesn't. Probably worth doing
  for evals (you want to be able to ask "why this shape?") even if the
  rationale field is never executed.
- **References.** Codex includes a References block with real LangChain
  doc URLs. This doc has none. The Codex one is more useful for someone
  who needs to verify a claim or learn a primitive.

### Net recommendation

Read both docs together; they cover different surface area. If forced to
pick one as the source of truth for the implementation, I'd take the
Codex doc's **structure** (the four-kinds framing, the versioned GraphSpec
with budget, the three-tier async distinction, the maturity curve) and
splice in this doc's **concretes** (the named LangGraph primitives, the
day-by-day MVP plan, the explicit stack-trace mitigation, the QuestForge
contrast). That composite would be the strongest starting brief.

The single biggest thing both docs agree on, and that I'd put at the top
of any final version: **the registry is the language; the graph is just
its temporary executable form**. Get the registry right and most other
choices are recoverable. Get the registry wrong and no amount of LangGraph
cleverness will save you.
