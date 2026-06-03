# SDK Recipes

Copy-pasteable patterns for the `dynamic_subgraphs` SDK. Each recipe says
*when to use it* and shows a runnable snippet. For the full option set an
agent can call `DynamicSubgraphs.capabilities()`.

```python
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model, Recording, Artifact
```

All engine configuration is built as an `EngineConfig` and passed to the
engine: `DynamicSubgraphs(EngineConfig(...))`. Per-run overrides go on `run()`.

> ⚠️ **Planner model capability.** The planner must emit a valid `GraphSpec`.
> Reliability depends on **both the model and the graph's complexity** — small
> local models can sometimes plan a trivial single-node graph but fail as soon
> as the plan needs multiple nodes, parallel branches, or a reduce step. Use a
> capable/hosted model for the planner and run small/local models as the
> `worker_model` (see recipe 3). Every model tested worked fine as a *worker*.

### Tested models (planner reliability)

Observed in local LM Studio + cloud testing (small samples; planning is
non-deterministic, so treat as directional):

| Model | Size | As planner | As worker |
|-------|------|-----------|-----------|
| OpenAI `gpt-5.4-nano` (cloud) | — | ✅ reliable across simple + complex graphs | ✅ |
| Anthropic `claude-haiku-4-5` (cloud) | — | ✅ reliable (tool-calling structured output) | ✅ |
| `google/gemma-3-27b` (local) | 27B | ⚠️ works, but occasionally flaky | ✅ |
| `openai/gpt-oss-20b` (local) | 20B | ❌ emitted invalid `GraphSpec`s even on a moderate graph | ✅ |
| `google/gemma-4-e4b` (local) | ~4B eff. | ⚠️ planned a simple single-node graph; **failed a multi-source graph 3/3** (2× explicit `plan_failed`) | ✅ |

Takeaways: it is **not** a clean "bigger = better" — `gemma-4-e4b` (small)
handled a trivial plan that `gpt-oss-20b` (larger) botched, yet both fail once
the graph gets non-trivial. The dependable pattern is **cloud/capable planner +
any worker** (recipe 3). For fully-local planning, prefer a larger instruct
model and keep prompts simple.

### Rough latency

Wall-clock for `engine.run()` on **one simple single-output graph**. Cloud rows
ran planner+worker on the named model; local rows ran the model as the
`worker_model` behind a cloud planner. `cold` = first call (includes LM Studio
model load/switch); `warm` = an immediate repeat. Hardware-dependent and
directional only.

**Test machine:** Intel Core i9-14900HX (24C/32T), 64 GB RAM, NVIDIA RTX 4060
Laptop GPU (**8 GB VRAM**), Windows 11; local models served by LM Studio. The
8 GB VRAM ceiling matters: models that fit run on-GPU and are fast, while
larger ones (e.g. 27B) spill to CPU/RAM and slow down sharply — your numbers
will differ a lot with more VRAM.

| Config | Cold | Warm |
|--------|------|------|
| `gpt-5.4-nano` (cloud, planner+worker) | 6.6s | 4.3s |
| `claude-haiku-4-5` (cloud, planner+worker) | 8.4s | 4.3s |
| `gpt-oss-20b` (local worker + cloud planner) | 16.3s | 6.5s |
| `gemma-4-e4b` (local worker + cloud planner) | 34.6s | 34.2s |
| `gemma-3-27b` (local worker + cloud planner) | 67.2s | 50.8s |

- Cloud models land around **4–8s** for a simple graph (~4s warm).
- Local worker latency is dominated by the local model + LM Studio: on this
  machine `gpt-oss-20b` was fast once warm (~6.5s, near-cloud), while the gemma
  models were much slower (`gemma-3-27b` 50–67s; `gemma-4-e4b` ~34s with little
  cold/warm difference).
- **Scales with graph size:** each node is a sequential model call, so a plan
  with N llm/reduce nodes costs roughly N× the per-call latency above. Keep
  this in mind for both cost and wall-clock when planning larger graphs.

## 1. Visual-only recording

**When:** you want a picture of the graph that was generated (to embed in a PR
or doc) without the trace/output/spec noise.

```python
engine = DynamicSubgraphs(EngineConfig(
    model=Model("openai", "gpt-5.4-nano"),
    recording=Recording.visual_only(),   # only graph.mmd
))
result = engine.run("Compare two sources and recommend one.", run_id="demo")
print(result.artifacts["graph.mmd"])   # the Mermaid diagram, Artifact.MERMAID
```

## 2. Local model (LM Studio / Ollama)

**When:** you want to run fully offline / on your own hardware.

```python
engine = DynamicSubgraphs(EngineConfig(model=Model.lmstudio("google/gemma-3-27b")))
engine = DynamicSubgraphs(EngineConfig(model=Model.ollama("llama3.1")))
```

**Tip:** `Model.lmstudio()` / `Model.openai_compatible()` default
`structured_method="json_schema"` because local OpenAI-compatible servers
reject the forced `tool_choice` that `function_calling` uses. And remember the
planner-capability warning above — small models make poor planners.

## 3. Hybrid: cheap cloud planner + local workers

**When:** you want a reliable planner but cheap/private execution. This is the
**recommended** way to use small/local models. Unset roles fall back to the
worker model, so set `planner_model` explicitly.

```python
engine = DynamicSubgraphs(EngineConfig(
    planner_model=Model("openai", "gpt-5.4-nano"),     # capable planner
    worker_model=Model.lmstudio("openai/gpt-oss-20b"),  # small local worker
))
# Or override per run — each run can pick its own models:
result = engine.run("Investigate X.", worker_model=Model.ollama("llama3.1"))
```

## 4. Debugging a failed plan

**When:** a run didn't do what you expected and you want the full trace.

```python
engine = DynamicSubgraphs(EngineConfig(
    model=Model("openai", "gpt-5.4-nano"),
    recording=Recording.debug(),   # everything
))
result = engine.run("...", run_id="dbg")
if not result.ok:
    print(result.status, result.errors)          # e.g. plan_failed + reason
    print(result.artifacts["summary.md"])        # Artifact.SUMMARY
    print(result.artifacts["trace.jsonl"])       # Artifact.TRACE, step timings
    print(result.artifacts["prompt.md"])         # Artifact.PROMPT, the input
print(result.plan)                                # the generated GraphSpec
```

`Recording.debug()` captures every artifact: `Artifact.SPEC`, `Artifact.TRACE`,
`Artifact.OUTPUT`, `Artifact.MERMAID`, `Artifact.SUMMARY`, `Artifact.PROMPT`,
and `Artifact.EMITTED`.

## 5. Everything except the spec

**When:** you want full recording but not the (large) `spec.json` — e.g. you
won't replay this run. `resume`/`replay` need `Artifact.SPEC`, so they'll fail
loudly on this run.

```python
engine = DynamicSubgraphs(EngineConfig(
    model=Model(...),
    recording=Recording.all() - {Artifact.SPEC},
))
```

## 6. Minimal replayable run

**When:** you want to be able to `replay`/`resume` later with the least on disk.
`Recording.replayable()` keeps `Artifact.SPEC` + `Artifact.OUTPUT`.

```python
engine = DynamicSubgraphs(EngineConfig(
    model=Model(...), recording=Recording.replayable(),
))
```

## 7. Capturing tool/report outputs

**When:** your graph uses `emit_artifact` nodes and you want those files on disk.
`Artifact.EMITTED` toggles the on-disk sink (vs in-memory).

```python
engine = DynamicSubgraphs(EngineConfig(
    model=Model(...), recording={Artifact.OUTPUT, Artifact.EMITTED},
))
```

## 8. Embed without clutter (default)

**When:** you're embedding the engine in another app and don't want files. This
is the default (no `recording`) — results still come back in memory.

```python
engine = DynamicSubgraphs(EngineConfig(model=Model("openai", "gpt-5.4-nano")))
result = engine.run("...")
print(result.response, result.values)   # nothing written to disk
```

## 9. Discover options programmatically (for agents)

**When:** an LLM/agent is driving the SDK and shouldn't guess option strings.

```python
caps = DynamicSubgraphs.capabilities()
caps["providers"]           # ["anthropic", "ollama", "openai"]
caps["artifacts"]           # ["spec.json", "trace.jsonl", ..., "emitted"]
caps["statuses"]            # ["ok", "plan_failed", ...]
result.to_dict()            # JSON-safe view of any RunResult
```
