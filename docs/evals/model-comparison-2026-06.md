# Eval: model comparison — gpt-5.4-nano vs claude-haiku-4-5 (2026-06)

End-to-end evaluation of the Dynamic Subgraphs engine on real models, comparing
OpenAI `gpt-5.4-nano` and Anthropic `claude-haiku-4-5`. The goal: confirm the
engine produces genuinely valuable responses, and characterize the cost /
latency / quality trade-off between a cheap and a premium model.

## Method

- **Path:** full SDK e2e per run — `DynamicSubgraphs(EngineConfig(model=...)).run(prompt)`
  → plan → validate → compile → execute → respond.
- **Models:** `gpt-5.4-nano` (OpenAI) and `claude-haiku-4-5` (Anthropic), each
  as planner **and** worker.
- **Scenarios (3):** `compare` (compare two evidence sources on remote-work
  productivity), `research` (nuclear vs solar for a mid-size city), `summarize`
  (pros/cons of a 4-day work week).
- **Observability:** LangSmith tracing on, project **`DS-model-comparison`**.
  Token counts captured locally via `UsageMetadataCallbackHandler` and
  cross-checked against LangSmith; **cost** is LangSmith-computed.
- **Machine (for latency):** Intel i9-14900HX, 64 GB RAM, RTX 4060 Laptop (8 GB
  VRAM), Windows 11. Cloud-model latency is network-bound, not machine-bound.

## Results

All 6 runs completed (`status: ok`) — the engine planned valid governed graphs
and produced real output on both models.

| Scenario | Model | Latency | Nodes | Tokens (in/out) | Cost (USD) |
|----------|-------|--------:|:-----:|----------------:|-----------:|
| compare   | nano  | 7.0s  | 3 | 2,716 / 611   | $0.0010 |
| compare   | haiku | 20.4s | 3 | 6,007 / 1,469 | $0.0134 |
| research  | nano  | 19.2s | 3 | 3,428 / 2,149 | $0.0034 |
| research  | haiku | 96.3s | 6 | 28,919 / 9,943 | $0.0786 |
| summarize | nano  | 4.3s  | 2 | 2,765 / 448   | $0.0011 |
| summarize | haiku | 6.7s  | 2 | 3,669 / 559   | $0.0065 |
| **total** | nano  | 30.5s | — | 12.1K | **~$0.0055** |
| **total** | haiku | 123.4s | — | 50.6K | **~$0.099** |

Implied per-token pricing (from LangSmith cost): `gpt-5.4-nano` ≈ **$0.20/M in,
$1.25/M out**; `claude-haiku-4-5` ≈ **$1.00/M in, $5.00/M out**.

**Headline:** haiku cost **~18×** more and ran **~4×** slower overall. The
research task was the extreme: haiku planned a **6-node** graph and emitted ~10K
output tokens → **23×** the cost of nano's lean 3-node plan ($0.079 vs $0.0034).

## Quality

Both models produced coherent, well-structured, genuinely useful output — the
engine is producing value, not just running.

- **research:** Both strong. nano gave a tight 8-dimension analysis with a clear
  recommendation and phased plan. haiku gave a consultant-grade 9-section report
  with concrete figures and a decision matrix — more exhaustive, a couple of
  fabricated specifics, 23× the cost.
- **summarize:** haiku noticeably more professional; nano solid but generic.
- **compare:** surfaced an engine finding (below).

## Findings & actions

1. **`mock_document_extract` removed from the default tool allowlist.**
   On `compare`, nano's planner selected the `mock_document_extract` tool (a
   placeholder that echoes *empty* content), so the run honestly dead-ended
   ("I can't compare — provide the sources"). haiku's planner selected the real
   `web_search` tool, grounded its answer in actual studies, and produced a
   substantive critique. A planner being *able* to pick a no-op tool undercuts
   the engine's value on open-ended prompts, so `mock_document_extract` was
   removed from `DEFAULT_TOOLS` (and the echo-tool map); retrieval now routes to
   `web_search`. *(The remaining `policy_lookup` / `create_follow_up_task`
   defaults are still echo placeholders pending real implementations.)*

2. **Planner style differs by model.** haiku builds larger, more thorough graphs
   (more nodes, longer node instructions, web-search-grounded); nano builds lean
   graphs. This is the main driver of haiku's cost/latency premium.

## Recommendation

- **Default to `nano`** (or a nano-class model) for most work — fast, cheap, and
  frequently "good enough" or genuinely strong (its research output was
  excellent and concise).
- **Use `haiku`** (or larger) for depth-critical, low-volume tasks where the
  extra thoroughness justifies ~18× cost and ~4× latency.
- **Hybrid** (per-role models) lets you put a capable planner over a cheaper
  worker — see `docs/recipes.md`.

## Reproduce

Run each scenario through both models with tracing on (requires
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `LANGSMITH_API_KEY` in `.env`), e.g.:

```python
from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

for m in (Model("openai", "gpt-5.4-nano"), Model("anthropic", "claude-haiku-4-5")):
    engine = DynamicSubgraphs(EngineConfig(model=m))
    result = engine.run("Research nuclear vs solar for a mid-size city ...")
    print(result.status, len(result.plan.nodes), result.values)
```

Set `LANGSMITH_PROJECT` to a dedicated name to isolate the run, then read the
traces via the LangSmith MCP tools or UI. Token usage is also available locally
via `langchain_core.callbacks.UsageMetadataCallbackHandler` (attach it through
`Model(..., extra_kwargs={"callbacks": [handler]})`).
