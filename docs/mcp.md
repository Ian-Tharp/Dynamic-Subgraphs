# MCP server

`python -m app.mcp` exposes governed subgraph runs over the
[Model Context Protocol](https://modelcontextprotocol.io), so an MCP host
(Claude Desktop, Claude Code, an IDE, or your own agent) can hand a task to a
LangGraph workflow that is **planned, validated, budget-capped, and recorded**
instead of running it in a free-form tool loop.

The server is a thin adapter over the same `Supervisor` the SDK and HTTP API use:
the agent gets the plan → validate → run → record contract, never raw code
execution.

## Install and run

```bash
pip install "dynamic-subgraphs[mcp]"            # + openai / anthropic / ollama for real models
python -m app.mcp                                # stdio (default)
python -m app.mcp --transport streamable-http    # HTTP on 127.0.0.1:8765/mcp
```

From a checkout: `uv sync --all-extras && uv run python -m app.mcp`.

It boots in **mock** mode (token-free, offline). Logs go to stderr; on stdio,
stdout carries the protocol.

### Register with an MCP host

Claude Code:

```bash
claude mcp add dynamic-subgraphs -- uv --directory /path/to/Dynamic-Subgraphs run python -m app.mcp
```

Claude Desktop (`claude_desktop_config.json`) or any host using the common
`mcpServers` format:

```json
{
  "mcpServers": {
    "dynamic-subgraphs": {
      "command": "uv",
      "args": ["--directory", "/path/to/Dynamic-Subgraphs", "run", "python", "-m", "app.mcp"],
      "env": {
        "DS_PLANNER": "llm",
        "DS_PROVIDER": "openai",
        "DS_MODEL": "gpt-5.4-nano",
        "OPENAI_API_KEY": "sk-...",
        "DS_MCP_MAX_LLM_CALLS": "4",
        "DS_RUNS_DIR": "/path/to/runs"
      }
    }
  }
}
```

## Tools

| Tool | Read-only | What it does |
|------|-----------|--------------|
| `run_subgraph` | no | Plan, validate, run, and record a subgraph for `prompt`. Optional `run_id`, `planner`, `provider`, `model`, and per-run limits `max_nodes`, `max_llm_calls`, `max_wall_seconds`. |
| `resume_run` | no | Feed `event` to a run paused on `wait_for_event`. |
| `replay_run` | no | Re-execute a recorded plan verbatim under a new id (no re-plan; re-validated against the current policy). |
| `server_info` | yes | The host policy, planner, and model allowlist. |
| `describe_registry` | yes | Node kinds, tools, and subagents available under the host policy. |
| `list_runs` | yes | Recorded runs (id, status, node count, eval scores). |
| `get_run` | yes | Outcome, values, errors, config, readable files, and artifacts for a run. |
| `get_run_file` | yes | One recorded file: `spec`, `output`, `trace`, `graph`, `summary`, `prompt`, `eval`, `receipt`. |
| `get_artifact` | yes | A file written by an `emit_artifact` node. |
| `get_receipt` | yes | A run's stored receipt plus its receipt-ledger entry and history (`supersedes` links a resumed receipt to the paused one). |
| `verify_receipt` | yes | Re-check a run's receipt against its recorded files and the ledger: `ok`, `problems`, `signature` (`valid` / `invalid` / `unsigned` / `unchecked`), and whether it was issued under the unnarrowed host policy. |
| `verify_ledger` | yes | Check the whole ledger: hash chain, `seq`, `supersedes` links, signatures. Returns `ok`, `problems`, `entries`, `signed`, `head`. |

`run_subgraph` / `resume_run` / `replay_run` return:

```jsonc
{
  "run_id": "run-1a2b3c4d5e6f",
  "status": "ok",            // ok | paused | plan_failed | validation_failed | policy_denied | execution_failed | ...
  "ok": true,
  "healthy": true,           // ok or paused
  "response": "...",
  "values": {"recommendation": "..."},
  "errors": [],
  "plan": {"graph_id": "...", "nodes": [{"id": "...", "kind": "llm_call"}], "edges": [...]},
  "granted_budget": {"max_nodes": 6, "max_llm_calls": 4, ...},  // min(host, planner request)
  "host_policy": {"max_nodes": 8, ...},                          // the ceiling for this run
  "plan_attempts": 1,
  "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "by_model": {}},
  "artifacts": [],
  "recorded": true,
  "receipt": {"status": "ok", "policy_hash": "...", "plan_hash": "...", "receipt_hash": "...", ...}  // sealed provenance receipt; see docs/governance.md
}
```

Outside the server, `ds verify --runs-dir <DS_RUNS_DIR>` checks the same runs and
ledger offline (see [governance](./governance.md#ds-verify)).

Long strings in `response` / `values` / `errors` are clipped to
`DS_MCP_MAX_VALUE_CHARS` so results fit an agent's context. The full values stay
on disk; read them with `get_run_file(name="output")`.

A refused request (disallowed planner or model, unsafe or duplicate run id,
unknown run) comes back as an MCP tool error with the reason. A run that starts
and then fails is **not** a tool error: you get the status and errors, and the
failure is recorded.

## Resources

| URI | Content |
|-----|---------|
| `ds://server` | Same as `server_info` (JSON). |
| `ds://registry` | Same as `describe_registry` (JSON). |
| `ds://runs/{run_id}/{name}` | A recorded run file (same names as `get_run_file`). |

## Governance at the MCP boundary

| Rule | How it's enforced |
|------|-------------------|
| The host owns the limits | Every run is validated against the server's `ExecutionPolicy`, including nested `spawn_subgraph` children. A client can **narrow** `max_nodes` / `max_llm_calls` / `max_wall_seconds` for one run; requests above the ceiling are capped, never honored. |
| No surprise token spend | `planner="mock"` always works. `planner="llm"` is refused unless the server runs with `DS_PLANNER=llm`. Models must be in `DS_MODEL_ALLOWLIST`. Missing provider credentials are refused before any work starts. LLM runs use strict runners (no silent echo fallback). |
| Every run is recorded | Recording is always on, for failures too. Each run leaves `spec.json`, `trace.jsonl`, `output.json`, `graph.mmd`, and `summary.md` under `runs/<run_id>/`. |
| No clobbering | Client-supplied ids are validated (no path traversal) and must be new. Replays always write to a new id. Two calls can't run, replay into, or resume the same id at once. |
| Stays governed on resume/replay | Resume and replay reuse the planner/model the run was created with, but only if the server still allows them, and replay re-validates the recorded plan against the **current** policy. |
| Bounded reads | Run files are read from a fixed set of names; artifacts go through the recorder's path-safe lookup; reads are size-capped. |
| Anchored receipts | Every persisted receipt is appended to a hash-chained receipt ledger (`<runs_dir>/.ledger/receipts.jsonl`), signed when a key is configured. `verify_receipt` flags a `receipt.json` that was rewritten and resealed, or rolled back to an older receipt. Run ids starting with `.` are refused so no run can claim the ledger's directory. |
| Fail closed on config | A malformed `DS_*` value stops the server at startup instead of falling back to a looser default. A signing key that's too short or can't be loaded is a startup error too. |

## Configuration

Model selection uses the same variables as the HTTP API, so one environment
configures both.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DS_PLANNER` | `mock` | `mock` (token-free) or `llm`. `openai` is a legacy alias for `llm` + `openai`. |
| `DS_PROVIDER` | `openai` | `openai`, `anthropic`, `ollama`, ... |
| `DS_MODEL` | `gpt-5.4-nano` | Default model. |
| `DS_MODEL_ALLOWLIST` | `$DS_MODEL` | Comma-separated; entries may be `model`, `provider:model`, or `*`. |
| `DS_RUNS_DIR` | `runs` | Where runs are recorded. |
| `DS_MCP_MAX_NODES` | `12` | Host ceiling on nodes per graph. |
| `DS_MCP_MAX_LLM_CALLS` | `8` | Host ceiling on LLM calls per run. |
| `DS_MCP_MAX_DEPTH` | `2` | Nesting depth for `spawn_subgraph` (hard rail: 3). |
| `DS_MCP_MAX_FANOUT` | `64` | Max `parallel_map` items. |
| `DS_MCP_MAX_WALL_SECONDS` | `90` | Wall-clock limit per run. |
| `DS_MCP_ALLOWED_TOOLS` | unset (registry set) | Comma-separated. Empty string bans all tools. `*` = no narrowing. |
| `DS_MCP_ALLOWED_SUBAGENTS` | unset | Same rules as tools. |
| `DS_MCP_ALLOWED_NODE_KINDS` | unset | Same rules; unknown kinds are a startup error. |
| `DS_MCP_MAX_PLAN_ATTEMPTS` | `2` | Plan-repair attempts (`1` = block on first invalid plan). |
| `DS_MCP_MAX_VALUE_CHARS` | `4000` | Per-string clip for values returned in tool results. |
| `DS_MCP_TRANSPORT` | `stdio` | `stdio` or `streamable-http` (`--transport` overrides). |
| `DS_MCP_HOST` / `DS_MCP_PORT` | `127.0.0.1` / `8765` | HTTP bind address (`--host` / `--port` override). |
| `DS_RECEIPT_LEDGER` | `$DS_RUNS_DIR/.ledger/receipts.jsonl` | Receipt ledger path; `off` disables it. |
| `DS_RECEIPT_HMAC_KEY` | unset | Sign ledger entries with HMAC-SHA256 (secret, >= 32 bytes). |
| `DS_RECEIPT_ED25519_KEY_FILE` | unset | Sign ledger entries with this Ed25519 PEM private key (needs the `signing` extra; `ds keygen` makes one). Set at most one of the two keys. |
| `DS_RECEIPT_KEY_ID` | `default` | Key id written on each signature. |

## Limits of this version

- **Paused runs live in memory.** The checkpointer is in-process, so
  `resume_run` works for runs paused in the same server process. A restart
  loses paused state (the recording is kept).
- **Runs are synchronous.** A tool call returns when the run finishes, bounded
  by `max_wall_seconds`. There is no progress streaming yet.
- **No auth on the HTTP transport.** It binds to localhost by default. Put it
  behind your own gateway before exposing it, or use stdio.
- **No eval gate yet.** `get_run` shows `eval.json` when a run was scored
  elsewhere, but the server doesn't score runs itself.

## Using the service without MCP

The governance lives in `app.mcp.service.GovernedRunService`, which has no MCP
dependency:

```python
from app.mcp import GovernedRunService, McpSettings

service = GovernedRunService(McpSettings.from_env())
out = service.run("Compare A and B", max_llm_calls=2)
print(out["status"], out["granted_budget"])
```
