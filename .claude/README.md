# Agent memory

Compressed, durable knowledge from prior coding-agent sessions so future
sessions (Claude Code, Cursor, Copilot, etc.) can pick up the project
context without re-deriving it.

These files **complement, don't replace**:
- `AGENTS.md` — package map + MVP sequence (orientation)
- `ARCHITECTURE.md` — package boundaries + dependency direction
- `docs/dynamic-graphs-canonical-design-v1.md` — canonical design brief

## Files

| File | Read when |
|---|---|
| `context.md` | Starting a new session — snapshot of what's shipped |
| `patterns.md` | About to add a feature — follow the established shapes |
| `gotchas.md` | Hit a strange LangGraph / OpenAI / Pydantic error |
| `workflows.md` | Setting up the dev loop, writing tests, debugging |
| `roadmap.md` | Deciding what to build next |

## How to maintain

When you learn something a future agent shouldn't have to re-discover:
- LangGraph / OpenAI / Pydantic surprise → `gotchas.md`
- Repeatable architectural shape that worked → `patterns.md`
- Workflow improvement → `workflows.md`
- Project state changed → `context.md`
- New candidate slice or shifted priorities → `roadmap.md`

Keep entries **short, specific, and load-bearing**. Save another agent a
debugging cycle. Don't write a textbook.
