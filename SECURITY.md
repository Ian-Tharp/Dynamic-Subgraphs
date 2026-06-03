# Security Policy

## Supported versions

Dynamic Subgraphs is pre-1.0. Security fixes are applied to the latest released
`0.x` version. Pin a version in production and upgrade promptly.

## Reporting a vulnerability

**Please do not open a public issue for security problems.** Instead, report
privately via GitHub Security Advisories
("Report a vulnerability" on the repository's Security tab), or email
**praht09ian@gmail.com**.

Include reproduction steps and affected versions. We aim to acknowledge within
a few days and will coordinate a fix and disclosure timeline with you.

## Handling secrets

This SDK reads provider credentials from the environment (e.g.
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and an optional `TAVILY_API_KEY`.

- Keep secrets in a local `.env` (already git-ignored) or your platform's
  secret manager. Never commit real keys — see [`.env.example`](./.env.example).
- Local providers (LM Studio / Ollama) need no key; use them for fully offline
  runs.
- Run recording (`record=...`) writes prompts, traces, and outputs to
  `runs/<run_id>/`. Treat that directory as potentially sensitive and keep it
  out of version control (it is git-ignored by default).

## Safety model

By design the planner emits **plans (a validated `GraphSpec`), not executable
code**. The compiler only instantiates **registry-approved node kinds**, and
LangGraph types stay behind the compiler/runtime boundaries. There is no
arbitrary code execution path from model output in v1. Tool-calling nodes are
limited to the allowlisted tools in the registry. If you extend the registry
with new node kinds or tools, you are responsible for their safety.
