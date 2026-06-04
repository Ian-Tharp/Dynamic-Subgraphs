"""Default allowlists for v1 — expand via Registry constructor, not runtime mutation."""

# `mock_document_extract` was removed: it echoed empty content, so a planner
# that selected it for a retrieval/compare task produced a dead-end run (it
# returns no document text). Retrieval should route to `web_search`. See
# docs/evals/model-comparison-2026-06.md.
DEFAULT_TOOLS: frozenset[str] = frozenset(
    {
        "policy_lookup",
        "create_follow_up_task",
        "web_search",
    }
)

DEFAULT_SUBAGENTS: frozenset[str] = frozenset(
    {
        "document_specialist",
        "critic",
    }
)

# Kinds explicitly withheld from v1 (design §7).
FORBIDDEN_KINDS: frozenset[str] = frozenset(
    {
        "python_eval",
        "shell",
        "arbitrary_network",
        "call_any_tool",
    }
)
