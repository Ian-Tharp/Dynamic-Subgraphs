"""Default allowlists for v1 — expand via Registry constructor, not runtime mutation."""

DEFAULT_TOOLS: frozenset[str] = frozenset(
    {
        "mock_document_extract",
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
