from enum import StrEnum


class NodeKind(StrEnum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    SPAWN_SUBAGENT = "spawn_subagent"
    PARALLEL_MAP = "parallel_map"
    REDUCE = "reduce"
    BRANCH = "branch"
    WAIT_FOR_EVENT = "wait_for_event"
    EMIT_ARTIFACT = "emit_artifact"
    SPAWN_SUBGRAPH = "spawn_subgraph"
