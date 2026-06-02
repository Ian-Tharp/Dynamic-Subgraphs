from __future__ import annotations

from pydantic import ValidationError as PydanticValidationError

from app.models.graph_spec import NodeSpec
from app.models.node_kinds import NodeKind
from app.registry.allowlists import DEFAULT_SUBAGENTS, DEFAULT_TOOLS, FORBIDDEN_KINDS
from app.registry.definitions import NodeKindDefinition, default_kind_definitions
from app.registry.errors import RegistryValidationError, RegistryValidationIssue
from app.registry.params import ParallelMapParams, ToolCallParams


class Registry:
    """Bounded vocabulary: which node kinds exist and what they may reference."""

    def __init__(
        self,
        *,
        tools: frozenset[str] | None = None,
        subagents: frozenset[str] | None = None,
        kinds: dict[NodeKind, NodeKindDefinition] | None = None,
    ) -> None:
        self._tools = tools if tools is not None else DEFAULT_TOOLS
        self._subagents = subagents if subagents is not None else DEFAULT_SUBAGENTS
        self._kinds = kinds if kinds is not None else default_kind_definitions()

    @property
    def tools(self) -> frozenset[str]:
        return self._tools

    @property
    def subagents(self) -> frozenset[str]:
        return self._subagents

    def allowed_kinds(self) -> frozenset[NodeKind]:
        return frozenset(self._kinds.keys())

    def get_definition(self, kind: NodeKind) -> NodeKindDefinition | None:
        return self._kinds.get(kind)

    def is_allowed_kind(self, kind: NodeKind | str) -> bool:
        if isinstance(kind, str):
            if kind in FORBIDDEN_KINDS:
                return False
            try:
                kind = NodeKind(kind)
            except ValueError:
                return False
        return kind in self._kinds

    def validate_node_params(self, node: NodeSpec) -> NodeSpec:
        """Validate params; return node with normalized params dict."""
        issues: list[RegistryValidationIssue] = []

        if str(node.kind) in FORBIDDEN_KINDS:
            issues.append(
                RegistryValidationIssue(
                    code="forbidden_kind",
                    message=f"Node kind '{node.kind}' is not permitted in v1",
                    node_id=node.id,
                )
            )

        definition = self._kinds.get(node.kind)
        if definition is None:
            issues.append(
                RegistryValidationIssue(
                    code="unknown_kind",
                    message=f"Unknown node kind '{node.kind}'",
                    node_id=node.id,
                    field="kind",
                )
            )
            raise RegistryValidationError(issues)

        try:
            parsed = definition.param_model.model_validate(node.params)
        except PydanticValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                issues.append(
                    RegistryValidationIssue(
                        code="invalid_params",
                        message=err["msg"],
                        node_id=node.id,
                        field=f"params.{loc}" if loc else "params",
                    )
                )
            raise RegistryValidationError(issues) from exc

        ref_issues = self._validate_references(node, parsed, definition)
        if ref_issues:
            raise RegistryValidationError(ref_issues)

        normalized = node.model_copy(update={"params": parsed.model_dump()})
        return normalized

    def _validate_references(
        self,
        node: NodeSpec,
        params: object,
        definition: NodeKindDefinition,
    ) -> list[RegistryValidationIssue]:
        issues: list[RegistryValidationIssue] = []

        if definition.requires_tool_allowlist and isinstance(params, ToolCallParams):
            if params.tool_name not in self._tools:
                issues.append(
                    RegistryValidationIssue(
                        code="tool_not_allowlisted",
                        message=f"Tool '{params.tool_name}' is not in the registry allowlist",
                        node_id=node.id,
                        field="params.tool_name",
                    )
                )

        if isinstance(params, ParallelMapParams):
            child = params
            if child.child_kind == "tool_call":
                tool_name = child.child_params.get("tool_name")
                if not tool_name:
                    issues.append(
                        RegistryValidationIssue(
                            code="missing_child_tool",
                            message="parallel_map child_params must include tool_name",
                            node_id=node.id,
                            field="params.child_params.tool_name",
                        )
                    )
                elif tool_name not in self._tools:
                    issues.append(
                        RegistryValidationIssue(
                            code="tool_not_allowlisted",
                            message=f"Child tool '{tool_name}' is not allowlisted",
                            node_id=node.id,
                            field="params.child_params.tool_name",
                        )
                    )
            elif child.child_kind == "llm_call":
                if not child.child_params.get("instruction"):
                    issues.append(
                        RegistryValidationIssue(
                            code="missing_child_instruction",
                            message="parallel_map llm child_params must include instruction",
                            node_id=node.id,
                            field="params.child_params.instruction",
                        )
                    )

        if definition.requires_subagent_allowlist:
            from app.registry.params import SpawnSubagentParams

            if isinstance(params, SpawnSubagentParams):
                if params.agent_name not in self._subagents:
                    issues.append(
                        RegistryValidationIssue(
                            code="subagent_not_allowlisted",
                            message=f"Subagent '{params.agent_name}' is not allowlisted",
                            node_id=node.id,
                            field="params.agent_name",
                        )
                    )

        return issues

    def counts_as_llm_call_for_node(self, node: NodeSpec) -> bool:
        """Whether executing this single node consumes one LLM call.

        Mirrors the per-node logic in `count_llm_calls` (the kind's
        `counts_as_llm_call` flag, plus `reduce`'s `llm_summarize` strategy)
        so the runtime ledger and the planner's static estimate agree on what
        counts. `parallel_map` is excluded here: its per-fan-out child calls
        are counted at runtime by the worker, not by the dispatcher node.
        """
        definition = self._kinds.get(node.kind)
        if definition is None:
            return False
        if definition.counts_as_llm_call:
            return True
        if node.kind == NodeKind.REDUCE:
            return node.params.get("strategy", "concat") == "llm_summarize"
        return False

    def count_llm_calls(self, nodes: list[NodeSpec]) -> int:
        total = 0
        for node in nodes:
            definition = self._kinds.get(node.kind)
            if definition is None:
                continue
            if definition.counts_as_llm_call:
                total += 1
            elif node.kind == NodeKind.REDUCE:
                strategy = node.params.get("strategy", "concat")
                if strategy == "llm_summarize":
                    total += 1
            elif node.kind == NodeKind.PARALLEL_MAP:
                child_kind = node.params.get("child_kind")
                if child_kind == NodeKind.LLM_CALL.value:
                    total += 1  # upper bound per fan-out handled at runtime
        return total


default_registry = Registry()
