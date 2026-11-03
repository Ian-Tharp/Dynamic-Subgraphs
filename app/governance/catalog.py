"""Approved-subgraph catalog: governed routing over pre-reviewed GraphSpecs.

In catalog mode the host restricts execution to a fixed set of reviewed
GraphSpecs. The planner no longer writes a plan; it *selects* one entry
(``CatalogPlanner``), and the plan gate refuses any plan that does not match an
entry. This mirrors the benchmark's finding that a fixed router over reviewed
subgraphs beats free planning, and gives auditors a closed list of shapes.

Entries are templates: any string in a node's params may contain the literal
``{prompt}`` placeholder, filled with the run's prompt (``fill_spec``). Matching
a candidate plan against a template treats ``{prompt}`` as a wildcard that must
take the **same** value everywhere it appears, so a plan matches iff it is the
template filled with some prompt. ``budget``, ``goal``, ``rationale`` and
``metadata`` are not compared: the host policy clamps the budget anyway, and the
rest is descriptive.

Layering: imports ``app.models`` and ``app.registry`` (for param normalization).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from app.errors import DynamicSubgraphsError
from app.governance.hashing import plan_hash
from app.models.graph_spec import GraphSpec, NodeSpec
from app.registry.errors import RegistryValidationError
from app.registry.registry import Registry

PROMPT_PLACEHOLDER = "{prompt}"


class CatalogSelectionError(DynamicSubgraphsError, ValueError):
    """The catalog selector returned no entry, or an id that is not in the catalog."""


def _fill_value(value: Any, prompt: str) -> Any:
    if isinstance(value, str):
        return value.replace(PROMPT_PLACEHOLDER, prompt)
    if isinstance(value, dict):
        return {k: _fill_value(v, prompt) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_value(item, prompt) for item in value]
    return value


def fill_spec(spec: GraphSpec, prompt: str) -> GraphSpec:
    """Deep copy of ``spec`` with ``{prompt}`` replaced in every node's params.

    Uses ``str.replace`` (never ``str.format``), so prompts containing braces,
    quotes or backslashes are safe. The input spec is not mutated.
    """
    copy = spec.model_copy(deep=True)
    for node in copy.nodes:
        node.params = _fill_value(node.params, prompt)
    return copy


@dataclass(frozen=True)
class CatalogEntry:
    """One pre-approved GraphSpec template.

    ``entry_id`` is the stable routing key; ``spec_hash`` (computed) is the
    content hash of the template, recorded on the receipt so an auditor can tie
    a run to the exact reviewed version.
    """

    entry_id: str
    spec: GraphSpec
    description: str = ""
    keywords: tuple[str, ...] = ()

    @property
    def spec_hash(self) -> str:
        return plan_hash(self.spec)


def _normalized_shape(spec: GraphSpec, registry: Registry) -> dict[str, Any]:
    """The compared view of a spec: graph id, normalized nodes, sorted edges."""
    nodes: list[dict[str, Any]] = []
    for node in spec.nodes:
        try:
            norm: NodeSpec = registry.validate_node_params(node)
        except RegistryValidationError:
            norm = node  # unknown params: compare raw; the validator rejects later
        nodes.append(norm.model_dump(mode="json", by_alias=True))
    edges = sorted((e.from_, e.to) for e in spec.edges)
    return {"graph_id": spec.graph_id, "nodes": nodes, "edges": edges}


class _Wildcard:
    """Tracks the single value ``{prompt}`` must take across one comparison."""

    def __init__(self) -> None:
        self.value: str | None = None

    def match_str(self, template: str, candidate: str) -> bool:
        if PROMPT_PLACEHOLDER not in template:
            return template == candidate
        if self.value is not None:
            return template.replace(PROMPT_PLACEHOLDER, self.value) == candidate
        parts = [re.escape(p) for p in template.split(PROMPT_PLACEHOLDER)]
        # Every occurrence must capture the same text: back-reference the first.
        pattern = parts[0] + "(.*?)" + parts[1] + "".join(r"\1" + p for p in parts[2:])
        m = re.fullmatch(pattern, candidate, flags=re.DOTALL)
        if m is None:
            return False
        self.value = m.group(1)
        return True


def _matches(template: Any, candidate: Any, wild: _Wildcard) -> bool:
    if isinstance(template, str):
        return isinstance(candidate, str) and wild.match_str(template, candidate)
    if isinstance(template, dict):
        if not isinstance(candidate, dict) or set(template) != set(candidate):
            return False
        return all(_matches(template[k], candidate[k], wild) for k in sorted(template))
    if isinstance(template, list | tuple):
        if not isinstance(candidate, list | tuple) or len(template) != len(candidate):
            return False
        return all(
            _matches(t, c, wild) for t, c in zip(template, candidate, strict=True)
        )
    return bool(template == candidate)


Selector = Callable[[str, Sequence[CatalogEntry]], str]


def keyword_selector(prompt: str, entries: Sequence[CatalogEntry]) -> str:
    """Token-free default: the entry whose keywords appear most in the prompt.

    Ties go to catalog order; with no keyword hit the first entry wins (put the
    safest general-purpose entry first).
    """
    if not entries:
        raise CatalogSelectionError("catalog is empty")
    text = prompt.lower()
    best = entries[0]
    best_score = 0
    for entry in entries:
        score = sum(1 for kw in entry.keywords if kw.lower() in text)
        if score > best_score:
            best, best_score = entry, score
    return best.entry_id


@dataclass(frozen=True)
class SubgraphCatalog:
    """A closed set of approved GraphSpec templates.

    Put it on ``ExecutionPolicy.approved_catalog`` to turn on catalog mode: the
    plan gate then refuses any plan that is not a filled entry, and the SDK
    plans with ``CatalogPlanner`` (selection, not synthesis).

    ``selector`` picks an entry for a prompt; it defaults to
    ``keyword_selector`` (token-free). Use ``build_llm_catalog_selector`` for a
    model-backed choice that is still constrained to catalog ids.
    """

    entries: tuple[CatalogEntry, ...]
    selector: Selector | None = field(default=None, compare=False)
    registry: Registry | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        ids = [e.entry_id for e in self.entries]
        if not ids:
            raise ValueError("SubgraphCatalog needs at least one entry")
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"SubgraphCatalog has duplicate entry ids: {dupes}")
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(
            self, "_by_id", MappingProxyType({e.entry_id: e for e in self.entries})
        )

    def __hash__(self) -> int:
        return hash(tuple((e.entry_id, e.spec_hash) for e in self.entries))

    @classmethod
    def from_specs(
        cls,
        specs: Mapping[str, GraphSpec] | Iterable[GraphSpec],
        *,
        selector: Selector | None = None,
    ) -> SubgraphCatalog:
        """Build from ``{entry_id: spec}`` or specs keyed by their ``graph_id``."""
        if isinstance(specs, Mapping):
            pairs = list(specs.items())
        else:
            pairs = [(s.graph_id, s) for s in specs]
        return cls(
            entries=tuple(CatalogEntry(entry_id=k, spec=v) for k, v in pairs),
            selector=selector,
        )

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(e.entry_id for e in self.entries)

    def get(self, entry_id: str) -> CatalogEntry:
        by_id: Mapping[str, CatalogEntry] = self._by_id  # type: ignore[attr-defined]
        if entry_id not in by_id:
            raise CatalogSelectionError(
                f"{entry_id!r} is not in the approved catalog "
                f"(entries: {', '.join(self.ids)})"
            )
        return by_id[entry_id]

    def by_hash(self, spec_hash: str) -> CatalogEntry | None:
        return next((e for e in self.entries if e.spec_hash == spec_hash), None)

    def match(self, spec: GraphSpec) -> CatalogEntry | None:
        """The entry ``spec`` is a filled instance of, or None."""
        reg = self.registry or Registry()
        candidate = _normalized_shape(spec, reg)
        for entry in self.entries:
            if _matches(_normalized_shape(entry.spec, reg), candidate, _Wildcard()):
                return entry
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [
                {"entry_id": e.entry_id, "spec_hash": e.spec_hash} for e in self.entries
            ]
        }

    def select(self, prompt: str) -> CatalogEntry:
        chosen = (self.selector or keyword_selector)(prompt, self.entries)
        return self.get(chosen)


class CatalogPlanner:
    """Planner for catalog mode: picks an approved entry, never writes a plan.

    Satisfies the supervisor's ``Planner`` type (``prompt -> GraphSpec``). The
    returned spec is the entry filled with the prompt; its
    ``metadata.purpose`` records ``catalog:<entry_id>`` for readability (the
    plan gate re-matches the content and never trusts that field).
    """

    def __init__(self, catalog: SubgraphCatalog) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> SubgraphCatalog:
        return self._catalog

    def __call__(self, prompt: str) -> GraphSpec:
        entry = self._catalog.select(prompt)
        spec = fill_spec(entry.spec, prompt)
        spec.metadata = spec.metadata.model_copy(
            update={"purpose": f"catalog:{entry.entry_id}"}
        )
        return spec


def build_llm_catalog_selector(chat_model: Any) -> Selector:
    """A model-backed selector constrained to catalog ids.

    The model sees each entry's id + description and must answer with one id;
    anything else raises ``CatalogSelectionError`` (the supervisor reports
    ``plan_failed``). Works with any LangChain chat model.
    """

    def select(prompt: str, entries: Sequence[CatalogEntry]) -> str:
        menu = "\n".join(
            f"- {e.entry_id}: {e.description or e.spec.goal}" for e in entries
        )
        message = (
            "Choose the single best pre-approved workflow for the task. "
            "Reply with the workflow id only.\n\n"
            f"Workflows:\n{menu}\n\nTask:\n{prompt}"
        )
        reply = chat_model.invoke(message)
        text = str(getattr(reply, "content", reply)).strip().strip("`'\" .")
        ids = {e.entry_id for e in entries}
        if text not in ids:
            raise CatalogSelectionError(
                f"selector answered {text!r}, which is not a catalog id"
            )
        return text

    return select
