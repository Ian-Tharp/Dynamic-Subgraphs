"""The benchmark's authored arms are validated, frozen plans-as-data (PR7b)."""

import json
from pathlib import Path

import pytest

from app.models import GraphSpec
from app.registry import validate_graph_spec

LIB = Path("docs/evals/router-library-v1")
SINGLE = Path("docs/evals/fixed-host-graph-v1.json")


@pytest.mark.parametrize(
    "path", sorted(LIB.glob("*.json")), ids=lambda p: p.stem
)
def test_router_specs_validate(path: Path) -> None:
    spec = GraphSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))
    validate_graph_spec(spec)  # raises on invalid


def test_library_covers_the_task_types() -> None:
    assert {p.stem for p in LIB.glob("*.json")} == {
        "compare", "research", "summarize", "extract"
    }


def test_grounded_specs_carry_prompt_placeholder() -> None:
    """The bench harness substitutes {prompt} into tool_call args per task.

    Without this, the authored arms' searches could never reference the
    actual task — a strawman baseline. See the harness's fill step (bench.py).
    """
    for path in [LIB / "research.json", SINGLE]:
        raw = path.read_text(encoding="utf-8")
        assert "{prompt}" in raw, f"{path.name} lost its placeholder"


def test_single_fixed_graph_validates() -> None:
    spec = GraphSpec.model_validate(json.loads(SINGLE.read_text(encoding="utf-8")))
    validate_graph_spec(spec)


@pytest.mark.parametrize(
    "path",
    sorted([*LIB.glob("*.json"), SINGLE]),
    ids=lambda p: p.stem,
)
def test_specs_run_offline_on_mock_runners(path: Path, tmp_path: Path) -> None:
    from dynamic_subgraphs import DynamicSubgraphs, EngineConfig

    spec = GraphSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))
    r = DynamicSubgraphs(EngineConfig(planner="mock", runs_dir=tmp_path)).run(
        "Compare apples and oranges for a lunchbox.", fixed_spec=spec, record=False
    )
    assert r.ok, f"{path.stem}: {r.errors}"
