"""Smoke demo: GraphSpec -> Supervisor -> printed result.

Default: token-free, uses `StaticPlanner` with a hardcoded research-shaped spec.
With `--llm` (or DEMO_USE_LLM=1): uses `LLMPlanner` backed by ChatOpenAI
(gpt-5.4-nano). Reads OPENAI_API_KEY from `.env`.

    python -m app.main                  # mock planner, free
    python -m app.main --llm            # real planner, costs tokens
    python -m app.main --llm "your prompt here"
"""

from __future__ import annotations

# Allow `python app/main.py` from the project root by ensuring the repo root is
# on sys.path. Running `python -m app.main` does this automatically.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# LLM responses regularly contain Unicode (arrows, em dashes, curly quotes).
# Windows' default console encoding is cp1252, so printing them crashes. Force
# UTF-8 (replace on encode errors) so the demo never dies for display reasons.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

from app.models import (
    DynamicRunState,
    GraphSpec,
    NodeKind,
    NodeSpec,
)
from app.models.graph_spec import EdgeSpec, GraphBudget
from app.recording import FileRecorder
from app.runtime import (
    FileArtifactSink,
    LangGraphExecutor,
    NodeRunner,
    build_grounded_tool_runner,
    build_openai_llm_runner,
    build_openai_reduce_runner,
    build_openai_spawn_subagent_runner,
    make_emit_artifact_runner,
)
from app.supervisor import (
    Planner,
    StaticPlanner,
    Supervisor,
    SupervisorResult,
    build_openai_planner,
)

_LLM_REDUCE_STRATEGIES = {"concat", "merge_dict", "llm_summarize"}

DEFAULT_LLM_MODEL = "gpt-5.4-nano"
DEFAULT_PROMPT = "Compare two evidence sources and recommend one."


def mock_llm_runner(
    state: DynamicRunState,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Stand-in for a real LLM call. Echoes instruction and visible upstream keys."""

    instruction = str(params["instruction"])
    upstream_keys = sorted(state.get("values", {}).keys())
    suffix = f" [seen: {', '.join(upstream_keys)}]" if upstream_keys else ""
    return {"result": f"<mock-llm>{instruction}{suffix}</mock-llm>"}


def build_demo_spec() -> GraphSpec:
    """Static fallback spec used when --llm is not passed."""

    nodes = [
        NodeSpec(
            id="gather_left",
            kind=NodeKind.LLM_CALL,
            outputs=["left_evidence"],
            params={"instruction": "summarize source A"},
        ),
        NodeSpec(
            id="gather_right",
            kind=NodeKind.LLM_CALL,
            outputs=["right_evidence"],
            params={"instruction": "summarize source B"},
        ),
        NodeSpec(
            id="join",
            kind=NodeKind.REDUCE,
            inputs=["left_evidence", "right_evidence"],
            outputs=["combined_evidence"],
            params={
                "strategy": "concat",
                "input_keys": ["left_evidence", "right_evidence"],
                "output_key": "combined_evidence",
            },
        ),
        NodeSpec(
            id="synthesize",
            kind=NodeKind.LLM_CALL,
            inputs=["combined_evidence"],
            outputs=["final_answer"],
            params={"instruction": "produce a recommendation"},
        ),
    ]

    edges = [
        EdgeSpec.model_validate({"from": "START", "to": "gather_left"}),
        EdgeSpec.model_validate({"from": "START", "to": "gather_right"}),
        EdgeSpec.model_validate({"from": "gather_left", "to": "join"}),
        EdgeSpec.model_validate({"from": "gather_right", "to": "join"}),
        EdgeSpec.model_validate({"from": "join", "to": "synthesize"}),
        EdgeSpec.model_validate({"from": "synthesize", "to": "END"}),
    ]

    return GraphSpec(
        graph_id="demo-research-001",
        goal="compare two evidence sources and recommend one",
        rationale="parallel gather then concat then synthesize",
        budget=GraphBudget(max_nodes=8, max_llm_calls=6),
        nodes=nodes,
        edges=edges,
    )


def render_supervisor_result(result: SupervisorResult) -> None:
    print("=" * 64)
    print(f"  Status:   {result.status}")
    print(f"  Run id:   {result.run_id}")
    print(f"  Response: {result.response}")
    print("-" * 64)
    if result.validated_spec is not None:
        spec = result.validated_spec
        print(f"  Graph:    {spec.graph_id}")
        print(f"  Goal:     {spec.goal}")
        if spec.rationale:
            print(f"  Rationale:{spec.rationale}")
        print(f"  Nodes:    {len(spec.nodes)}   Edges: {len(spec.edges)}")
    if result.result is not None:
        print("-" * 64)
        print("  Final values:")
        for key, value in sorted(result.result.state.get("values", {}).items()):
            print(f"    {key}: {value!r}")
        print("  Trace:")
        for event in result.result.trace.events:
            duration = event.data.get("duration_ms")
            suffix = f"  ({duration} ms)" if duration is not None else ""
            node = event.node_id or "-"
            print(f"    {event.kind:<14}  node={node:<14}{suffix}")
    if result.record is not None:
        print("-" * 64)
        print(f"  Recorded run: {result.record.directory}")
        for name, path in sorted(result.record.artifacts.items()):
            print(f"    {name:<8} -> {path.name}")
    if result.errors:
        print("-" * 64)
        print("  Errors:")
        for entry in result.errors:
            print(
                f"    [{entry.get('stage')}] {entry.get('type')}: {entry.get('message')}"
            )
    print("=" * 64)


def _build_planner(*, use_llm: bool, model: str) -> Planner:
    if use_llm:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY is not set. Add it to .env or your environment."
            )
        return build_openai_planner(
            model=model,
            executable_reduce_strategies=_LLM_REDUCE_STRATEGIES,
        )
    return StaticPlanner(build_demo_spec())


def _build_llm_call_runner(*, use_llm: bool, model: str) -> NodeRunner:
    """Pick the runner for llm_call nodes: real model with --llm, mock otherwise."""

    if use_llm:
        return build_openai_llm_runner(model=model)
    return mock_llm_runner


def _build_reduce_runner(*, use_llm: bool, model: str) -> NodeRunner | None:
    """Override reduce only when --llm is set; otherwise default deterministic runner."""

    if use_llm:
        return build_openai_reduce_runner(model=model)
    return None


def _build_spawn_subagent_runner(*, use_llm: bool, model: str) -> NodeRunner | None:
    """Override spawn_subagent only when --llm is set; otherwise default echo."""

    if use_llm:
        return build_openai_spawn_subagent_runner(model=model)
    return None


def _build_tool_call_runner(*, use_llm: bool) -> NodeRunner | None:
    """Use concrete allowlisted tools for LLM-backed demos."""

    if use_llm:
        return build_grounded_tool_runner()
    return None


def _build_emit_artifact_runner(*, runs_dir: Path) -> NodeRunner:
    """Persist emitted artifacts to runs/<run_id>/artifacts/<name>.<ext>.

    Wired for both LLM and non-LLM demos — artifacts are useful in either
    case (they let you inspect what the workflow actually produced).
    """
    return make_emit_artifact_runner(FileArtifactSink(root_dir=runs_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic Subgraphs demo")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the planner. Default: a research-shaped prompt.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=os.environ.get("DEMO_USE_LLM") == "1",
        help="Use the LLMPlanner (ChatOpenAI). Default: StaticPlanner (free).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=f"OpenAI model name when --llm is set. Default: {DEFAULT_LLM_MODEL}.",
    )
    parser.add_argument(
        "--run-id",
        default="demo-run-001",
        help="Run id used by the recorder. Default: demo-run-001.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()

    runs_dir = Path(__file__).resolve().parents[1] / "runs"

    planner = _build_planner(use_llm=args.llm, model=args.model)
    llm_call_runner = _build_llm_call_runner(use_llm=args.llm, model=args.model)
    reduce_runner = _build_reduce_runner(use_llm=args.llm, model=args.model)
    subagent_runner = _build_spawn_subagent_runner(use_llm=args.llm, model=args.model)
    tool_runner = _build_tool_call_runner(use_llm=args.llm)
    emit_runner = _build_emit_artifact_runner(runs_dir=runs_dir)

    runners: dict[NodeKind, NodeRunner] = {
        NodeKind.LLM_CALL: llm_call_runner,
        NodeKind.EMIT_ARTIFACT: emit_runner,
    }
    if reduce_runner is not None:
        runners[NodeKind.REDUCE] = reduce_runner
    if subagent_runner is not None:
        runners[NodeKind.SPAWN_SUBAGENT] = subagent_runner
    if tool_runner is not None:
        runners[NodeKind.TOOL_CALL] = tool_runner

    supervisor = Supervisor(
        planner=planner,
        executor=LangGraphExecutor(runners=runners, strict_runners=args.llm),
        recorder=FileRecorder(root_dir=runs_dir, overwrite=True),
    )

    planner_label = f"LLMPlanner({args.model})" if args.llm else "StaticPlanner"
    runner_label = f"OpenAILlmRunner({args.model})" if args.llm else "mock_llm_runner"
    reduce_label = (
        f"LlmReduceRunner({args.model})" if args.llm else "run_reduce (default)"
    )
    subagent_label = (
        f"OpenAI subagents ({args.model})"
        if args.llm
        else "run_spawn_subagent (default echo)"
    )
    tool_label = (
        "grounded allowlisted tools" if args.llm else "run_tool_call (default echo)"
    )
    print(f"[demo] planner   = {planner_label}")
    print(f"[demo] runner    = {runner_label}")
    print(f"[demo] reduce    = {reduce_label}")
    print(f"[demo] subagents = {subagent_label}")
    print(f"[demo] tools     = {tool_label}")
    print(f"[demo] artifacts = FileArtifactSink({runs_dir})")
    print(f"[demo] prompt    = {args.prompt!r}")

    result = supervisor.run(args.prompt, run_id=args.run_id)
    render_supervisor_result(result)


if __name__ == "__main__":
    main()
