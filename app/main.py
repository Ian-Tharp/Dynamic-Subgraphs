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
from pathlib import Path

from dotenv import load_dotenv

# LLM responses regularly contain Unicode (arrows, em dashes, curly quotes).
# Windows' default console encoding is cp1252, so printing them crashes. Force
# UTF-8 (replace on encode errors) so the demo never dies for display reasons.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

from app.assembly import RunConfig, build_supervisor
from app.recording import FileRecorder
from app.supervisor import SupervisorResult

DEFAULT_LLM_MODEL = "gpt-5.4-nano"
DEFAULT_PROMPT = "Compare two evidence sources and recommend one."


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

    config = RunConfig(
        planner="llm" if args.llm else "mock",
        provider="openai",
        model=args.model,
        strict_runners=args.llm,
    )
    recorder = FileRecorder(root_dir=runs_dir, overwrite=True)
    supervisor = build_supervisor(config, recorder=recorder)

    planner_label = f"LLMPlanner({args.model})" if args.llm else "StaticPlanner"
    runner_label = (
        f"ChatLlmRunner(openai:{args.model})" if args.llm else "mock_llm_runner"
    )
    print(f"[demo] planner = {planner_label}")
    print(f"[demo] runner  = {runner_label}")
    print(f"[demo] runs    = {runs_dir}")
    print(f"[demo] prompt  = {args.prompt!r}")

    result = supervisor.run(args.prompt, run_id=args.run_id)
    render_supervisor_result(result)


if __name__ == "__main__":
    main()
