"""Engine wiring for the eval gate (Slice 7 PR4). Mock planner = token-free."""

import json

from dynamic_subgraphs import (
    DeterministicEvalGate,
    DynamicSubgraphs,
    EngineConfig,
    EvalReference,
    Recording,
)


def _engine(tmp_path, **cfg):
    return DynamicSubgraphs(EngineConfig(planner="mock", runs_dir=tmp_path, **cfg))


def test_eval_off_by_default(tmp_path) -> None:
    r = _engine(tmp_path).run("compare A and B", record=True)
    assert r.eval is None
    assert not (tmp_path / r.run_id / "eval.json").exists()


def test_eval_gate_populates_run_result(tmp_path) -> None:
    r = _engine(tmp_path, eval_gate=DeterministicEvalGate()).run(
        "compare A and B", record=True
    )
    assert r.eval is not None
    assert r.eval.run_id == r.run_id
    assert 0.0 <= r.eval.quality <= 1.0
    assert r.eval.deterministic is True
    assert r.to_dict()["eval"]["gate"] == "deterministic@v1"


def test_eval_json_written_when_selected(tmp_path) -> None:
    r = _engine(tmp_path, eval_gate=DeterministicEvalGate()).run(
        "compare A and B", record=Recording.evaluated()
    )
    payload = json.loads((tmp_path / r.run_id / "eval.json").read_text("utf-8"))
    assert payload["run_id"] == r.run_id


def test_no_eval_json_with_null_recorder(tmp_path) -> None:
    # recording=False -> NullRecorder: score returned in memory, nothing written.
    r = _engine(tmp_path, eval_gate=DeterministicEvalGate()).run("compare A and B")
    assert r.eval is not None
    assert not (tmp_path / r.run_id).exists()


def test_run_kwargs_thread_tags(tmp_path) -> None:
    ref = EvalReference(checklist=["alpha"], grounding_required=False)
    r = _engine(tmp_path, eval_gate=DeterministicEvalGate()).run(
        "say alpha", record=False, task_id="t1", origin="routed", reference=ref
    )
    assert r.eval is not None
    assert r.eval.fingerprint.origin == "routed"


def test_engine_eval_tags_default_and_per_call_override(tmp_path) -> None:
    from dynamic_subgraphs import EvalTags

    eng = _engine(
        tmp_path,
        eval_gate=DeterministicEvalGate(),
        eval_tags=EvalTags(task_id="default-task", origin="authored"),
    )
    r1 = eng.run("hello", record=False)
    assert r1.eval.fingerprint.origin == "authored"
    r2 = eng.run("hello", record=False, origin="invented")
    assert r2.eval.fingerprint.origin == "invented"


def test_gate_failure_never_fails_the_run(tmp_path) -> None:
    class Boom:
        name = "boom@v1"

        def evaluate(self, context):
            raise RuntimeError("gate exploded")

    r = _engine(tmp_path, eval_gate=Boom()).run("compare A and B", record=True)
    assert r.status == "ok"
    assert r.eval is None


def test_execution_failed_run_writes_quality_zero_eval(tmp_path) -> None:
    # A valid plan whose worker node raises -> execution_failed; the gate's
    # not-ok branch still persists a quality-0, verdict-fail eval.json.
    from app.models import GraphSpec, NodeKind, NodeSpec
    from app.models.graph_spec import EdgeSpec
    from app.runtime import ProviderRegistry
    from dynamic_subgraphs import Model

    class _BoomChat:
        def invoke(self, messages, /, **kwargs):
            raise RuntimeError("worker exploded")

    class _FixedPlanner:
        def invoke(self, messages, /, **kwargs):
            return GraphSpec(
                graph_id="eval-execfail",
                goal="fail on purpose",
                nodes=[
                    NodeSpec(
                        id="answer",
                        kind=NodeKind.LLM_CALL,
                        outputs=["final"],
                        params={"instruction": "answer"},
                    )
                ],
                edges=[
                    EdgeSpec.model_validate({"from": "START", "to": "answer"}),
                    EdgeSpec.model_validate({"from": "answer", "to": "END"}),
                ],
            )

    class _Provider:
        name = "boom"

        def required_env_vars(self):
            return ()

        def build_chat(self, ref):
            return _BoomChat()

        def build_structured_output(self, ref, schema):
            return _FixedPlanner()

    registry = ProviderRegistry()
    registry.register(_Provider())
    eng = DynamicSubgraphs(
        EngineConfig(
            model=Model("boom", "boom-model"),
            providers=registry,
            runs_dir=tmp_path,
            eval_gate=DeterministicEvalGate(),
        )
    )
    r = eng.run("compare A and B", record=Recording.evaluated())

    assert r.status == "execution_failed"
    assert r.eval is not None
    assert r.eval.quality == 0.0
    payload = json.loads((tmp_path / r.run_id / "eval.json").read_text("utf-8"))
    assert payload["quality"] == 0.0
    assert payload["overall_verdict"] == "fail"


def test_capabilities_lists_eval_gates() -> None:
    caps = DynamicSubgraphs.capabilities()
    assert "deterministic" in caps["eval_gates"]
    assert "evaluated" in caps["recording_presets"]
