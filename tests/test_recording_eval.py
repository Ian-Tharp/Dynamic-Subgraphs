"""Recording-vocabulary tests for Artifact.EVAL (Slice 7 PR3)."""

from dynamic_subgraphs.recording import RECORDER_ARTIFACTS, Artifact, Recording


def test_eval_artifact_exists_with_filename_value() -> None:
    assert Artifact.EVAL == "eval.json"


def test_eval_is_a_recorder_artifact() -> None:
    assert Artifact.EVAL in RECORDER_ARTIFACTS


def test_record_true_includes_eval() -> None:
    # Spec C2: Recording.all() (== record=True) includes EVAL — accepted + tested.
    assert Artifact.EVAL in Recording.all()


def test_evaluated_preset() -> None:
    expected = frozenset(
        {Artifact.SPEC, Artifact.TRACE, Artifact.OUTPUT, Artifact.EVAL}
    )
    assert Recording.evaluated().kinds == expected
    assert "evaluated" in Recording.preset_names()
