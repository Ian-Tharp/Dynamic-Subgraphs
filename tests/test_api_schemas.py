# tests/test_api_schemas.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import ChainRequest, RunRequest


def test_run_request_defaults_mode_auto() -> None:
    req = RunRequest(prompt="hi")
    assert req.mode == "auto"
    assert req.run_id is None
    assert req.planner is None
    assert req.model is None


def test_run_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        RunRequest(prompt="")


def test_run_request_rejects_bad_mode() -> None:
    with pytest.raises(ValidationError):
        RunRequest(prompt="hi", mode="turbo")


def test_chain_request_defaults() -> None:
    req = ChainRequest(prompt="hi")
    assert req.max_iterations == 3
    assert req.mode == "auto"
    assert req.decider == "status"
    assert req.success_criteria is None
    assert req.judge_failed_runs is False
