# tests/test_api_schemas.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import MAX_PROMPT_CHARS, ChainRequest, RunRequest


def test_run_request_defaults_mode_auto() -> None:
    req = RunRequest(prompt="hi")
    assert req.mode == "auto"
    assert req.run_id is None
    assert req.planner is None
    assert req.provider is None
    assert req.model is None


def test_run_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        RunRequest(prompt="")


def test_run_request_rejects_bad_mode() -> None:
    with pytest.raises(ValidationError):
        RunRequest(prompt="hi", mode="turbo")


def test_run_and_chain_requests_bound_prompt_length() -> None:
    # The prompt drives a paid LLM call and is persisted — it must be bounded.
    over = "x" * (MAX_PROMPT_CHARS + 1)
    with pytest.raises(ValidationError):
        RunRequest(prompt=over)
    with pytest.raises(ValidationError):
        ChainRequest(prompt=over)
    # A prompt exactly at the ceiling is still accepted.
    assert RunRequest(prompt="x" * MAX_PROMPT_CHARS).prompt


def test_chain_request_defaults() -> None:
    req = ChainRequest(prompt="hi")
    assert req.max_iterations == 3
    assert req.mode == "auto"
    assert req.decider == "status"
    assert req.success_criteria is None
    assert req.judge_failed_runs is False
