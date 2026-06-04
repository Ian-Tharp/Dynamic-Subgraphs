# app/api/settings.py
"""Environment-driven API settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

PlannerMode = Literal["mock", "llm"]


@dataclass(frozen=True)
class ApiSettings:
    planner: PlannerMode = "mock"
    provider: str = "openai"
    model: str = "gpt-5.4-nano"
    model_allowlist: tuple[str, ...] = ("gpt-5.4-nano",)
    runs_dir: str = "runs"
    api_key: str | None = None
    auto_sync_seconds: int = 25
    max_sync_seconds: int = 120

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ApiSettings:
        env = os.environ if env is None else env
        planner = env.get("DS_PLANNER", "mock")
        provider = env.get("DS_PROVIDER", "openai").strip().lower() or "openai"
        if planner == "openai":
            planner = "llm"
            provider = "openai"
        if planner not in ("mock", "llm"):
            planner = "mock"
        allowlist_raw = env.get("DS_MODEL_ALLOWLIST", "gpt-5.4-nano")
        allowlist = tuple(
            item.strip() for item in allowlist_raw.split(",") if item.strip()
        )
        return cls(
            planner=planner,  # type: ignore[arg-type]
            provider=provider,
            model=env.get("DS_MODEL", "gpt-5.4-nano"),
            model_allowlist=allowlist or ("gpt-5.4-nano",),
            runs_dir=env.get("DS_RUNS_DIR", "runs"),
            api_key=env.get("DS_API_KEY") or None,
            auto_sync_seconds=int(env.get("DS_AUTO_SYNC_SECONDS", "25")),
            max_sync_seconds=int(env.get("DS_MAX_SYNC_SECONDS", "120")),
        )

    def is_model_allowed(self, model: str, *, provider: str | None = None) -> bool:
        provider = (provider or self.provider).strip().lower()
        allowed = set(self.model_allowlist)
        return "*" in allowed or model in allowed or f"{provider}:{model}" in allowed
