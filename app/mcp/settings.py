"""Environment-driven settings for the MCP server.

Model selection reuses the HTTP API's knobs (`DS_PLANNER`, `DS_PROVIDER`,
`DS_MODEL`, `DS_MODEL_ALLOWLIST`, `DS_RUNS_DIR`) so one environment configures
both surfaces identically. The host-owned `ExecutionPolicy` is set with
`DS_MCP_*` knobs.

Receipts: every persisted receipt is anchored in an append-only, hash-chained
ledger (`DS_RECEIPT_LEDGER`, default `<runs_dir>/.ledger/receipts.jsonl`; `off`
disables it). Ledger entries are signed when a key is configured:
`DS_RECEIPT_HMAC_KEY` (shared secret, >= 32 bytes) or
`DS_RECEIPT_ED25519_KEY_FILE` (PEM private key; needs `cryptography`), with
`DS_RECEIPT_KEY_ID` naming the key.

Unlike the API's lenient parsing, a malformed value here raises at startup: an
MCP server is usually launched by an agent host with no one watching its logs,
so silently falling back to a default would quietly widen what an agent may do.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.governance import (
    LEDGER_RELATIVE_PATH,
    Ed25519Signer,
    HmacSigner,
    ReceiptSigner,
)
from app.models import NodeKind
from app.policy import ExecutionPolicy

PlannerMode = Literal["mock", "llm"]
Transport = Literal["stdio", "streamable-http"]

_POLICY_INT_KNOBS: dict[str, str] = {
    "DS_MCP_MAX_NODES": "max_nodes",
    "DS_MCP_MAX_LLM_CALLS": "max_llm_calls",
    "DS_MCP_MAX_DEPTH": "max_depth",
    "DS_MCP_MAX_FANOUT": "max_fanout",
    "DS_MCP_MAX_WALL_SECONDS": "max_wall_seconds",
}


def _parse_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _parse_allow_set(env: Mapping[str, str], name: str) -> frozenset[str] | None:
    """Parse a comma-separated allow-set, keeping the policy sentinel contract.

    Unset or ``*`` -> None (no host narrowing; the registry set applies).
    Set but empty -> frozenset() (ban all of that category — fail closed).
    """

    raw = env.get(name)
    if raw is None or raw.strip() == "*":
        return None
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


def _parse_node_kinds(env: Mapping[str, str]) -> frozenset[NodeKind] | None:
    names = _parse_allow_set(env, "DS_MCP_ALLOWED_NODE_KINDS")
    if names is None:
        return None
    valid = {kind.value for kind in NodeKind}
    unknown = sorted(names - valid)
    if unknown:
        raise ValueError(
            f"DS_MCP_ALLOWED_NODE_KINDS has unknown kinds {unknown}; "
            f"valid kinds are {sorted(valid)}"
        )
    return frozenset(NodeKind(name) for name in names)


def policy_from_env(env: Mapping[str, str]) -> ExecutionPolicy:
    """Build the host `ExecutionPolicy` from `DS_MCP_*` knobs (defaults otherwise)."""

    defaults = ExecutionPolicy()
    # Any: ExecutionPolicy now also has non-int fields (cost_model, catalog...),
    # so mypy can't match a **dict[str, int] against every keyword.
    ints: dict[str, Any] = {
        attr: _parse_int(env, knob, getattr(defaults, attr))
        for knob, attr in _POLICY_INT_KNOBS.items()
    }
    return ExecutionPolicy(
        **ints,
        allowed_tools=_parse_allow_set(env, "DS_MCP_ALLOWED_TOOLS"),
        allowed_subagents=_parse_allow_set(env, "DS_MCP_ALLOWED_SUBAGENTS"),
        allowed_node_kinds=_parse_node_kinds(env),
    )


_LEDGER_OFF = frozenset({"off", "none", "false", "0", "disabled"})


def ledger_path_from_env(env: Mapping[str, str], runs_dir: str) -> str | None:
    """`DS_RECEIPT_LEDGER`: unset -> under the runs dir; `off` -> no ledger."""

    raw = env.get("DS_RECEIPT_LEDGER")
    if raw is None or not raw.strip():
        return str(Path(runs_dir) / LEDGER_RELATIVE_PATH)
    if raw.strip().lower() in _LEDGER_OFF:
        return None
    return raw.strip()


def signer_from_env(env: Mapping[str, str]) -> ReceiptSigner | None:
    """Build the ledger signer from `DS_RECEIPT_*`; any bad key fails startup."""

    hmac_key = env.get("DS_RECEIPT_HMAC_KEY")
    ed_file = env.get("DS_RECEIPT_ED25519_KEY_FILE")
    key_id = (env.get("DS_RECEIPT_KEY_ID") or "default").strip() or "default"
    if hmac_key and ed_file:
        raise ValueError(
            "set only one of DS_RECEIPT_HMAC_KEY and DS_RECEIPT_ED25519_KEY_FILE"
        )
    if hmac_key:
        try:
            return HmacSigner(hmac_key, key_id=key_id)
        except ValueError as exc:
            raise ValueError(f"DS_RECEIPT_HMAC_KEY: {exc}") from exc
    if ed_file:
        try:
            return Ed25519Signer.from_pem_file(ed_file, key_id=key_id)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(
                f"DS_RECEIPT_ED25519_KEY_FILE {ed_file!r} is not a usable "
                f"Ed25519 private key: {exc}"
            ) from exc
    return None


@dataclass(frozen=True)
class McpSettings:
    planner: PlannerMode = "mock"
    provider: str = "openai"
    model: str = "gpt-5.4-nano"
    model_allowlist: tuple[str, ...] = ("gpt-5.4-nano",)
    runs_dir: str = "runs"
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    max_plan_attempts: int = 2
    # Per-string cap on values echoed back in tool results. Full values always
    # stay on disk (`output.json`) and are readable with `get_run_file`.
    max_value_chars: int = 4000
    transport: Transport = "stdio"
    host: str = "127.0.0.1"
    port: int = 8765
    # Receipt ledger file; None disables anchoring. "" (the default) means
    # "<runs_dir>/.ledger/receipts.jsonl", resolved by the service.
    receipt_ledger: str | None = ""
    # Signs ledger entries (None = unsigned). Never compared or printed.
    receipt_signer: ReceiptSigner | None = field(
        default=None, compare=False, repr=False
    )

    def ledger_path(self) -> Path | None:
        if self.receipt_ledger is None:
            return None
        if self.receipt_ledger == "":
            return Path(self.runs_dir) / LEDGER_RELATIVE_PATH
        return Path(self.receipt_ledger)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> McpSettings:
        env = os.environ if env is None else env

        planner = env.get("DS_PLANNER", "mock").strip().lower() or "mock"
        provider = env.get("DS_PROVIDER", "openai").strip().lower() or "openai"
        if planner == "openai":  # legacy alias, same as the API
            planner, provider = "llm", "openai"
        if planner not in ("mock", "llm"):
            raise ValueError(f"DS_PLANNER must be 'mock' or 'llm', got {planner!r}")

        model = env.get("DS_MODEL", "gpt-5.4-nano").strip() or "gpt-5.4-nano"
        allowlist_raw = env.get("DS_MODEL_ALLOWLIST", model)
        allowlist = tuple(i.strip() for i in allowlist_raw.split(",") if i.strip())

        transport = env.get("DS_MCP_TRANSPORT", "stdio").strip().lower() or "stdio"
        if transport not in ("stdio", "streamable-http"):
            raise ValueError(
                "DS_MCP_TRANSPORT must be 'stdio' or 'streamable-http', "
                f"got {transport!r}"
            )

        max_plan_attempts = _parse_int(env, "DS_MCP_MAX_PLAN_ATTEMPTS", 2)
        max_value_chars = _parse_int(env, "DS_MCP_MAX_VALUE_CHARS", 4000)
        if max_plan_attempts < 1:
            raise ValueError("DS_MCP_MAX_PLAN_ATTEMPTS must be >= 1")
        if max_value_chars < 1:
            raise ValueError("DS_MCP_MAX_VALUE_CHARS must be >= 1")

        runs_dir = env.get("DS_RUNS_DIR", "runs")
        return cls(
            planner=planner,  # type: ignore[arg-type]
            provider=provider,
            model=model,
            model_allowlist=allowlist or (model,),
            runs_dir=runs_dir,
            policy=policy_from_env(env),
            max_plan_attempts=max_plan_attempts,
            max_value_chars=max_value_chars,
            transport=transport,  # type: ignore[arg-type]
            host=env.get("DS_MCP_HOST", "127.0.0.1"),
            port=_parse_int(env, "DS_MCP_PORT", 8765),
            receipt_ledger=ledger_path_from_env(env, runs_dir),
            receipt_signer=signer_from_env(env),
        )

    def is_model_allowed(self, model: str, *, provider: str | None = None) -> bool:
        provider = (provider or self.provider).strip().lower()
        allowed = set(self.model_allowlist)
        return "*" in allowed or model in allowed or f"{provider}:{model}" in allowed
