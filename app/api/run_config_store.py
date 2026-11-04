"""Back-compat shim: the run-config sidecar now lives in `app.recording`.

It moved so non-HTTP adapters (the MCP server) can share it without importing
FastAPI. Import from `app.recording.run_config` in new code.
"""

from app.recording.run_config import load_run_config, save_run_config

__all__ = ["load_run_config", "save_run_config"]
