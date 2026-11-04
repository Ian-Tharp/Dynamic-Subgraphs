"""MCP server: expose governed subgraph runs to MCP hosts (Claude, IDEs, agents).

`GovernedRunService` (no MCP dependency) owns the governance rules;
`build_server` (needs the `mcp` extra) adapts it to MCP tools and resources.
Run it with `python -m app.mcp`. See `docs/mcp.md`.
"""

from app.mcp.service import GovernanceError, GovernedRunService, NotFoundError
from app.mcp.settings import McpSettings

__all__ = ["GovernanceError", "GovernedRunService", "McpSettings", "NotFoundError"]
