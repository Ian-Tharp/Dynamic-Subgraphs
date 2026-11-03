"""`python -m app.mcp` -> run the MCP server (stdio by default).

    python -m app.mcp                              # stdio (for local MCP hosts)
    python -m app.mcp --transport streamable-http  # HTTP on DS_MCP_HOST:DS_MCP_PORT

Logs go to stderr: on stdio, stdout carries the protocol and must stay clean.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.mcp", description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"])
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    try:
        from app.mcp.server import build_server
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise SystemExit(
            'The MCP server needs the `mcp` extra: pip install "dynamic-subgraphs[mcp]"'
        ) from exc
    from app.mcp.service import GovernedRunService
    from app.mcp.settings import McpSettings

    settings = McpSettings.from_env()
    overrides = {
        key: value
        for key, value in {
            "transport": args.transport,
            "host": args.host,
            "port": args.port,
        }.items()
        if value is not None
    }
    settings = replace(settings, **overrides)
    server = build_server(GovernedRunService(settings))
    if settings.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
