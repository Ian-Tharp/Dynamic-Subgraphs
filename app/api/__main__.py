# app/api/__main__.py
"""`python -m app.api` -> run uvicorn."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("DS_HOST", "127.0.0.1")
    port = int(os.environ.get("DS_PORT", "8000"))
    uvicorn.run("app.api:create_app", host=host, port=port, factory=True)


if __name__ == "__main__":
    main()
