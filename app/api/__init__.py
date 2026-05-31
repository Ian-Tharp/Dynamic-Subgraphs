# app/api/__init__.py
"""HTTP API (FastAPI) — thin layer over the supervisor."""

from app.api.app import create_app

__all__ = ["create_app"]
