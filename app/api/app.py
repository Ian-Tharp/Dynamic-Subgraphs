# app/api/app.py
"""create_app() — wire settings, context, routers, error handlers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.deps import AppContext
from app.api.errors import install_error_handlers
from app.api.routers import chains, health, registry, runs
from app.api.settings import ApiSettings


def _api_version() -> str:
    """The installed package version, or a dev fallback from a source checkout."""
    try:
        return version("dynamic-subgraphs")
    except PackageNotFoundError:  # running from source without installed metadata
        return "0+unknown"


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    load_dotenv()
    settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Nothing to start; tear down the background job executor on shutdown.
        yield
        app.state.context.jobs.shutdown()

    app = FastAPI(
        title="Dynamic Subgraphs API",
        version=_api_version(),
        lifespan=lifespan,
    )
    app.state.context = AppContext.build(settings)

    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(registry.router)
    app.include_router(runs.router)
    app.include_router(chains.router)

    return app
