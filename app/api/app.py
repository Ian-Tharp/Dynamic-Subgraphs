# app/api/app.py
"""create_app() — wire settings, context, routers, error handlers."""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.deps import AppContext
from app.api.errors import install_error_handlers
from app.api.routers import chains, health, registry, runs
from app.api.settings import ApiSettings


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    load_dotenv()
    settings = settings or ApiSettings.from_env()

    app = FastAPI(title="Dynamic Subgraphs API", version="1.0.0")
    app.state.context = AppContext.build(settings)

    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(registry.router)
    app.include_router(runs.router)
    app.include_router(chains.router)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        app.state.context.jobs.shutdown()

    return app
