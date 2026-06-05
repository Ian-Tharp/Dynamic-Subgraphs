# app/api/errors.py
"""API exceptions and JSON error-envelope handlers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    status_code = 500
    error_type = "ApiError"

    def __init__(self, message: str, *, detail: object = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class NotFound(ApiError):
    status_code = 404
    error_type = "NotFound"


class Conflict(ApiError):
    status_code = 409
    error_type = "Conflict"


class Unauthorized(ApiError):
    status_code = 401
    error_type = "Unauthorized"


class BadRequest(ApiError):
    status_code = 400
    error_type = "BadRequest"


class ServiceUnavailable(ApiError):
    status_code = 503
    error_type = "ServiceUnavailable"


def _envelope(error_type: str, message: str, detail: object = None) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": message, "detail": detail}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.error_type, exc.message, exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "ValidationError", "Request validation failed", exc.errors()
            ),
        )
