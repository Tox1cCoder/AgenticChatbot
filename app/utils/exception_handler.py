"""
Centralized exception handler registration for FastAPI API layer.
Usage: from app.utils.exception_handler import register_exception_handlers
       register_exception_handlers(app)
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from app.core.exceptions import (
    CustomHTTPException,
)


def register_exception_handlers(app: FastAPI):
    @app.exception_handler(CustomHTTPException)
    async def custom_http_exception_handler(request: Request, exc: CustomHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code or "CUSTOM_ERROR",
                "detail": exc.detail,
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(FastAPIHTTPException)
    async def fastapi_http_exception_handler(
        request: Request, exc: FastAPIHTTPException
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "detail": str(exc),
            },
        )