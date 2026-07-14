"""
Centralized exception handler registration for FastAPI API layer.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import (
    HTTPException as FastAPIHTTPException,
)
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import CustomHTTPException
from app.schemas.responses.api_response import ApiResponse


def register_exception_handlers(app: FastAPI):
    @app.exception_handler(CustomHTTPException)
    async def custom_http_exception_handler(request: Request, exc: CustomHTTPException):
        api_response = ApiResponse(
            success=False,
            code=exc.error_code,
            message=exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=api_response.model_dump(by_alias=True, exclude_none=True),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(FastAPIHTTPException)
    async def fastapi_http_exception_handler(request: Request, exc: FastAPIHTTPException):
        api_response = ApiResponse(
            success=False,
            code="http_error",
            message=exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=api_response.model_dump(by_alias=True, exclude_none=True),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            api_response = ApiResponse(
                success=False,
                code="not_found",
                message=exc.detail or "Resource not found.",
            )
            return JSONResponse(
                status_code=404,
                content=api_response.model_dump(by_alias=True, exclude_none=True),
            )
        api_response = ApiResponse(
            success=False,
            code="http_error",
            message=exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=api_response.model_dump(by_alias=True, exclude_none=True),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        error_details = {}
        for error in exc.errors():
            loc = ".".join(map(str, error["loc"]))
            if loc not in error_details:
                error_details[loc] = []
            error_details[loc].append(error["msg"])

        api_response = ApiResponse(
            success=False,
            code="invalid_input",
            message="Invalid input",
            error=error_details,
        )
        return JSONResponse(
            status_code=422,
            content=api_response.model_dump(by_alias=True, exclude_none=True),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        api_response = ApiResponse(
            success=False,
            code="internal_server_error",
            message="An internal server error occurred.",
        )
        return JSONResponse(
            status_code=500,
            content=api_response.model_dump(by_alias=True, exclude_none=True),
        )
