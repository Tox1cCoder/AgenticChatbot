"""
Shared API helpers for the client backend.
"""

import json
from typing import Any

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from client_backend.core.logging import get_logger
from client_backend.services.runtime_bridge import get_runtime_bridge
from client_backend.services.server_api import (
    AuthenticationError,
    ServerAPIError,
    ServerConnectionError,
    get_server_client,
)

logger = get_logger(__name__)


def raise_server_error(exc: Exception) -> None:
    """Map upstream client errors to local HTTP responses."""
    if isinstance(exc, AuthenticationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    if isinstance(exc, ServerConnectionError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if isinstance(exc, ServerAPIError):
        raise HTTPException(
            status_code=exc.status_code or status.HTTP_502_BAD_GATEWAY,
            detail=exc.detail or str(exc),
        ) from exc

    raise exc


def add_device_context(
    payload: dict[str, Any],
    *,
    snake_key: str = "device_id",
    camel_key: str = "deviceId",
) -> dict[str, Any]:
    """
    Stamp the payload with this installation's registered device id.

    The proxy is the only component that knows where a request physically
    originated, so it always asserts its own identity: any incoming device id
    (stale UI state, or a value replayed from another machine) is overwritten.
    When the local bridge is not connected, both keys are stripped so the
    server binds no client tools rather than trusting a forwarded id.
    """
    normalized = dict(payload)
    incoming = [
        normalized.pop(key) for key in (snake_key, camel_key) if key in normalized
    ]

    device_id = get_runtime_bridge().get_registered_device_id()
    foreign = [value for value in incoming if value and value != device_id]
    if foreign:
        logger.warning(
            "Overriding incoming device context %s with local device id %s; "
            "a payload carrying another device's id is the cross-client "
            "dispatch bug signature",
            foreign,
            device_id or "<bridge not connected>",
        )

    if device_id:
        normalized[snake_key] = device_id
    return normalized


def make_api_response(
    *,
    success: bool,
    message: str,
    data: Any = None,
    error: dict[str, Any] | None = None,
    status_code: int = status.HTTP_200_OK,
) -> JSONResponse:
    """Build a server-style ApiResponse envelope."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": success,
            "message": message,
            "data": data,
            "error": error,
        },
    )


async def proxy_server_request(
    request: Request,
    *,
    upstream_path: str,
    inject_device_context: bool = False,
    params_override: Any = None,
) -> Response:
    """
    Forward a request to the canonical server while preserving its response body.
    """
    try:
        kwargs: dict[str, Any] = {
            "params": (
                params_override
                if params_override is not None
                else list(request.query_params.multi_items())
            ),
        }

        content_type = str(request.headers.get("content-type") or "")
        body = await request.body()
        headers: dict[str, str] = {}

        if body:
            if "application/json" in content_type:
                payload = json.loads(body.decode("utf-8"))
                if inject_device_context and isinstance(payload, dict):
                    payload = add_device_context(payload)
                kwargs["json"] = payload
            else:
                kwargs["content"] = body
                if content_type:
                    headers["Content-Type"] = content_type

        if headers:
            kwargs["headers"] = headers

        response = await get_server_client().request_response(
            request.method,
            upstream_path,
            **kwargs,
        )
    except Exception as exc:
        raise_server_error(exc)

    if response.headers.get("content-type", "").lower().startswith("application/json"):
        try:
            return JSONResponse(
                status_code=response.status_code,
                content=response.json(),
            )
        except Exception:
            pass

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
    )
