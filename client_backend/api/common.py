"""
Shared API helpers for the client backend.
"""

import contextlib
import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

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
    incoming = [normalized.pop(key) for key in (snake_key, camel_key) if key in normalized]

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


def rewrite_widget_ws_url(payload: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a relative widget ``ws_url`` to an absolute URL on the canonical server.

    The canonical server returns a root-relative path (``/widgets/{id}/connect?...``)
    which, left untouched, the browser would resolve against the sidecar — where no
    widget WebSocket route exists, yielding a 403. Pointing the URL at the canonical
    server lets the browser open the socket directly (no local WS relay in this phase).
    """
    ws_url = payload.get("ws_url")
    if not isinstance(ws_url, str) or not ws_url or ws_url.startswith(("ws://", "wss://")):
        return payload

    base = urlsplit(get_server_client().base_url)
    rel = urlsplit(ws_url)
    scheme = "wss" if base.scheme == "https" else "ws"
    path = f"{base.path.rstrip('/')}{rel.path}" if rel.path.startswith("/") else rel.path
    updated = dict(payload)
    updated["ws_url"] = urlunsplit((scheme, base.netloc, path, rel.query, rel.fragment))
    return updated


async def proxy_server_request(
    request: Request,
    *,
    upstream_path: str,
    params_override: Any = None,
    json_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Response:
    """
    Forward a request to the canonical server while preserving its response body.

    Chat/streaming routes that need device context call ``add_device_context``
    explicitly before forwarding; this generic proxy never injects it. When
    ``json_transform`` is provided, it rewrites a JSON object response body before
    it is returned to the caller.
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
                kwargs["json"] = json.loads(body.decode("utf-8"))
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
            payload = response.json()
            if json_transform is not None and isinstance(payload, dict):
                payload = json_transform(payload)
            return JSONResponse(
                status_code=response.status_code,
                content=payload,
            )
        except Exception:
            pass

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
    )


# ── Protected media (binary) proxy ─────────────────────────────────────────
# Streams a protected upstream read (e.g. ``/chat-images/{id}``) to the local
# caller without buffering the whole payload, forwarding cache validators and
# stamping hardening headers, while never leaking upstream internals.

# Hard ceiling on a single proxied media read. Enforced twice: up front against
# a DECLARED content-length (clean 413, body never drained), and again as a
# running byte count while streaming, so a chunked/undeclared upstream cannot
# relay an unbounded body through the sidecar.
MAX_MEDIA_PROXY_BYTES = 25 * 1024 * 1024  # 25 MiB


class MediaTooLargeError(RuntimeError):
    """Raised mid-stream when a proxied media body exceeds the hard ceiling.

    The response headers are already sent by then, so this cannot become a 413;
    aborting the body is the bounded failure mode, and the truncated read is
    visible to the caller as a broken response rather than an unbounded one.
    """

# Response headers safe to forward verbatim: cache validators, caching policy,
# and content framing. Deliberately excludes ``content-type`` (set explicitly
# as the media type) and any upstream server/identity headers.
_FORWARDED_MEDIA_RESPONSE_HEADERS = (
    "etag",
    "last-modified",
    "cache-control",
    "expires",
    "vary",
    "content-disposition",
    "content-length",
)

# Generic, non-leaking details keyed by the canonical status we preserve.
_MEDIA_ERROR_DETAIL = {
    status.HTTP_401_UNAUTHORIZED: "Authentication required",
    status.HTTP_404_NOT_FOUND: "Image not found",
    status.HTTP_413_CONTENT_TOO_LARGE: "Image too large",
}

# Defense-in-depth for a directly-served binary: never let a browser sniff the
# body into an active type, and fully sandbox it.
_MEDIA_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; img-src 'self' data:; sandbox",
}


def _raise_media_error(status_code: int) -> None:
    """Map an upstream media failure to a local ``HTTPException``.

    Preserves the canonical status (401/404/413/5xx) but replaces the body with
    a generic detail so upstream internals (storage paths, existence oracles)
    never reach the caller.
    """
    detail = _MEDIA_ERROR_DETAIL.get(status_code, "Upstream media request failed")
    raise HTTPException(status_code=status_code, detail=detail)


def _safe_media_headers(upstream_headers: Any) -> dict[str, str]:
    """Forward whitelisted cache/content headers and add hardening headers."""
    headers: dict[str, str] = {}
    for name in _FORWARDED_MEDIA_RESPONSE_HEADERS:
        value = upstream_headers.get(name)
        if value:
            headers[name] = value
    headers.update(_MEDIA_SECURITY_HEADERS)
    return headers


async def proxy_media_request(*, upstream_path: str) -> Response:
    """Stream a protected binary read from the canonical server to the caller.

    The caller must already have passed the local-session gate (the route
    dependency), so this never re-checks auth; it attaches the upstream
    credentials via the server client, streams the body chunk by chunk (never
    buffering the whole payload), forwards cache validators plus hardening
    headers, and maps upstream failures onto local status codes without
    exposing upstream internals.
    """
    stream_cm = get_server_client().stream_media(upstream_path, method="GET")
    try:
        response = await stream_cm.__aenter__()
    except Exception as exc:
        raise_server_error(exc)

    async def _close() -> None:
        with contextlib.suppress(Exception):
            await stream_cm.__aexit__(None, None, None)

    if response.status_code >= 400:
        await _close()
        _raise_media_error(response.status_code)

    content_length = response.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_MEDIA_PROXY_BYTES:
        await _close()
        _raise_media_error(status.HTTP_413_CONTENT_TOO_LARGE)

    async def _body() -> Any:
        relayed = 0
        try:
            async for chunk in response.aiter_bytes():
                relayed += len(chunk)
                if relayed > MAX_MEDIA_PROXY_BYTES:
                    # Undeclared/chunked upstream: the pre-check could not fire,
                    # so bound it here instead of relaying without limit.
                    logger.error(
                        "Aborting proxied media read for %s: body exceeded %d bytes",
                        upstream_path,
                        MAX_MEDIA_PROXY_BYTES,
                    )
                    raise MediaTooLargeError(upstream_path)
                yield chunk
        finally:
            await _close()

    return StreamingResponse(
        _body(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
        headers=_safe_media_headers(response.headers),
    )
