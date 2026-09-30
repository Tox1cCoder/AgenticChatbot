"""Transport checks that decide whether a request may reach the sidecar at all.

The sidecar runs local tools as the signed-in user, so being reachable on
loopback must not be enough. Three checks run before any route:

* **Host** must be loopback or the configured ``backend_host``. A DNS-rebinding
  page reaches 127.0.0.1 under its own hostname, so its requests fail here (400).
* **Origin**, when a browser sends one on POST/PUT/PATCH/DELETE, must be one of
  ``allowed_origins``. CORS stops a foreign page reading a response, not sending
  the request (403).
* **Launch token.** A request must carry ``X-Kani-Client`` with this launch's
  token (see ``launch_token``), or come from an allowed browser Origin. A browser
  cannot read the token file, and the AI SDK frontend and the widget iframe call
  the sidecar from the browser; those still need a bearer session, which the
  routes check. The routes that hand out a session without any credential --
  ``/auth/restore`` and ``/auth/users`` -- accept the launch token only (401).
  With no token issued (startup failed to write it) every guarded route answers
  503 instead of opening up.

An Origin header can be forged by a non-browser client, so it is never a
credential by itself: it only admits a request to routes that authenticate it.

``trust_checks_enabled=false`` turns all three off. The test suite does that;
startup logs a warning whenever it is off.
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from client_backend.core.config import client_settings
from client_backend.core.launch_token import current_launch_token, launch_token_matches

__all__ = ["HostAllowlistMiddleware", "RequestTrustMiddleware"]


def _with_api_prefix(*paths: str) -> frozenset[str]:
    # Every compatibility router is mounted at both the bare path and /api.
    return frozenset(paths) | frozenset(f"/api{path}" for path in paths)


OPEN_PATHS = _with_api_prefix("/health", "/health/ready", "/health/live") | {
    "/docs",
    "/openapi.json",
}
LAUNCH_TOKEN_ONLY_PATHS = _with_api_prefix("/auth/restore", "/auth/users")
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _host_name(value: str) -> str:
    """The host part of a Host header or bind address, without port or brackets."""

    raw = value.strip().lower()
    if raw.startswith("["):
        end = raw.find("]")
        return raw[1:end] if end > 0 else ""
    if raw.count(":") == 1:
        return raw.split(":", 1)[0]
    return raw


def _allowed_hosts() -> frozenset[str]:
    return _LOOPBACK_HOSTS | {_host_name(client_settings.backend_host)}


async def _refuse(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
    await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)


class HostAllowlistMiddleware:
    """Refuse requests whose Host is not this machine (DNS rebinding)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not client_settings.trust_checks_enabled:
            await self.app(scope, receive, send)
            return
        host = _host_name(Headers(scope=scope).get("host", ""))
        if host not in _allowed_hosts():
            await _refuse(scope, receive, send, 400, "Host is not allowed")
            return
        await self.app(scope, receive, send)


class RequestTrustMiddleware:
    """Enforce the Origin check on mutations and the launch-token requirement."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not client_settings.trust_checks_enabled:
            await self.app(scope, receive, send)
            return
        refusal = _refusal(scope)
        if refusal is not None:
            await _refuse(scope, receive, send, *refusal)
            return
        await self.app(scope, receive, send)


def _refusal(scope: Scope) -> tuple[int, str] | None:
    headers = Headers(scope=scope)
    origin = headers.get("origin")
    origin_allowed = origin is not None and origin in client_settings.allowed_origins
    if scope["method"] in _MUTATING_METHODS and origin is not None and not origin_allowed:
        return 403, "Origin is not allowed"

    path = scope["path"].rstrip("/") or "/"
    if path in OPEN_PATHS:
        return None
    if current_launch_token() is None:
        return 503, "The sidecar has no launch token; restart it"

    presented = headers.get("x-kani-client")
    if presented is not None:
        return None if launch_token_matches(presented) else (401, "Invalid launch token")
    if path in LAUNCH_TOKEN_ONLY_PATHS:
        return 401, "This route requires the sidecar launch token"
    if origin_allowed:
        return None
    return 401, "Missing launch token"
