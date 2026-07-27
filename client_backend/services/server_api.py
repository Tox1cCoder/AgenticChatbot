"""
Server API client for communicating with the canonical backend.

This module provides an async HTTP client wrapper for all server API calls.
"""

import contextlib
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, TypeVar
from urllib.parse import urlencode, urlparse, urlunparse
from uuid import UUID

import httpx
from pydantic import BaseModel

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.schemas.runtime import CatalogSyncResult, DeviceRegistrationResult

logger = get_logger(__name__)
TModel = TypeVar("TModel", bound=BaseModel)


class ServerAPIError(Exception):
    """Base exception for server API errors."""

    def __init__(self, message: str, status_code: int | None = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class AuthenticationError(ServerAPIError):
    """Raised when authentication fails."""

    pass


class ServerConnectionError(ServerAPIError):
    """Raised when server is unreachable."""

    pass


class TokenPair(BaseModel):
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str | None = None
    expires_in: int | None = None
    expires_at: datetime | None = None


class OperationResult(BaseModel):
    """Normalized status payload for side-effect-oriented upstream operations."""

    message: str


class UploadProxyResponse(BaseModel):
    """Status-aware payload returned by the batch upload proxy."""

    status_code: int
    payload: dict[str, Any]


class ServerAPIClient:
    """
    Async HTTP client for the canonical server backend.

    Handles authentication, token refresh, and request proxying.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or client_settings.server_api_base_url).rstrip("/")
        self.timeout = timeout or client_settings.server_api_timeout_seconds
        self._tokens: TokenPair | None = None
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": f"CodexClientBackend/{client_settings.device_name}"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authorization headers if tokens are available."""
        if self._tokens:
            return {"Authorization": f"Bearer {self._tokens.access_token}"}
        return {}

    @staticmethod
    def _unwrap_api_data(payload: dict[str, Any], *, context: str) -> dict[str, Any]:
        """Extract the `data` object from the server's ApiResponse envelope."""
        data = payload.get("data")
        if payload.get("success") is False:
            raise ServerAPIError(
                f"Server reported failure during {context}",
                detail=payload,
            )
        if not isinstance(data, dict):
            raise ServerAPIError(
                f"Server response for {context} did not contain an object payload",
                detail=payload,
            )
        return data

    @classmethod
    def _parse_typed_payload(
        cls,
        payload: dict[str, Any],
        *,
        model: type[TModel],
        context: str,
    ) -> TModel:
        """Validate a direct payload or ApiResponse envelope into one typed runtime model."""
        normalized_payload = payload
        if any(key in payload for key in ("success", "data", "error")):
            normalized_payload = cls._unwrap_api_data(payload, context=context)

        if not isinstance(normalized_payload, dict):
            raise ServerAPIError(
                f"Server response for {context} did not contain an object payload",
                detail=payload,
            )

        try:
            return model.model_validate(normalized_payload)
        except Exception as exc:
            raise ServerAPIError(
                f"Server response for {context} did not match {model.__name__}",
                detail=normalized_payload,
            ) from exc

    async def request_response(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an authenticated request and return the raw HTTP response."""
        client = await self._get_client()
        headers = dict(kwargs.pop("headers", {}))
        include_auth_headers = bool(kwargs.pop("include_auth_headers", True))
        if include_auth_headers:
            auth_headers = self._get_auth_headers()
            auth_headers.update(headers)
            headers = auth_headers

        try:
            return await client.request(
                method,
                path,
                headers=headers,
                **kwargs,
            )
        except httpx.ConnectError as e:
            raise ServerConnectionError(f"Cannot connect to server at {self.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise ServerConnectionError(f"Server request timed out: {e}") from e

    async def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle response and raise appropriate errors."""
        body = await response.aread()

        def _parse_detail() -> Any:
            if not body:
                return None
            with contextlib.suppress(Exception):
                return response.json()
            return body.decode("utf-8", errors="replace")

        if response.status_code == 401:
            raise AuthenticationError(
                "Authentication required",
                status_code=401,
                detail=_parse_detail(),
            )

        if response.status_code == 403:
            raise AuthenticationError(
                "Access forbidden",
                status_code=403,
                detail=_parse_detail(),
            )

        if response.status_code >= 400:
            raise ServerAPIError(
                f"Server error: {response.status_code}",
                status_code=response.status_code,
                detail=_parse_detail(),
            )

        if not body:
            return {}

        try:
            return response.json()
        except Exception as exc:
            raise ServerAPIError(
                f"Server returned a non-JSON response: {response.status_code}",
                status_code=response.status_code,
                detail=body.decode("utf-8", errors="replace"),
            ) from exc

    async def request(
        self,
        method: str,
        path: str,
        *,
        _retry_on_auth: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make an authenticated request to the server.

        On a 401 with a refresh token available, transparently refresh the access
        token once and replay the request, so a normally-expired access token does
        not surface as an error to the caller. The ``/auth/*`` endpoints are excluded
        (they carry their own credentials and must not recurse through refresh).

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: API path (e.g., "/api/conversations")
            **kwargs: Additional arguments passed to httpx

        Returns:
            Parsed JSON response.

        Raises:
            ServerConnectionError: If the server is unreachable.
            AuthenticationError: If authentication fails (and refresh could not recover it).
            ServerAPIError: For other server errors.
        """
        response = await self.request_response(method, path, **kwargs)
        if (
            response.status_code == 401
            and _retry_on_auth
            and self._tokens is not None
            and self._tokens.refresh_token
            and not path.startswith("/auth/")
        ):
            try:
                await self.refresh_token()
            except (AuthenticationError, ServerAPIError):
                # Refresh genuinely failed — surface the original 401.
                return await self._handle_response(response)
            # Replay once with the refreshed token; never retry a second time.
            return await self.request(method, path, _retry_on_auth=False, **kwargs)
        return await self._handle_response(response)

    async def get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Make a GET request."""
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Make a POST request."""
        return await self.request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Make a PUT request."""
        return await self.request("PUT", path, **kwargs)

    async def patch(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Make a PATCH request."""
        return await self.request("PATCH", path, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Make a DELETE request."""
        return await self.request("DELETE", path, **kwargs)

    async def stream_sse(
        self,
        path: str,
        method: str = "POST",
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream Server-Sent Events from the server.

        Uses an unbounded read timeout because the server's AI SDK streaming
        endpoint does not emit heartbeats during tool execution.  The
        connection stays alive until the server sends ``[DONE]`` or closes.

        Args:
            path: API path for the SSE endpoint.
            method: HTTP method (usually POST).
            **kwargs: Additional arguments passed to httpx.

        Yields:
            Parsed SSE event data.
        """
        client = await self._get_client()
        headers = dict(kwargs.pop("headers", {}))
        auth_headers = self._get_auth_headers()
        auth_headers.update(headers)
        headers = auth_headers
        headers["Accept"] = "text/event-stream"

        # Streaming needs an unbounded read timeout: the server may pause for
        # tens of seconds during tool execution without sending any bytes.
        stream_timeout = httpx.Timeout(
            connect=min(self.timeout, 30),
            read=None,
            write=min(self.timeout, 30),
            pool=min(self.timeout, 30),
        )

        try:
            async with client.stream(
                method,
                path,
                headers=headers,
                timeout=stream_timeout,
                **kwargs,
            ) as response:
                if response.status_code >= 400:
                    await self._handle_response(response)

                proxied = 0
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        # Break on the FIRST upstream [DONE] and never yield it:
                        # the sidecar appends its own single trailing [DONE], so
                        # exactly one survives the proxy boundary even when the
                        # upstream emits [DONE] followed by trailing bytes.
                        if data.strip() == "[DONE]":
                            break
                        try:
                            import json

                            parsed = json.loads(data)
                        except Exception:
                            proxied += 1
                            yield {"raw": data}
                            continue

                        # Filter server heartbeat events - the sidecar's own
                        # SSE proxy injects keepalive comments independently.
                        if isinstance(parsed, dict) and parsed.get("type") == "heartbeat":
                            continue

                        proxied += 1
                        yield parsed
                logger.debug("SSE proxied %d upstream events from %s", proxied, path)

        except httpx.ConnectError as e:
            raise ServerConnectionError(f"Cannot connect to server at {self.base_url}: {e}") from e
        except httpx.ReadTimeout:
            logger.warning(
                "SSE stream read timeout on %s (this should not happen with read=None)", path
            )

    @contextlib.asynccontextmanager
    async def stream_media(
        self,
        path: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[httpx.Response]:
        """Open an authenticated streaming response to an upstream binary route.

        Yields the live ``httpx.Response`` so the caller can forward the status,
        headers, and body chunks without buffering the whole payload into memory.
        The response context stays open for the duration of the ``async with``
        block (and any body iteration inside it) and is torn down on exit.
        """
        client = await self._get_client()
        request_headers = self._get_auth_headers()
        if headers:
            request_headers.update(headers)

        # Bounded connect/write, unbounded read: a large image body may take a
        # while to arrive, but the sidecar must never hang forever on a dead peer.
        stream_timeout = httpx.Timeout(
            connect=min(self.timeout, 30),
            read=None,
            write=min(self.timeout, 30),
            pool=min(self.timeout, 30),
        )

        try:
            async with client.stream(
                method,
                path,
                headers=request_headers,
                timeout=stream_timeout,
            ) as response:
                yield response
        except httpx.ConnectError as e:
            raise ServerConnectionError(f"Cannot connect to server at {self.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise ServerConnectionError(f"Server request timed out: {e}") from e

    # ── Authentication Methods ──────────────────────────────────────────

    async def login(self, email: str, password: str) -> TokenPair:
        """
        Authenticate with the server.

        Args:
            email: User email.
            password: User's password.

        Returns:
            The normalized active token pair.
        """
        response = await self.post(
            "/auth/login",
            json={"email": email, "password": password},
            include_auth_headers=False,
        )
        data = self._unwrap_api_data(response, context="login")

        self._tokens = TokenPair(
            access_token=str(data["accessToken"]),
            refresh_token=str(data["refreshToken"]),
            token_type=str(data.get("tokenType", "bearer")),
            user_id=str(data.get("userId")) if data.get("userId") else None,
            expires_in=int(data.get("expiresIn")) if data.get("expiresIn") is not None else None,
        )

        logger.info("Logged in as %s", email)
        return self._tokens

    async def refresh_token(self) -> TokenPair:
        """
        Refresh the access token using the refresh token.

        Returns:
            The normalized active token pair.

        Raises:
            AuthenticationError: If refresh fails.
        """
        if not self._tokens or not self._tokens.refresh_token:
            raise AuthenticationError("No refresh token available")

        response = await self.post(
            "/auth/refresh",
            headers={"Authorization": f"Bearer {self._tokens.refresh_token}"},
        )
        data = self._unwrap_api_data(response, context="refresh token")
        refreshed_access_token = data.get("accessToken") or data.get("access_token")
        if not refreshed_access_token:
            raise ServerAPIError(
                "Server response for refresh token did not include an access token",
                detail=response,
            )

        self._tokens = TokenPair(
            access_token=str(refreshed_access_token),
            refresh_token=self._tokens.refresh_token,
            token_type=str(
                data.get("tokenType")
                or data.get("token_type")
                or self._tokens.token_type
                or "bearer"
            ),
            user_id=self._tokens.user_id,
            expires_in=int(data.get("expiresIn") or data.get("expires_in"))
            if (data.get("expiresIn") or data.get("expires_in")) is not None
            else None,
        )

        logger.debug("Token refreshed successfully")
        return self._tokens

    async def logout(self) -> OperationResult:
        """Log out, clear local tokens, and return a normalized operation result."""
        message = "Successfully logged out. Please discard your tokens."
        if self._tokens:
            try:
                response = await self.post("/auth/logout")
                if isinstance(response, dict):
                    response_message = response.get("message")
                    if isinstance(response_message, str) and response_message:
                        message = response_message
            except Exception as e:
                logger.warning(f"Logout request failed: {e}")
            finally:
                self._tokens = None
        return OperationResult(message=message)

    def set_tokens(self, tokens: TokenPair | None) -> None:
        """Set authentication tokens directly (e.g., from stored credentials)."""
        self._tokens = tokens

    def get_tokens(self) -> TokenPair | None:
        """Get current tokens."""
        return self._tokens

    def is_authenticated(self) -> bool:
        """Check if we have valid tokens."""
        return self._tokens is not None

    # ── Conversation Methods ────────────────────────────────────────────

    # ── Message Methods ─────────────────────────────────────────────────

    async def create_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a message."""
        return await self.post("/messages/", json=payload)

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """Fetch a specific message."""
        return await self.get(f"/messages/{message_id}")

    async def list_user_messages(
        self,
        *,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List user messages."""
        return await self.get("/messages/", params={"page": page, "limit": limit})

    async def stream_message(
        self,
        conversation_id: str,
        content: str,
        device_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Send a message and stream the response.

        Args:
            conversation_id: The conversation ID.
            content: The message content.
            device_id: Optional device ID for tool dispatch context.

        Yields:
            SSE events from the server.
        """
        # The AI SDK route expects the canonical ``{"messages": [...]}``
        # payload; the server picks the latest user message and relies on
        # server-side memory for prior turns. Sending the raw ``{"content":
        # ...}`` shape produced 422s.
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
        }
        if device_id:
            canonical_device_id = str(UUID(str(device_id)))
            if str(device_id).strip().lower() != canonical_device_id:
                raise ValueError("device_id must be a canonical hyphenated UUID")
            payload["device_id"] = canonical_device_id

        async for event in self.stream_sse(
            f"/api/chat/{conversation_id}",
            json=payload,
        ):
            yield event

    async def stream_internal_message(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Proxy the server's internal message SSE stream."""
        async for event in self.stream_sse("/messages/stream", json=payload):
            yield event

    async def resume_interrupt(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Resume an interrupted internal stream."""
        async for event in self.stream_sse("/messages/resume-interrupt", json=payload):
            yield event

    async def stream_ai_sdk_chat(
        self,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Proxy the AI SDK UI message stream."""
        async for event in self.stream_sse(f"/api/chat/{conversation_id}", json=payload):
            yield event

    async def resume_ai_sdk_interrupt(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Resume an interrupted AI SDK stream."""
        async for event in self.stream_sse("/ai/resume-interrupt", json=payload):
            yield event

    # ── Client Runtime Methods ───────────────────────────────────────────

    async def register_device(
        self,
        *,
        device_identifier: str,
        display_name: str,
        platform: str,
        app_version: str,
        runtime_version: str,
        capabilities: dict[str, Any] | None = None,
    ) -> DeviceRegistrationResult:
        """Register the current local runtime and return the normalized runtime session payload."""
        response = await self.post(
            "/client-devices/register",
            json={
                "device_identifier": device_identifier,
                "display_name": display_name,
                "platform": platform,
                "app_version": app_version,
                "runtime_version": runtime_version,
                "capabilities": capabilities or {},
            },
        )
        return self._parse_typed_payload(
            response,
            model=DeviceRegistrationResult,
            context="register device",
        )

    async def update_device_tool_catalog(
        self,
        *,
        device_id: str,
        catalog: dict[str, Any],
    ) -> CatalogSyncResult:
        """Sync the sanitized local tool catalog and return the normalized sync result."""
        response = await self.put(
            f"/client-devices/{device_id}/tool-catalog",
            json={"device_id": device_id, "catalog": catalog},
        )
        return self._parse_typed_payload(
            response,
            model=CatalogSyncResult,
            context="update device tool catalog",
        )

    async def update_device_skill_catalog(
        self,
        *,
        device_id: str,
        catalog: dict[str, Any],
    ) -> CatalogSyncResult:
        """Sync the local skill catalog and return the normalized sync result."""
        response = await self.put(
            f"/client-devices/{device_id}/skill-catalog",
            json={"device_id": device_id, "catalog": catalog},
        )
        return self._parse_typed_payload(
            response,
            model=CatalogSyncResult,
            context="update device skill catalog",
        )

    async def list_connected_devices(self) -> dict[str, Any]:
        """Return the authenticated user's currently connected runtime devices."""
        return await self.get("/device-runtime/connected-devices")

    def build_runtime_websocket_url(
        self,
        *,
        device_id: str,
        session_id: str,
    ) -> str:
        """Build the outbound device-runtime WebSocket URL for the server."""
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = f"/device-runtime/{device_id}/connect"
        query = urlencode({"session_id": session_id})
        return urlunparse((scheme, parsed.netloc, path, "", query, ""))

    # ── Document Methods ────────────────────────────────────────────────

    async def upload_documents_bytes_with_status(
        self,
        *,
        conversation_id: str,
        files: list[dict[str, Any]],
    ) -> "UploadProxyResponse":
        """Forward a batch of files to the canonical batch upload endpoint.

        Returns the upstream status code alongside the parsed body so the
        sidecar can preserve 207/409 outcomes for the UI without flattening
        them into 500s.
        """
        multipart = [
            ("files", (item["filename"], item["content"], item["content_type"])) for item in files
        ]
        response = await self.request_response(
            "POST",
            "/documents/uploads",
            data={"conversation_id": conversation_id},
            files=multipart,
        )
        payload = await self._handle_batch_upload_response(response)
        return UploadProxyResponse(status_code=response.status_code, payload=payload)

    async def upload_documents_bytes(
        self,
        *,
        conversation_id: str,
        files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Convenience wrapper that returns the payload only."""
        result = await self.upload_documents_bytes_with_status(
            conversation_id=conversation_id,
            files=files,
        )
        return result.payload

    async def _handle_batch_upload_response(self, response: httpx.Response) -> dict[str, Any]:
        """Treat 201/207/409 as structured batch responses; raise otherwise."""
        if response.status_code in {200, 201, 207, 400, 409}:
            body = await response.aread()
            if not body:
                return {}
            try:
                return response.json()
            except Exception as exc:
                raise ServerAPIError(
                    f"Server returned a non-JSON batch upload response: {response.status_code}",
                    status_code=response.status_code,
                    detail=body.decode("utf-8", errors="replace"),
                ) from exc
        return await self._handle_response(response)

    async def upload_document_bytes(
        self,
        conversation_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Upload one document through the legacy single-file contract."""
        response = await self.request_response(
            "POST",
            "/documents/upload",
            data={"conversation_id": conversation_id},
            files={"file": (filename, content, content_type)},
        )
        return await self._handle_response(response)

    async def upload_document(
        self,
        conversation_id: str,
        file_path: str,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Upload a document to a conversation.

        Args:
            conversation_id: The conversation ID.
            file_path: Local path to the file.
            filename: Optional override for the filename.

        Returns:
            Document metadata from the server.
        """
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return await self.upload_document_bytes(
            conversation_id=conversation_id,
            filename=filename or path.name,
            content=path.read_bytes(),
        )


# Global client instance
_server_client: ServerAPIClient | None = None


def get_server_client() -> ServerAPIClient:
    """Get the global server API client."""
    global _server_client
    if _server_client is None:
        _server_client = ServerAPIClient()
    return _server_client


async def close_server_client() -> None:
    """Close the global server API client."""
    global _server_client
    if _server_client:
        await _server_client.close()
        _server_client = None
