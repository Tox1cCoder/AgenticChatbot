"""
Server API client for communicating with the canonical backend.

This module provides an async HTTP client wrapper for all server API calls.
"""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse
from uuid import UUID

import httpx
from pydantic import BaseModel

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger

logger = get_logger(__name__)


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
            raise ServerConnectionError(f"Cannot connect to server at {self.base_url}: {e}")
        except httpx.TimeoutException as e:
            raise ServerConnectionError(f"Server request timed out: {e}")

    async def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle response and raise appropriate errors."""
        if response.status_code == 401:
            raise AuthenticationError(
                "Authentication required",
                status_code=401,
                detail=response.json() if response.content else None,
            )

        if response.status_code == 403:
            raise AuthenticationError(
                "Access forbidden",
                status_code=403,
                detail=response.json() if response.content else None,
            )

        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise ServerAPIError(
                f"Server error: {response.status_code}",
                status_code=response.status_code,
                detail=detail,
            )

        if not response.content:
            return {}

        return response.json()

    async def request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Make an authenticated request to the server.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: API path (e.g., "/api/conversations")
            **kwargs: Additional arguments passed to httpx

        Returns:
            Parsed JSON response.

        Raises:
            ServerConnectionError: If the server is unreachable.
            AuthenticationError: If authentication fails.
            ServerAPIError: For other server errors.
        """
        response = await self.request_response(method, path, **kwargs)
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

        try:
            async with client.stream(
                method,
                path,
                headers=headers,
                **kwargs,
            ) as response:
                if response.status_code >= 400:
                    await self._handle_response(response)

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            import json

                            yield json.loads(data)
                        except Exception:
                            yield {"raw": data}

        except httpx.ConnectError as e:
            raise ServerConnectionError(f"Cannot connect to server at {self.base_url}: {e}")

    # ── Authentication Methods ──────────────────────────────────────────

    async def login(self, email: str, password: str) -> dict[str, Any]:
        """
        Authenticate with the server.

        Args:
            email: User email.
            password: User's password.

        Returns:
            The wrapped upstream ApiResponse payload.
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
        return response

    async def refresh_token(self) -> dict[str, Any]:
        """
        Refresh the access token using the refresh token.

        Returns:
            The wrapped upstream ApiResponse payload.

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
        return response

    async def logout(self) -> None:
        """Log out and clear tokens."""
        if self._tokens:
            try:
                await self.post("/auth/logout")
            except Exception as e:
                logger.warning(f"Logout request failed: {e}")
            finally:
                self._tokens = None

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

    async def list_conversations(
        self,
        page: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List user's conversations."""
        return await self.get(
            "/conversations/",
            params={"page": page, "limit": limit},
        )

    async def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Get a specific conversation."""
        return await self.get(f"/conversations/{conversation_id}")

    async def create_conversation(self, title: str | None = None) -> dict[str, Any]:
        """Create a new conversation."""
        return await self.post(
            "/conversations/",
            json={"title": title} if title else {},
        )

    async def update_conversation(
        self,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a conversation."""
        return await self.patch(
            f"/conversations/{conversation_id}",
            json=payload,
        )

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation."""
        await self.delete(f"/conversations/{conversation_id}")

    # ── Message Methods ─────────────────────────────────────────────────

    async def get_messages(
        self,
        conversation_id: str,
        page: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get messages in a conversation."""
        return await self.get(
            f"/conversations/{conversation_id}/messages",
            params={"page": page, "limit": limit},
        )

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
        payload = {"content": content}
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
    ) -> dict[str, Any]:
        """Register the current local runtime as an active client device."""
        return await self.post(
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

    async def update_device_tool_catalog(
        self,
        *,
        device_id: str,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        """Sync the sanitized local tool catalog to the server."""
        return await self.put(
            f"/client-devices/{device_id}/tool-catalog",
            json={"device_id": device_id, "catalog": catalog},
        )

    async def update_device_skill_catalog(
        self,
        *,
        device_id: str,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        """Sync the local skill catalog to the server."""
        return await self.put(
            f"/client-devices/{device_id}/skill-catalog",
            json={"device_id": device_id, "catalog": catalog},
        )

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

    async def upload_document_bytes(
        self,
        conversation_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """
        Upload a document using in-memory bytes.
        """
        return await self.post(
            "/documents/upload",
            data={"conversation_id": conversation_id},
            files={
                "file": (
                    filename,
                    content,
                    content_type,
                )
            },
        )

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

    async def get_documents(self, conversation_id: str) -> dict[str, Any]:
        """Get documents in a conversation."""
        return await self.get(f"/documents/conversation/{conversation_id}")

    async def get_document_status(self, document_id: str) -> dict[str, Any]:
        """Get a document by ID."""
        return await self.get(f"/documents/{document_id}")

    async def get_document_task_status(self, task_id: str) -> dict[str, Any]:
        """Get document background task status."""
        return await self.get(f"/documents/task/{task_id}")

    async def update_document(self, document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Update a document."""
        return await self.put(f"/documents/{document_id}", json=payload)

    async def delete_document(self, document_id: str) -> dict[str, Any]:
        """Delete a document."""
        return await self.delete(f"/documents/{document_id}")


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
