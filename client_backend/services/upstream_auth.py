"""
Upstream authentication service for managing server credentials.

Handles token storage, refresh, and credential management for the server connection.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import profile_subdir_path
from client_backend.core.security import decrypt_local_secret, encrypt_local_secret
from client_backend.services.server_api import (
    AuthenticationError,
    OperationResult,
    ServerAPIClient,
    ServerAPIError,
    ServerConnectionError,
    TokenPair,
    get_server_client,
)

logger = get_logger(__name__)


class StoredCredentials(BaseModel):
    """Stored credentials for a user session."""

    user_id: str
    username: str
    tokens: TokenPair
    stored_at: datetime
    server_url: str


class UpstreamAuthService:
    """
    Service for managing upstream server authentication.

    Handles:
    - Login/logout with the canonical server
    - Token refresh
    - Secure token storage (file-based, with OS keychain support planned)
    """

    def __init__(self, server_client: ServerAPIClient | None = None):
        self._client = server_client or get_server_client()
        self._current_user_id: str | None = None
        self._credentials: StoredCredentials | None = None

    def _get_credentials_path(self, user_id: str) -> Path:
        # Resolution only: a read must not create a directory for an id that
        # arrived in a query string. _save_credentials creates the parent.
        return profile_subdir_path(user_id, "session") / "credentials.json"

    def _save_credentials(self, credentials: StoredCredentials) -> None:
        """
        Save credentials to disk.

        Uses OS-protected encryption on Windows and an encrypted file fallback elsewhere.
        """
        path = self._get_credentials_path(credentials.user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        encrypted_payload = encrypt_local_secret(credentials.model_dump_json().encode("utf-8"))

        # Write atomically
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(encrypted_payload, indent=2), encoding="utf-8")
        temp_path.replace(path)

        logger.debug(f"Saved credentials for user {credentials.user_id}")

    def _load_credentials(self, user_id: str) -> StoredCredentials | None:
        try:
            path = self._get_credentials_path(user_id)
        except ValueError:
            logger.warning("Refusing to load credentials for a malformed user id")
            return None
        if not path.exists():
            return None

        try:
            raw_data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw_data, dict) and "ciphertext" in raw_data:
                decrypted = decrypt_local_secret(raw_data)
                data = json.loads(decrypted.decode("utf-8"))
                return StoredCredentials(**data)

            credentials = StoredCredentials(**raw_data)
            self._save_credentials(credentials)
            return credentials
        except Exception as e:
            # Type only: a pydantic error repeats its input, which holds the tokens.
            logger.warning("Failed to load credentials: %s", type(e).__name__)
            return None

    def _clear_credentials(self, user_id: str) -> None:
        """Clear stored credentials."""
        path = self._get_credentials_path(user_id)
        if path.exists():
            path.unlink()
            logger.debug(f"Cleared credentials for user {user_id}")

    async def login(self, email: str, password: str) -> TokenPair:
        tokens = await self._client.login(email, password)
        if not tokens.user_id:
            raise AuthenticationError("Login succeeded but no user ID was returned by the server")

        user_id = str(tokens.user_id)

        # Store credentials
        self._credentials = StoredCredentials(
            user_id=user_id,
            username=email,
            tokens=tokens,
            stored_at=datetime.now(UTC),
            server_url=self._client.base_url,
        )
        self._current_user_id = user_id
        self._save_credentials(self._credentials)

        logger.info("Logged in as %s (user_id: %s)", email, user_id)
        return tokens

    async def logout(self) -> OperationResult:
        """Logout from the upstream server."""
        if self._current_user_id:
            response = OperationResult(
                message="Successfully logged out. Please discard your tokens."
            )
            if self._client.is_authenticated():
                response = await self._client.logout()
            self._clear_credentials(self._current_user_id)
            self._current_user_id = None
            self._credentials = None
            logger.info("Logged out")
            return response
        return OperationResult(message="Successfully logged out. Please discard your tokens.")

    async def restore_session(self, user_id: str) -> bool:
        """Restore a session from stored credentials; True when it is usable."""
        credentials = self._load_credentials(user_id)
        if not credentials:
            logger.debug(f"No stored credentials for user {user_id}")
            return False

        # Check if server URL matches
        if credentials.server_url != self._client.base_url:
            logger.warning(
                f"Stored credentials are for different server: "
                f"{credentials.server_url} vs {self._client.base_url}"
            )
            return False

        previous_tokens = self._client.get_tokens()
        self._client.set_tokens(credentials.tokens)

        try:
            # Try to refresh the token to verify it's still valid
            await self._client.refresh_token()
            new_tokens = self._client.get_tokens()
            if new_tokens is None:
                raise AuthenticationError("Token refresh did not yield active credentials")

            # Update stored credentials with new tokens
            credentials.tokens = new_tokens
            credentials.stored_at = datetime.now(UTC)
            self._save_credentials(credentials)

            self._credentials = credentials
            self._current_user_id = user_id

            logger.info(f"Restored session for user {user_id}")
            return True

        except AuthenticationError as e:
            # The refresh token was genuinely rejected by the server — the stored
            # credentials are dead, so clear them and require a fresh login.
            logger.warning(f"Stored credentials rejected during restore: {e}")
            self._clear_credentials(user_id)
            # The shared client must not keep bearing tokens the server just
            # rejected; every later request would carry them.
            self._client.set_tokens(previous_tokens)
            return False
        except (ServerConnectionError, ServerAPIError) as e:
            # Transient failure (server unreachable, timeout, 5xx). Do NOT wipe the
            # stored credentials over a hiccup — keep the session set so ordinary
            # requests (which refresh-and-retry on 401) can recover once the server
            # is reachable again, instead of forcing the user to log in.
            logger.warning(
                "Could not verify session for user %s (transient: %s); keeping credentials.",
                user_id,
                e,
            )
            self._credentials = credentials
            self._current_user_id = user_id
            return True

    async def refresh(self) -> TokenPair:
        """Refresh the current upstream access token and persist the new value."""
        if not self._credentials:
            raise AuthenticationError("No active session to refresh")

        new_tokens = await self._client.refresh_token()
        self._credentials.tokens = new_tokens
        self._credentials.stored_at = datetime.now(UTC)
        self._save_credentials(self._credentials)
        return new_tokens

    def get_current_user_id(self) -> str | None:
        """Get the current authenticated user ID."""
        return self._current_user_id

    def get_current_username(self) -> str | None:
        """Return the login identifier for the current active upstream session."""
        if self._credentials is None:
            return None
        return self._credentials.username

    def is_authenticated(self) -> bool:
        """Check if we have an active authenticated session."""
        return self._current_user_id is not None and self._client.is_authenticated()

    def get_current_access_token(self) -> str | None:
        """Return the currently active upstream access token."""
        tokens = self._client.get_tokens()
        return tokens.access_token if tokens else None

    def get_current_refresh_token(self) -> str | None:
        """Return the currently active upstream refresh token."""
        tokens = self._client.get_tokens()
        return tokens.refresh_token if tokens else None

    def list_stored_users(self) -> list[str]:
        """List user IDs with stored credentials."""
        profile_root = Path(client_settings.profile_root)
        users = []

        # Look for server hash directories
        for server_dir in profile_root.iterdir():
            if not server_dir.is_dir():
                continue
            # Look for user directories
            for user_dir in server_dir.iterdir():
                if not user_dir.is_dir():
                    continue
                creds_file = user_dir / "session" / "credentials.json"
                if creds_file.exists():
                    users.append(user_dir.name)

        return users


# Global service instance
_auth_service: UpstreamAuthService | None = None


def get_upstream_auth_service() -> UpstreamAuthService:
    """Get the global upstream auth service."""
    global _auth_service
    if _auth_service is None:
        _auth_service = UpstreamAuthService()
    return _auth_service
