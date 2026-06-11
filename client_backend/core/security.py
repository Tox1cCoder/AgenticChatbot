"""
Security utilities for the client backend.

Handles local session tokens and secure storage patterns.
"""

import base64
import contextlib
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import jwt
from pydantic import BaseModel

from client_backend.core.config import client_settings, initialize_client_environment
from client_backend.core.logging import get_logger

logger = get_logger(__name__)


class LocalSessionPayload(BaseModel):
    """Payload for local session tokens."""

    user_id: str
    server_user_id: str
    device_id: str | None = None
    device_identifier: str
    exp: datetime
    iat: datetime


class LocalSessionError(Exception):
    """Raised when local session operations fail."""

    pass


def _get_local_session_secret() -> str:
    settings = initialize_client_environment()
    return settings.local_session_secret


def create_local_session_token(
    user_id: str,
    server_user_id: str,
    device_identifier: str,
    device_id: str | None = None,
) -> str:
    """
    Create a local session token for UI to client backend communication.

    Args:
        user_id: The local user identifier.
        server_user_id: The server-side user ID.
        device_identifier: Stable installation identifier.
        device_id: Server-assigned device UUID when available.

    Returns:
        A signed JWT token for local session.
    """
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=client_settings.local_session_expire_minutes)

    payload = {
        "user_id": user_id,
        "server_user_id": server_user_id,
        "device_id": device_id,
        "device_identifier": device_identifier,
        "exp": expires,
        "iat": now,
        "jti": secrets.token_urlsafe(16),
    }

    return jwt.encode(
        payload,
        _get_local_session_secret(),
        algorithm="HS256",
    )


def verify_local_session_token(token: str) -> LocalSessionPayload:
    """
    Verify and decode a local session token.

    Args:
        token: The JWT token to verify.

    Returns:
        The decoded session payload.

    Raises:
        LocalSessionError: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(
            token,
            _get_local_session_secret(),
            algorithms=["HS256"],
        )
        return LocalSessionPayload(
            user_id=payload["user_id"],
            server_user_id=payload["server_user_id"],
            device_id=payload.get("device_id"),
            device_identifier=payload.get("device_identifier") or payload["device_id"],
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
        )
    except jwt.ExpiredSignatureError as e:
        raise LocalSessionError("Local session has expired") from e
    except jwt.InvalidTokenError as e:
        raise LocalSessionError(f"Invalid local session token: {e}") from e


DEVICE_IDENTITY_FILENAME = "device_identity.json"


def generate_device_identifier(config_dir: Path | str | None = None) -> str:
    """
    Return this installation's device identifier, unique per installation and
    stable across restarts.

    The identifier is random — never derived from machine attributes, so two
    installations on the same machine are distinct devices. It is generated
    once on first run and persisted in the installation's profile directory
    (or ``config_dir`` when given).

    Returns:
        A device identifier string.
    """
    root = Path(config_dir) if config_dir is not None else Path(client_settings.profile_root)
    identity_path = root / DEVICE_IDENTITY_FILENAME

    if identity_path.exists():
        try:
            stored = json.loads(identity_path.read_text(encoding="utf-8"))
            identifier = str(stored.get("device_identifier") or "").strip()
            if identifier:
                return identifier
            logger.warning(
                "Device identity file %s has no identifier; regenerating. "
                "This installation will re-register as a new device.",
                identity_path,
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "Device identity file %s is unreadable (%s); regenerating. "
                "This installation will re-register as a new device.",
                identity_path,
                exc,
            )

    identifier = uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(
        json.dumps({"device_identifier": identifier}),
        encoding="utf-8",
    )
    with contextlib.suppress(OSError):
        # Best-effort restrictive permissions (no-op on Windows ACLs).
        os.chmod(identity_path, 0o600)
    return identifier


class LocalSecretStorageError(Exception):
    """Raised when secure local secret storage fails."""

    pass


def _derive_local_storage_key() -> bytes:
    digest = hashlib.sha256(_get_local_session_secret().encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _encrypt_with_fernet(plaintext: bytes) -> str:
    from cryptography.fernet import Fernet

    cipher = Fernet(_derive_local_storage_key())
    return cipher.encrypt(plaintext).decode("utf-8")


def _decrypt_with_fernet(ciphertext: str) -> bytes:
    from cryptography.fernet import Fernet, InvalidToken

    cipher = Fernet(_derive_local_storage_key())
    try:
        return cipher.decrypt(ciphertext.encode("utf-8"))
    except InvalidToken as exc:
        raise LocalSecretStorageError("Encrypted local secret is invalid or corrupted") from exc


if sys.platform == "win32":
    from ctypes import POINTER, Structure, byref, c_char, cast, windll
    from ctypes.wintypes import DWORD

    CRYPTPROTECT_UI_FORBIDDEN = 0x01

    class DATA_BLOB(Structure):
        _fields_ = [("cbData", DWORD), ("pbData", POINTER(c_char))]

    def _to_blob(data: bytes) -> DATA_BLOB:
        buffer = cast(None, POINTER(c_char))
        if data:
            raw = (c_char * len(data)).from_buffer_copy(data)
            buffer = cast(raw, POINTER(c_char))
        blob = DATA_BLOB()
        blob.cbData = len(data)
        blob.pbData = buffer
        blob._buffer = locals().get("raw")
        return blob

    def _blob_to_bytes(blob: DATA_BLOB) -> bytes:
        if not blob.cbData:
            return b""
        return bytes(cast(blob.pbData, POINTER(c_char * blob.cbData)).contents)

    def _encrypt_with_dpapi(plaintext: bytes) -> str:
        input_blob = _to_blob(plaintext)
        output_blob = DATA_BLOB()
        success = windll.crypt32.CryptProtectData(
            byref(input_blob),
            None,
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            byref(output_blob),
        )
        if not success:
            raise LocalSecretStorageError("Windows DPAPI encryption failed")
        try:
            return base64.b64encode(_blob_to_bytes(output_blob)).decode("ascii")
        finally:
            windll.kernel32.LocalFree(output_blob.pbData)

    def _decrypt_with_dpapi(ciphertext: str) -> bytes:
        input_bytes = base64.b64decode(ciphertext.encode("ascii"))
        input_blob = _to_blob(input_bytes)
        output_blob = DATA_BLOB()
        success = windll.crypt32.CryptUnprotectData(
            byref(input_blob),
            None,
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            byref(output_blob),
        )
        if not success:
            raise LocalSecretStorageError("Windows DPAPI decryption failed")
        try:
            return _blob_to_bytes(output_blob)
        finally:
            windll.kernel32.LocalFree(output_blob.pbData)
else:
    _encrypt_with_dpapi = None
    _decrypt_with_dpapi = None


def encrypt_local_secret(plaintext: bytes) -> dict[str, str | int]:
    """
    Encrypt a secret for local at-rest storage.

    Uses OS-native protection on Windows and a Fernet fallback elsewhere.
    """
    if sys.platform == "win32" and _encrypt_with_dpapi is not None:
        return {
            "version": 1,
            "encryption": "windows-dpapi",
            "ciphertext": _encrypt_with_dpapi(plaintext),
        }

    return {
        "version": 1,
        "encryption": "fernet",
        "ciphertext": _encrypt_with_fernet(plaintext),
    }


def decrypt_local_secret(payload: dict[str, Any]) -> bytes:
    """Decrypt a secret stored by `encrypt_local_secret`."""
    encryption = str(payload.get("encryption") or "").strip().lower()
    ciphertext = str(payload.get("ciphertext") or "")
    if not ciphertext:
        raise LocalSecretStorageError("Encrypted local secret is missing ciphertext")

    if encryption == "windows-dpapi":
        if _decrypt_with_dpapi is None:
            raise LocalSecretStorageError(
                "Windows-protected secrets can only be decrypted on Windows"
            )
        return _decrypt_with_dpapi(ciphertext)

    if encryption == "fernet":
        return _decrypt_with_fernet(ciphertext)

    raise LocalSecretStorageError(f"Unsupported local secret encryption mode: {encryption}")


def redact_path_for_audit(path: str, workspace_roots: list[str] | None = None) -> str:
    """
    Redact a path for safe audit logging.

    Replaces user home directories and workspace roots with tokens.

    Args:
        path: The path to redact.
        workspace_roots: Optional list of workspace roots to redact.

    Returns:
        The redacted path string.
    """
    from pathlib import Path

    result = path

    # Redact home directory
    home = str(Path.home())
    if result.startswith(home):
        result = "~" + result[len(home) :]

    # Redact workspace roots
    roots = workspace_roots or client_settings.workspace_roots
    for i, root in enumerate(roots):
        root_str = str(Path(root).resolve())
        if result.startswith(root_str):
            result = f"$WORKSPACE_{i}" + result[len(root_str) :]
            break

    return result


def redact_env_for_audit(env: dict[str, str]) -> dict[str, str]:
    """
    Redact environment variables for safe audit logging.

    Args:
        env: Environment variables dictionary.

    Returns:
        A copy with sensitive values redacted.
    """
    sensitive_patterns = [
        "KEY",
        "SECRET",
        "TOKEN",
        "PASSWORD",
        "CREDENTIAL",
        "AUTH",
        "API_KEY",
    ]

    redacted = {}
    for key, value in env.items():
        key_upper = key.upper()
        if any(pattern in key_upper for pattern in sensitive_patterns):
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value

    return redacted
