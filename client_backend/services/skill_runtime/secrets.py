"""Secret lookup and encrypted storage for skill readiness checks and execution.

Answers "is this secret available?" and "what is its value?" for
:mod:`client_backend.services.skill_runtime.manager` and
:mod:`client_backend.services.skill_runtime.execution`, and backs the local
secret-management API in :mod:`client_backend.api.skills`.

Lookup order is: encrypted per-profile storage first, then the process
environment (or an injected mapping in tests). This lets a user set a secret
through the local API without needing to restart the process with a new
environment variable, while keeping the original env-only behavior intact for
callers (including tests) that never touch profile storage.

Profile storage is per active user (resolved via
:func:`client_backend.services.upstream_auth.get_upstream_auth_service`), so
without an active user id all profile operations degrade gracefully: reads
behave as env-only, and writes raise a clear error rather than silently doing
nothing.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from client_backend.core.logging import get_logger
from client_backend.core.paths import get_profile_subdir
from client_backend.core.security import (
    LocalSecretStorageError,
    decrypt_local_secret,
    encrypt_local_secret,
)
from client_backend.services.upstream_auth import get_upstream_auth_service

logger = get_logger(__name__)

# One JSON file per profile holding the `encrypt_local_secret` envelope
# ({version, encryption, ciphertext}) whose plaintext is the {name: value}
# JSON. No sibling key file: on Windows the envelope is OS-user-bound via
# DPAPI, and the non-Windows Fernet fallback's key is managed centrally by
# client_backend.core.security (the same primitive upstream_auth uses).
_SECRETS_FILENAME = "secrets.json"

_REDACTION_PLACEHOLDER = "<redacted>"


def redact_secret_values(text: str, values: Iterable[str]) -> str:
    """Replace every occurrence of each value in ``text`` with a placeholder.

    Values are replaced LONGEST-FIRST. If one secret value is a substring of
    another (e.g. ``"abc"`` and ``"abcdef"``), replacing the shorter one first
    would fragment the longer one and leak its tail; replacing the longest
    first makes redaction complete and independent of iteration order. Empty
    values are skipped so an unset secret can never become a no-op
    ``str.replace("", ...)`` that would corrupt the output.

    This is the shared home for the redaction logic the execution engine's
    own ``_redact`` helper implements privately for subprocess output; use
    this function for anything else that echoes secret-bearing text back to
    a client or a log (e.g. API error messages).
    """
    redacted = text
    for value in sorted(set(values), key=len, reverse=True):
        if not value:
            continue
        redacted = redacted.replace(value, _REDACTION_PLACEHOLDER)
    return redacted


class SkillSecretStore:
    """Looks up secret values by name: stored profile secret first, then env."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ: Mapping[str, str] = environ if environ is not None else os.environ

    def get(self, name: str) -> str | None:
        """Return the secret's value, or None if it is not set anywhere."""
        stored = self._read_profile_secrets()
        if name in stored:
            return stored[name]
        return self._environ.get(name)

    def has(self, name: str) -> bool:
        """Return True iff the secret is present and non-empty."""
        return bool(self.get(name))

    def set(self, name: str, value: str) -> None:
        """Persist a secret value into the active user's encrypted profile store.

        Raises:
            RuntimeError: If there is no active user profile to store into.
        """
        user_id = self._require_user_id()
        stored = self._read_profile_secrets(user_id=user_id)
        stored[name] = value
        self._write_profile_secrets(user_id, stored)

    def delete(self, name: str) -> bool:
        """Remove a stored secret. Returns whether it existed.

        Raises:
            RuntimeError: If there is no active user profile to modify.
        """
        user_id = self._require_user_id()
        stored = self._read_profile_secrets(user_id=user_id)
        if name not in stored:
            return False
        del stored[name]
        self._write_profile_secrets(user_id, stored)
        return True

    def list_stored_names(self) -> set[str]:
        """Return the names (never values) present in the profile store."""
        return set(self._read_profile_secrets().keys())

    # -- profile storage internals -------------------------------------

    def _resolve_user_id(self) -> str | None:
        return get_upstream_auth_service().get_current_user_id()

    def _require_user_id(self) -> str:
        user_id = self._resolve_user_id()
        if not user_id:
            raise RuntimeError("no active user profile for secret storage")
        return user_id

    def _secrets_path(self, user_id: str) -> Path:
        return get_profile_subdir(user_id, "skills") / _SECRETS_FILENAME

    def _read_profile_secrets(self, user_id: str | None = None) -> dict[str, str]:
        """Load the stored `{name: value}` dict, tolerating any missing/corrupt state.

        Never raises: a missing file, unreadable file, malformed envelope, or a
        payload that fails to decrypt all just mean "no secrets stored yet"
        from the caller's point of view.
        """
        if user_id is None:
            user_id = self._resolve_user_id()
        if not user_id:
            return {}

        secrets_path = self._secrets_path(user_id)
        if not secrets_path.exists():
            return {}

        try:
            envelope = json.loads(secrets_path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict):
                raise ValueError("stored secret envelope is not a JSON object")
            plaintext = decrypt_local_secret(envelope)
            data = json.loads(plaintext.decode("utf-8"))
        except (OSError, ValueError, LocalSecretStorageError) as exc:
            logger.warning(
                "Failed to read stored skill secrets for user %s (%s); treating as empty",
                user_id,
                exc,
            )
            return {}

        if not isinstance(data, dict):
            logger.warning(
                "Stored skill secrets for user %s are not a JSON object; treating as empty",
                user_id,
            )
            return {}

        return {str(k): str(v) for k, v in data.items()}

    def _write_profile_secrets(self, user_id: str, secrets: dict[str, str]) -> None:
        # Delegate at-rest protection to the shared local-secret primitive:
        # OS-user-bound DPAPI on Windows, managed Fernet elsewhere. No sibling
        # key file for this module to generate or guard.
        envelope = encrypt_local_secret(json.dumps(secrets).encode("utf-8"))
        secrets_path = self._secrets_path(user_id)
        secrets_path.write_text(json.dumps(envelope), encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(secrets_path, 0o600)
