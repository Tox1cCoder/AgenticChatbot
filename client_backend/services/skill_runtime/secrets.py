"""Encrypted, profile-local secret bindings scoped by skill name."""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Iterable
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

_SECRETS_FILENAME = "secrets.json"
_STORAGE_VERSION = 1
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REDACTION_PLACEHOLDER = "<redacted>"


def redact_secret_values(text: str, values: Iterable[str]) -> str:
    redacted = text
    for value in sorted(set(values), key=len, reverse=True):
        if value:
            redacted = redacted.replace(value, _REDACTION_PLACEHOLDER)
    return redacted


class SkillSecretStore:
    """Persist and retrieve only the secret bindings owned by one skill."""

    def get_for_skill(self, skill_name: str) -> dict[str, str]:
        skills = self._read_bindings()
        values = skills.get(skill_name)
        return dict(values) if isinstance(values, dict) else {}

    def list_for_skill(self, skill_name: str) -> list[str]:
        return sorted(self.get_for_skill(skill_name))

    def set_for_skill(self, skill_name: str, name: str, value: str) -> None:
        self._validate_binding(skill_name, name)
        user_id = self._require_user_id()
        bindings = self._read_bindings(user_id=user_id)
        bindings.setdefault(skill_name, {})[name] = value
        self._write_bindings(user_id, bindings)

    def delete_for_skill(self, skill_name: str, name: str) -> bool:
        self._validate_binding(skill_name, name)
        user_id = self._require_user_id()
        bindings = self._read_bindings(user_id=user_id)
        skill_bindings = bindings.get(skill_name)
        if not isinstance(skill_bindings, dict) or name not in skill_bindings:
            return False
        del skill_bindings[name]
        if not skill_bindings:
            bindings.pop(skill_name, None)
        self._write_bindings(user_id, bindings)
        return True

    def remove_skill(self, skill_name: str) -> bool:
        user_id = self._require_user_id()
        bindings = self._read_bindings(user_id=user_id)
        if skill_name not in bindings:
            return False
        del bindings[skill_name]
        self._write_bindings(user_id, bindings)
        return True

    @staticmethod
    def _validate_binding(skill_name: str, name: str) -> None:
        if not skill_name.strip():
            raise ValueError("skill name must not be empty")
        if not _ENV_NAME.fullmatch(name):
            raise ValueError("secret name must be a valid environment variable identifier")

    @staticmethod
    def _resolve_user_id() -> str | None:
        return get_upstream_auth_service().get_current_user_id()

    def _require_user_id(self) -> str:
        user_id = self._resolve_user_id()
        if not user_id:
            raise RuntimeError("no active user profile for secret storage")
        return user_id

    @staticmethod
    def _secrets_path(user_id: str) -> Path:
        return get_profile_subdir(user_id, "skills") / _SECRETS_FILENAME

    def _read_bindings(self, user_id: str | None = None) -> dict[str, dict[str, str]]:
        user_id = user_id or self._resolve_user_id()
        if not user_id:
            return {}
        path = self._secrets_path(user_id)
        if not path.is_file():
            return {}
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict):
                raise ValueError("secret envelope is not an object")
            plaintext = decrypt_local_secret(envelope)
            payload = json.loads(plaintext.decode("utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != _STORAGE_VERSION:
                raise ValueError("unsupported secret storage format")
            raw_skills = payload.get("skills")
            if not isinstance(raw_skills, dict):
                raise ValueError("secret bindings are not an object")
        except (OSError, ValueError, LocalSecretStorageError) as exc:
            logger.warning(
                "Failed to read skill secret bindings for user %s (%s); treating as empty",
                user_id,
                exc,
            )
            return {}

        result: dict[str, dict[str, str]] = {}
        for skill_name, raw_values in raw_skills.items():
            if not isinstance(raw_values, dict):
                continue
            result[str(skill_name)] = {str(name): str(value) for name, value in raw_values.items()}
        return result

    def _write_bindings(
        self,
        user_id: str,
        bindings: dict[str, dict[str, str]],
    ) -> None:
        payload = {"version": _STORAGE_VERSION, "skills": bindings}
        envelope = encrypt_local_secret(json.dumps(payload).encode("utf-8"))
        path = self._secrets_path(user_id)
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
