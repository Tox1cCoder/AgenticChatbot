"""Minimal secret lookup for skill readiness checks.

This is intentionally the smallest thing that can answer "is this secret
available?" for :mod:`client_backend.services.skill_runtime.manager`. It
reads from the process environment (or an injected mapping in tests) and
does nothing else.

Task 10 owns the real secret store: encrypted per-profile storage, a secret
management API, and redaction helpers for anything that echoes secret state
back to a client or the server. Do NOT add encryption, redaction, or API
surface to this class before that task — it will extend
:class:`SkillSecretStore` rather than replace it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


class SkillSecretStore:
    """Looks up secret values by name from an environment-like mapping."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ: Mapping[str, str] = environ if environ is not None else os.environ

    def get(self, name: str) -> str | None:
        """Return the secret's value, or None if it is not set."""
        return self._environ.get(name)

    def has(self, name: str) -> bool:
        """Return True iff the secret is present and non-empty."""
        value = self.get(name)
        return bool(value)
