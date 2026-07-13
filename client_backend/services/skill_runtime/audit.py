"""Per-execution JSONL audit trail for skill capability executions.

Every call through :class:`~client_backend.services.skill_runtime.execution.
SkillExecutionEngine` writes exactly one JSON line here describing WHO ran
WHAT capability, WHEN, with what outcome -- never the raw stdout/stderr and
never a resolved secret value. Audit is profile-scoped (one JSONL file per
user, under their profile directory), so with no active user session there
is nothing to scope the record to and writing is skipped rather than
degrading to some shared/global location.

Auditing is a defensive, best-effort side channel: any failure while
building or appending a record (disk full, serialization error, no profile)
is logged and swallowed. An audit-trail bug must never turn a successful
skill execution into a failure, nor crash a failed one further.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from client_backend.core.logging import get_logger
from client_backend.core.paths import get_profile_subdir
from client_backend.services.skill_runtime.secrets import redact_secret_values
from client_backend.services.upstream_auth import get_upstream_auth_service

logger = get_logger(__name__)

_AUDIT_FILENAME = "audit.jsonl"

_UNAVAILABLE_ARGUMENTS = {"<unavailable>": True}


def new_audit_id() -> str:
    """Return a fresh, sortable, unique audit id for one execution."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"skill-exec-{timestamp}-{uuid4().hex[:6]}"


class SkillAuditWriter:
    """Appends one JSON line per skill capability execution to a profile-scoped log."""

    def write(
        self,
        *,
        audit_id: str,
        skill: str,
        capability: str,
        qualified_id: str,
        arguments: dict,
        secret_values: set[str],
        status: str,
        duration_ms: int,
        error_code: str | None = None,
        device_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Write one audit record. Never raises -- failures are logged and swallowed."""
        try:
            resolved_user_id = user_id or get_upstream_auth_service().get_current_user_id()
            if not resolved_user_id:
                logger.debug(
                    "skipping audit record for %s: no active user profile", qualified_id
                )
                return

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "audit_id": audit_id,
                "user_id": resolved_user_id,
                "device_id": device_id,
                "session_id": session_id,
                "skill": skill,
                "capability": capability,
                "qualified_id": qualified_id,
                "arguments_redacted": self._redact_arguments(arguments, secret_values),
                "status": status,
                "duration_ms": duration_ms,
                "error_code": error_code,
            }

            audit_path = get_profile_subdir(resolved_user_id, "skills") / _AUDIT_FILENAME
            self._append_line(audit_path, json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001 - auditing must never break execution
            logger.warning("failed to write audit record for %s", qualified_id, exc_info=True)

    @staticmethod
    def _append_line(audit_path: Path, line: str) -> None:
        """Append one already-serialized JSON line to the audit file.

        Isolated as its own method so the on-disk append step (the one part
        of ``write`` that can fail for external reasons -- disk full,
        permissions) is a single, easily-faked seam in tests.
        """
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    @classmethod
    def _redact_arguments(cls, arguments: dict, secret_values: set[str]) -> object:
        """Strip any resolved secret value from ``arguments``, preserving structure.

        Redaction walks the RAW structure and replaces secret values inside
        string leaves directly -- NOT against the JSON-serialized text.
        Redacting the serialized form would silently miss any secret
        containing a character ``json.dumps`` escapes (a quote, a backslash,
        or -- with the default ``ensure_ascii=True`` -- any non-ASCII
        character), because the raw secret then never appears as a literal
        substring of the escaped text and ``json.loads`` would reconstruct it
        verbatim. Skill arguments are non-secret by design (secrets live in
        the store, injected at execution time), so this is a defensive
        backstop -- but it must actually hold for arbitrary secret values.

        The result is then round-tripped through JSON (with ``default=str``)
        so the record is guaranteed serializable when the whole line is
        written, even if an argument value was an exotic type.
        """
        try:
            redacted = cls._redact_structure(arguments, secret_values)
            return json.loads(json.dumps(redacted, default=str))
        except Exception:  # noqa: BLE001 - never let a bad argument shape break audit
            return dict(_UNAVAILABLE_ARGUMENTS)

    @classmethod
    def _redact_structure(cls, value: object, secret_values: set[str]) -> object:
        """Recursively redact secret values in string leaves of a nested value."""
        if isinstance(value, str):
            return redact_secret_values(value, secret_values)
        if isinstance(value, dict):
            return {k: cls._redact_structure(v, secret_values) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._redact_structure(item, secret_values) for item in value]
        return value
