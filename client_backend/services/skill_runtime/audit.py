"""Per-execution JSONL audit trail for device-local skill commands.

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
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from client_backend.core.logging import get_logger
from client_backend.core.paths import get_profile_subdir
from client_backend.services.skill_runtime.secrets import redact_secret_values
from client_backend.services.upstream_auth import get_upstream_auth_service

logger = get_logger(__name__)

_AUDIT_FILENAME = "audit.jsonl"
_LIFECYCLE_AUDIT_FILENAME = "lifecycle.jsonl"

_UNAVAILABLE_ARGUMENTS = {"<unavailable>": True}

# The complete set of keys a lifecycle record may contain. This is an allowlist
# rather than a denylist on purpose: callers pass through archive-derived and
# filesystem-derived values, so anything not named here -- a staging path, an
# uploaded filename, setup output, a token -- is dropped instead of audited.
LIFECYCLE_AUDIT_FIELDS = frozenset(
    {
        "timestamp",
        "event",
        "user_id",
        "device_id",
        "upload_id",
        "operation_id",
        "skill",
        "source_hash",
        "status",
        "phase",
        "error_code",
        "duration_ms",
        "sync_status",
        "metrics",
    }
)


def new_audit_id() -> str:
    """Return a fresh, sortable, unique audit id for one execution."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"skill-exec-{timestamp}-{uuid4().hex[:6]}"


class SkillAuditWriter:
    """Append one JSON line per skill command attempt to a profile-scoped log."""

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
                logger.debug("skipping audit record for %s: no active user profile", qualified_id)
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


class SkillLifecycleAuditWriter:
    """Append one JSON line per upload/installation lifecycle transition.

    Separate from :class:`SkillAuditWriter` because the two answer different
    questions from different inputs. Execution audit records what a skill command
    did; this records how a bundle arrived and was installed, and its inputs are
    attacker-influenced (archive member names, front-matter skill names, uploaded
    filenames). Fields are therefore allowlisted by
    :data:`LIFECYCLE_AUDIT_FIELDS` and anything else a caller passes is dropped.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    def write(
        self,
        *,
        event: str,
        user_id: str,
        device_id: str | None = None,
        upload_id: str | None = None,
        operation_id: str | None = None,
        skill: str | None = None,
        source_hash: str | None = None,
        status: str | None = None,
        phase: str | None = None,
        error_code: str | None = None,
        duration_ms: int | None = None,
        sync_status: str | None = None,
        metrics: Mapping[str, int] | None = None,
        **ignored: object,
    ) -> None:
        """Write one lifecycle record. Never raises -- failures are logged only.

        Unknown keyword arguments are accepted and discarded so a future caller
        cannot leak a new field by passing it; ``**ignored`` exists to swallow
        them, not to forward them.
        """
        try:
            if ignored:
                logger.debug(
                    "dropping %d non-allowlisted lifecycle audit field(s) for %s",
                    len(ignored),
                    event,
                )
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": str(event),
                "user_id": str(user_id),
                "device_id": _optional_str(device_id),
                "upload_id": _optional_str(upload_id),
                "operation_id": _optional_str(operation_id),
                "skill": _optional_str(skill),
                "source_hash": _optional_str(source_hash),
                "status": _optional_str(status),
                "phase": _optional_str(phase),
                "error_code": _optional_str(error_code),
                "duration_ms": int(duration_ms) if duration_ms is not None else None,
                "sync_status": _optional_str(sync_status),
                "metrics": _numeric_metrics(metrics),
            }
            record = {key: value for key, value in record.items() if value is not None}
            self._append_line(self._resolve_path(user_id), json.dumps(record) + "\n")
        except Exception:  # noqa: BLE001 - auditing must never break installation
            logger.warning("failed to write skill lifecycle audit record", exc_info=True)

    def _resolve_path(self, user_id: str) -> Path:
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return self._path
        return get_profile_subdir(user_id, "skills") / _LIFECYCLE_AUDIT_FILENAME

    @staticmethod
    def _append_line(audit_path: Path, line: str) -> None:
        """Append one serialized record; the single failure seam for tests."""
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _optional_str(value: object) -> str | None:
    """Normalize an optional identifier to a non-empty string."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _numeric_metrics(metrics: Mapping[str, int] | None) -> dict[str, int] | None:
    """Keep only integral counters, dropping anything that could carry text.

    Byte and file counts are safe to record; a caller that slips a path or a
    filename in under a counter key must not have it persisted.
    """
    if not metrics:
        return None
    numeric = {
        str(key): int(value)
        for key, value in metrics.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return numeric or None
