"""Normalized skill-runtime error codes and lightweight error types.

Cross-cutting: Task 4 (installation) and Task 7 (execution engine) both raise
:class:`SkillRuntimeError` with one of the codes defined here so every
skill-runtime failure -- install-time or execution-time -- surfaces through
the same normalized shape. This module has no dependencies (no pydantic, no
filesystem access) so it can be imported from anywhere without side effects.
"""

from __future__ import annotations

# Installation errors (Task 4).
SKILL_INSTALL_INVALID = "SKILL_INSTALL_INVALID"
SKILL_INSTALL_CONFLICT = "SKILL_INSTALL_CONFLICT"
UNSAFE_BUNDLE_PATH = "UNSAFE_BUNDLE_PATH"
SKILL_SETUP_REQUIRED = "SKILL_SETUP_REQUIRED"
SKILL_SETUP_FAILED = "SKILL_SETUP_FAILED"
SKILL_RUNTIME_STALE = "SKILL_RUNTIME_STALE"
SKILL_PORTABILITY_UNSUPPORTED = "SKILL_PORTABILITY_UNSUPPORTED"

# Readiness / capability lookup errors (Task 5/6/7).
SKILL_NOT_READY = "SKILL_NOT_READY"
CAPABILITY_NOT_FOUND = "CAPABILITY_NOT_FOUND"
MISSING_SECRET = "MISSING_SECRET"

# Permission errors (Task 6).
PERMISSION_REQUIRED = "PERMISSION_REQUIRED"
PERMISSION_DENIED = "PERMISSION_DENIED"

# Execution errors (Task 7).
COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
REMOTE_API_ERROR = "REMOTE_API_ERROR"
RUNTIME_ERROR = "RUNTIME_ERROR"


class SkillRuntimeError(Exception):
    """A normalized, structured skill-runtime failure.

    Carries a stable ``code`` (one of the module-level constants above), a
    human-readable ``message``, and an optional ``repair`` hint dict. Callers
    that assemble a full failure envelope (Task 7/11: ``ok``/``skill``/
    ``capability``/``duration_ms``/``audit_id`` plus this error) read these
    three fields; this class does not build that envelope itself.
    """

    def __init__(self, code: str, message: str, repair: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.repair = repair

    def __str__(self) -> str:
        return self.message


def error_payload(code: str, message: str, repair: dict | None = None) -> dict:
    """Build the ``error`` sub-object of the plan's failure envelope.

    Does not build the full execution envelope (``ok``/``skill``/
    ``capability``/``duration_ms``/``audit_id``) -- that assembly belongs to
    the caller (Task 7/11).
    """
    return {"code": code, "message": message, "repair": repair}
