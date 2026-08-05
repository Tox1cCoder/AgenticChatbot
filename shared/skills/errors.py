"""Normalized skill-runtime error codes and lightweight error types.

Installation and execution both raise :class:`SkillRuntimeError` with one of
the codes defined here so every
skill-runtime failure -- install-time or execution-time -- surfaces through
the same normalized shape. This module has no dependencies (no pydantic, no
filesystem access) so it can be imported from anywhere without side effects.
"""

from __future__ import annotations

# Installation errors.
SKILL_INSTALL_INVALID = "SKILL_INSTALL_INVALID"
SKILL_INSTALL_CONFLICT = "SKILL_INSTALL_CONFLICT"
UNSAFE_BUNDLE_PATH = "UNSAFE_BUNDLE_PATH"
# Raised when a guarded replacement no longer matches what the client previewed:
# either the uploaded bundle or the installed bundle changed in between. The
# client must re-preview rather than retry with a fresh hash.
SKILL_SOURCE_CHANGED = "SKILL_SOURCE_CHANGED"
SKILL_BUNDLE_INVALID = "SKILL_BUNDLE_INVALID"
SKILL_SETUP_REQUIRED = "SKILL_SETUP_REQUIRED"
SKILL_SETUP_FAILED = "SKILL_SETUP_FAILED"
SKILL_RUNTIME_STALE = "SKILL_RUNTIME_STALE"
SKILL_PORTABILITY_UNSUPPORTED = "SKILL_PORTABILITY_UNSUPPORTED"

# Readiness and capability lookup errors.
SKILL_NOT_READY = "SKILL_NOT_READY"
CAPABILITY_NOT_FOUND = "CAPABILITY_NOT_FOUND"
MISSING_SECRET = "MISSING_SECRET"

# Permission errors.
PERMISSION_REQUIRED = "PERMISSION_REQUIRED"
PERMISSION_DENIED = "PERMISSION_DENIED"

# Execution errors.
COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
REMOTE_API_ERROR = "REMOTE_API_ERROR"
RUNTIME_ERROR = "RUNTIME_ERROR"


class SkillRuntimeError(Exception):
    """A normalized, structured skill-runtime failure.

    Carries a stable ``code`` (one of the module-level constants above), a
    human-readable ``message``, and an optional ``repair`` hint dict. Callers
    that assemble a full failure envelope (``ok``/``skill``/``capability``/
    ``duration_ms``/``audit_id`` plus this error) read these
    three fields; this class does not build that envelope itself.
    """

    def __init__(self, code: str, message: str, repair: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.repair = repair

    def __str__(self) -> str:
        return self.message


# Internal codes that must not reach a client under their own name:
# UNSAFE_BUNDLE_PATH and SKILL_INSTALL_INVALID both describe bundle problems the
# published contract names SKILL_BUNDLE_INVALID. This table lives here rather
# than in the route layer because the async installation receipt also stores a
# code that a client later reads, and both boundaries must agree.
_PUBLISHED_ALIASES = {
    SKILL_INSTALL_INVALID: SKILL_BUNDLE_INVALID,
    UNSAFE_BUNDLE_PATH: SKILL_BUNDLE_INVALID,
}


def publish_code(code: str) -> str:
    """Map an internal error code onto the published contract."""
    return _PUBLISHED_ALIASES.get(code, code)


def error_payload(code: str, message: str, repair: dict | None = None) -> dict:
    """Build the ``error`` sub-object of the plan's failure envelope.

    Does not build the full execution envelope (``ok``/``skill``/
    ``capability``/``duration_ms``/``audit_id``) -- that assembly belongs to
    the caller.
    """
    return {"code": code, "message": message, "repair": repair}
