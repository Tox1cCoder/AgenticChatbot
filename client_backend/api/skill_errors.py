"""One error vocabulary for every skill route.

Skill management is driven by a browser that has to branch on *what went wrong*:
re-preview a stale hash, cancel another upload to free quota, keep polling after a
too-late cancel. Branching on prose is not possible, so every failure leaves this
layer as a stable ``code`` plus a ``retryable`` hint, in the envelope published in
``plans/SKILL_INSTALLATION_FE_CONTRACT.md``.

Two rules hold here. Every code in :data:`SKILL_ERROR_STATUS` is documented in
that contract -- a route may not invent one, and the contract test in
``tests/test_production_readiness_contract.py`` checks both directions. And
nothing derived from the archive or the filesystem is echoed: messages are
written for the person choosing another file, never quoting a member name, a
staging path, setup output, or a secret.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from client_backend.api.common import make_api_response
from client_backend.core.logging import get_logger
from client_backend.services.skill_runtime.archive import SkillArchiveError
from client_backend.services.skill_runtime.locks import SkillLockTimeoutError
from client_backend.services.skill_runtime.operations import (
    SKILL_INSTALL_LOCKED,
    SkillOperationError,
)
from client_backend.services.skill_runtime.uploads import SkillUploadError
from shared.skills.errors import (
    SKILL_CONFIGURED_ROOT_CONFLICT,
    SKILL_INSTALL_CONFLICT,
    SKILL_INSTALL_INVALID,
    SKILL_SETUP_FAILED,
    SKILL_SETUP_REQUIRED,
    SKILL_SOURCE_CHANGED,
    UNSAFE_BUNDLE_PATH,
    SkillRuntimeError,
)

logger = get_logger(__name__)

UNAUTHENTICATED = "UNAUTHENTICATED"
SKILL_BUNDLE_INVALID = "SKILL_BUNDLE_INVALID"

# The single authoritative code-to-status table for skill routes.
SKILL_ERROR_STATUS: dict[str, int] = {
    "SKILL_ARCHIVE_INVALID": 400,
    "SKILL_ARCHIVE_PATH_UNSAFE": 400,
    SKILL_BUNDLE_INVALID: 400,
    "SKILL_UPLOAD_STATE_INVALID": 400,
    SKILL_SETUP_REQUIRED: 400,
    SKILL_SETUP_FAILED: 400,
    UNAUTHENTICATED: 401,
    "SKILL_UPLOAD_NOT_FOUND": 404,
    "SKILL_OPERATION_NOT_FOUND": 404,
    SKILL_INSTALL_CONFLICT: 409,
    SKILL_SOURCE_CHANGED: 409,
    "SKILL_UPLOAD_CONSUMED": 409,
    "SKILL_OPERATION_COMMITTED": 409,
    "SKILL_ARCHIVE_TOO_LARGE": 413,
    "SKILL_ARCHIVE_TOO_MANY_FILES": 413,
    "SKILL_UPLOAD_QUOTA_EXCEEDED": 413,
    "SKILL_ARCHIVE_TYPE_UNSUPPORTED": 415,
    SKILL_INSTALL_LOCKED: 423,
    "SKILL_STORAGE_INSUFFICIENT": 507,
}

# Codes a client may retry with the same inputs. Anything else needs the user to
# change something first.
RETRYABLE_CODES = frozenset(
    {SKILL_INSTALL_LOCKED, "SKILL_STORAGE_INSUFFICIENT", SKILL_SETUP_FAILED}
)

# Internal codes that must not reach a client under their own name. The contract
# expresses a configured-root collision through preview.existingSkill.replaceable
# and publishes only the generic conflict; UNSAFE_BUNDLE_PATH and
# SKILL_INSTALL_INVALID describe bundle problems the contract names
# SKILL_BUNDLE_INVALID.
_INTERNAL_CODE_ALIASES = {
    SKILL_CONFIGURED_ROOT_CONFLICT: SKILL_INSTALL_CONFLICT,
    SKILL_INSTALL_INVALID: SKILL_BUNDLE_INVALID,
    UNSAFE_BUNDLE_PATH: SKILL_BUNDLE_INVALID,
}


def publish_code(code: str) -> str:
    """Map an internal error code onto the published contract."""
    return _INTERNAL_CODE_ALIASES.get(code, code)


def status_for(code: str) -> int:
    """Return the documented status for ``code``, defaulting to 400."""
    return SKILL_ERROR_STATUS.get(code, status.HTTP_400_BAD_REQUEST)


def skill_error_response(
    code: str,
    message: str,
    *,
    status_code: int | None = None,
    retryable: bool | None = None,
) -> JSONResponse:
    """Build the failure envelope for one skill error."""
    published = publish_code(code)
    resolved_status = status_code if status_code is not None else status_for(published)
    resolved_retryable = retryable if retryable is not None else published in RETRYABLE_CODES
    return make_api_response(
        success=False,
        message=message,
        data=None,
        code=published,
        error={"retryable": resolved_retryable},
        status_code=resolved_status,
    )


def response_for_exception(exc: Exception) -> JSONResponse:
    """Translate any known skill-layer exception into the failure envelope.

    Status comes from the exception when it carries one -- upload and operation
    errors already decided theirs -- and from the table otherwise, so the two can
    never disagree about the same code.
    """
    if isinstance(exc, (SkillUploadError, SkillOperationError)):
        published = publish_code(exc.code)
        return skill_error_response(
            published,
            exc.message,
            status_code=exc.status_code,
            retryable=getattr(exc, "retryable", None),
        )
    if isinstance(exc, SkillArchiveError):
        return skill_error_response(exc.code, exc.message, status_code=exc.status_code)
    if isinstance(exc, SkillLockTimeoutError):
        return skill_error_response(
            SKILL_INSTALL_LOCKED,
            "Another skill operation is in progress. Try again in a moment.",
        )
    if isinstance(exc, SkillRuntimeError):
        return skill_error_response(exc.code, exc.message)
    raise exc


def is_skill_path(path: str) -> bool:
    """Report whether a request path belongs to the skill API surface."""
    return (
        path == "/skills"
        or path == "/api/skills"
        or path.startswith("/skills/")
        or path.startswith("/api/skills/")
    )


def register_skill_exception_handlers(app: FastAPI) -> None:
    """Give skill routes the coded envelope, leaving every other route alone.

    Registered globally because FastAPI resolves exception handlers per app, not
    per router, so the handlers themselves check the path and delegate to
    FastAPI's defaults for anything else. Changing the shape of unrelated
    endpoints' errors is not in scope for this feature.
    """

    @app.exception_handler(HTTPException)
    async def _skill_http_exception_handler(request: Request, exc: HTTPException):
        if not is_skill_path(request.url.path):
            return await http_exception_handler(request, exc)
        code, message = _normalize_http_detail(exc)
        return skill_error_response(code, message, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _skill_validation_exception_handler(request: Request, exc: RequestValidationError):
        if not is_skill_path(request.url.path):
            return await request_validation_exception_handler(request, exc)
        # The default 422 body enumerates every rejected field, including values.
        # A skill request body can carry an uploaded filename or a hash, so only
        # the field locations are echoed.
        fields = sorted({_field_label(error) for error in exc.errors()} - {""})
        detail = f" Check: {', '.join(fields)}." if fields else ""
        return skill_error_response(
            "SKILL_REQUEST_INVALID",
            f"The request body was not accepted.{detail}",
            # Literal 422: the repository forbids the version-specific
            # starlette symbols, whose names differ across releases.
            status_code=422,
            retryable=False,
        )


def _field_label(error: dict[str, Any]) -> str:
    location = [str(part) for part in error.get("loc", ()) if part not in {"body", "query"}]
    return ".".join(location)


def _normalize_http_detail(exc: HTTPException) -> tuple[str, str]:
    """Extract a stable code and a safe message from an HTTPException.

    Routes raise ``HTTPException`` with either a plain string or the
    ``{"code", "message"}`` shape the installer routes already used; both have to
    arrive at the client as one envelope.
    """
    detail = exc.detail
    if isinstance(detail, dict):
        code = str(detail.get("code") or _default_code_for_status(exc.status_code))
        message = str(detail.get("message") or "The request could not be completed.")
        return code, message
    message = str(detail) if detail else "The request could not be completed."
    return _default_code_for_status(exc.status_code), message


def _default_code_for_status(status_code: int) -> str:
    if status_code == status.HTTP_401_UNAUTHORIZED:
        return UNAUTHENTICATED
    if status_code == status.HTTP_404_NOT_FOUND:
        return "SKILL_NOT_FOUND"
    return "SKILL_REQUEST_INVALID"
