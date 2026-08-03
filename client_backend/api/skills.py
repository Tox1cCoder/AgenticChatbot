"""
Local skills management endpoints with server-compatible response envelopes.
"""

from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from client_backend.api.common import make_api_response
from client_backend.api.skill_errors import response_for_exception
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.schemas.skill_installation import SkillInstallationRequest
from client_backend.schemas.skills import (
    SkillInstallPreviewRequest,
    SkillInstallRequest,
    SkillSecretSetRequest,
    SkillSetupRequest,
    SkillUninstallRequest,
)
from client_backend.services.local_skills_registry import get_skills_registry
from client_backend.services.runtime_bridge import get_runtime_bridge
from client_backend.services.skill_runtime.install import SkillBundleInstaller
from client_backend.services.skill_runtime.manager import SkillRuntimeManager
from client_backend.services.skill_runtime.operations import (
    SkillOperationError,
    get_skill_installation_service,
)
from client_backend.services.skill_runtime.secrets import SkillSecretStore
from client_backend.services.skill_runtime.uploads import (
    SkillUploadError,
    get_skill_upload_service,
)
from shared.skills.errors import SKILL_INSTALL_CONFLICT, SKILL_INSTALL_INVALID, SkillRuntimeError

router = APIRouter(prefix="/skills", tags=["skills"])

# Status codes for install/uninstall failures that aren't the generic 400.
# Every other installation error is a client-supplied-bad-input case, so 400
# is the correct default rather than enumerating each one here.
_INSTALL_ERROR_STATUS_OVERRIDES = {SKILL_INSTALL_CONFLICT: 409}
_UNINSTALL_ERROR_STATUS_OVERRIDES = {SKILL_INSTALL_INVALID: 404}


def get_skill_installer() -> SkillBundleInstaller:
    """Factory for the bundle installer, overridable in tests."""
    return SkillBundleInstaller()


def get_secret_store() -> SkillSecretStore:
    """Factory for the skill secret store, overridable in tests."""
    return SkillSecretStore()


def _skill_summary(skill) -> dict:
    readiness = SkillRuntimeManager().evaluate_readiness(skill)
    return {
        "name": skill.name,
        "description": skill.description,
        "enabled": skill.enabled,
        "folderPath": str(skill.path.parent),
        "sourceHash": getattr(skill, "source_hash", None),
        "commandCapable": readiness.status == "ready",
        "runtimeStatus": readiness.status,
        "setupStatus": readiness.setup_status,
    }


def _skill_detail(skill) -> dict:
    payload = _skill_summary(skill)
    payload["content"] = skill.content
    return payload


async def _refresh_runtime_bridge_catalogs_if_connected() -> None:
    bridge = get_runtime_bridge()
    if not bridge.is_connected() or not bridge.get_registered_device_id():
        return
    await bridge.refresh_catalogs()


@router.get("")
async def list_skills(
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """List local skills in the server's `ApiResponse[SkillListResponse]` shape."""
    registry = get_skills_registry()
    await registry.initialize()
    skills = [_skill_summary(skill) for skill in registry.get_all_skills()]
    return make_api_response(
        success=True,
        message="Skills retrieved",
        data={
            "skills": skills,
            "totalCount": len(skills),
            "enabledCount": sum(1 for skill in skills if skill["enabled"]),
        },
    )


@router.post("/install")
async def install_skill(
    payload: SkillInstallRequest,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Install a local skill bundle directory into the profile skill root."""
    installer = get_skill_installer()
    try:
        result = await installer.install(
            payload.source_path,
            expected_source_hash=payload.expected_source_hash,
            approve_setup=payload.approve_setup,
        )
    except SkillRuntimeError as exc:
        status_code = _INSTALL_ERROR_STATUS_OVERRIDES.get(exc.code, 400)
        raise HTTPException(
            status_code=status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    # SkillBundleInstaller.install() already refreshes the registry and the
    # runtime bridge catalogs (if connected) as its last step, so this route
    # does not repeat that refresh.
    return make_api_response(
        success=True,
        message=f"Skill bundle '{result['name']}' installed",
        data=result,
    )


@router.post("/install/preview")
async def preview_skill_install(
    payload: SkillInstallPreviewRequest,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Inspect a one-skill bundle without copying or executing it."""
    try:
        result = await get_skill_installer().preview(payload.source_path)
    except SkillRuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return make_api_response(
        success=True,
        message=f"Skill bundle '{result['name']}' previewed",
        data=result,
    )


# --- ZIP upload and asynchronous installation -------------------------------
#
# Declared before the dynamic ``/{name}`` routes below. None of these shapes
# currently collide with one, but keeping the static prefixes first means adding
# a future ``POST /{name}`` cannot silently capture ``/uploads``.


async def _session_user(session: LocalSessionPayload) -> str:
    """Resolve the profile that owns this request and recover its state once.

    Recovery runs on the first authenticated skill request for a profile rather
    than at process startup, because at startup there is no user session to scope
    uploads, receipts, or locks to.
    """
    user_id = str(session.user_id)
    await get_skill_upload_service().ensure_recovered(user_id)
    await get_skill_installation_service().ensure_recovered(user_id)
    return user_id


@router.post("/uploads", status_code=201)
async def stage_skill_upload(
    file: UploadFile = File(...),
    session: LocalSessionPayload = Depends(require_local_session),
):
    """Accept one ZIP, validate and extract it, and return its preview.

    Nothing in the archive is executed here; installation is a separate,
    explicitly confirmed step.
    """
    user_id = await _session_user(session)
    try:
        record = await get_skill_upload_service().stage(
            user_id=user_id,
            filename=file.filename or "",
            stream=file,
        )
    except SkillUploadError as exc:
        return response_for_exception(exc)
    finally:
        # Starlette spills large uploads to a temporary file; closing is what
        # removes it, and it must happen whether or not staging succeeded.
        await file.close()

    return make_api_response(
        success=True,
        message="Skill archive staged",
        data=record.to_api(),
        status_code=201,
    )


@router.delete("/uploads/{upload_id}")
async def cancel_skill_upload(
    upload_id: str,
    session: LocalSessionPayload = Depends(require_local_session),
):
    """Discard a staged upload and its bytes."""
    user_id = await _session_user(session)
    try:
        get_skill_upload_service().delete(user_id, upload_id)
    except SkillUploadError as exc:
        return response_for_exception(exc)
    return make_api_response(
        success=True,
        message="Skill upload cancelled",
        data={"uploadId": upload_id, "state": "cancelled"},
    )


@router.post("/uploads/{upload_id}/install", status_code=202)
async def start_skill_installation(
    upload_id: str,
    payload: SkillInstallationRequest,
    session: LocalSessionPayload = Depends(require_local_session),
):
    """Begin installing a staged upload and return its polling receipt."""
    user_id = await _session_user(session)
    try:
        operation = await get_skill_installation_service().start(user_id, upload_id, payload)
    except (SkillUploadError, SkillOperationError) as exc:
        return response_for_exception(exc)

    data = operation.to_api()
    data["statusUrl"] = f"/skills/installations/{quote(operation.operation_id, safe='')}"
    return make_api_response(
        success=True,
        message="Skill installation started",
        data=data,
        status_code=202,
    )


@router.get("/installations/{operation_id}")
async def get_skill_installation(
    operation_id: str,
    session: LocalSessionPayload = Depends(require_local_session),
):
    """Report one installation's state.

    A failed installation is still a successful *retrieval*, so this stays HTTP
    200 with ``state: "failed"``; only an unknown or foreign id is a 404.
    """
    user_id = await _session_user(session)
    try:
        operation = get_skill_installation_service().get_owned(user_id, operation_id)
    except SkillOperationError as exc:
        return response_for_exception(exc)
    return make_api_response(
        success=True,
        message=_operation_message(operation.state),
        data=operation.to_api(),
    )


@router.delete("/installations/{operation_id}")
async def cancel_skill_installation(
    operation_id: str,
    session: LocalSessionPayload = Depends(require_local_session),
):
    """Cancel an installation that has not yet crossed the commit boundary."""
    user_id = await _session_user(session)
    try:
        operation = await get_skill_installation_service().cancel(user_id, operation_id)
    except SkillOperationError as exc:
        return response_for_exception(exc)
    return make_api_response(
        success=True,
        message="Skill installation cancelled",
        data=operation.to_api(),
    )


def _operation_message(state: str) -> str:
    return {
        "pending": "Skill installation queued",
        "running": "Skill installation running",
        "succeeded": "Skill installed",
        "failed": "Skill installation failed",
        "cancelled": "Skill installation cancelled",
    }.get(state, "Skill installation state retrieved")


@router.post("/{name}/setup")
async def setup_skill(
    name: str,
    payload: SkillSetupRequest,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Prepare or rebuild one skill's device-local Python runtime."""
    try:
        result = await get_skill_installer().setup(
            name,
            expected_source_hash=payload.expected_source_hash,
            approve_setup=payload.approve_setup,
        )
    except SkillRuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    return make_api_response(
        success=True,
        message=f"Skill '{name}' runtime prepared",
        data=result,
    )


@router.post("/uninstall")
async def uninstall_skill(
    payload: SkillUninstallRequest,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Uninstall a previously installed local skill bundle."""
    installer = get_skill_installer()
    try:
        result = await installer.uninstall(payload.name)
    except SkillRuntimeError as exc:
        status_code = _UNINSTALL_ERROR_STATUS_OVERRIDES.get(exc.code, 400)
        raise HTTPException(
            status_code=status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    # SkillBundleInstaller.uninstall() already refreshes the registry and the
    # runtime bridge catalogs (if connected) as its last step, so this route
    # does not repeat that refresh.
    return make_api_response(
        success=True,
        message=f"Skill bundle '{payload.name}' uninstalled",
        data=result,
    )


@router.get("/installed")
async def list_installed_skills(
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """List installed local skill bundles (device-local view; may include local paths)."""
    installer = get_skill_installer()
    installed = installer.list_installed()
    return make_api_response(
        success=True,
        message="Installed skill bundles retrieved",
        data={"installed": installed, "totalCount": len(installed)},
    )


async def _require_known_skill(name: str):
    registry = get_skills_registry()
    await registry.initialize()
    skill = registry.get_skill(name)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return skill


@router.post("/{name}/secrets")
async def set_skill_secret(
    name: str,
    payload: SkillSecretSetRequest,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Bind a secret to exactly one local skill. Never echo the value."""
    await _require_known_skill(name)
    store = get_secret_store()
    try:
        store.set_for_skill(name, payload.name, payload.value)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return make_api_response(
        success=True,
        message=f"Secret '{payload.name}' saved",
        data={"skill": name, "name": payload.name, "configured": True},
    )


@router.get("/{name}/secrets")
async def get_skill_secrets(
    name: str,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """List configured binding names for one skill, never their values."""
    await _require_known_skill(name)
    store = get_secret_store()
    secrets = [
        {"name": secret_name, "configured": True} for secret_name in store.list_for_skill(name)
    ]
    return make_api_response(
        success=True,
        message=f"Secrets for skill '{name}' retrieved",
        data={"secrets": secrets},
    )


@router.delete("/{name}/secrets/{secret_name}")
async def delete_skill_secret(
    name: str,
    secret_name: str,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Delete one binding from one local skill."""
    await _require_known_skill(name)
    try:
        removed = get_secret_store().delete_for_skill(name, secret_name)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return make_api_response(
        success=True,
        message=f"Secret '{secret_name}' removed from skill '{name}'",
        data={"skill": name, "name": secret_name, "removed": removed},
    )


@router.get("/{name}")
async def get_skill(
    name: str,
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Get detail for one local skill."""
    registry = get_skills_registry()
    await registry.initialize()
    skill = registry.get_skill(name)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found")

    return make_api_response(
        success=True,
        message=f"Skill '{name}' retrieved",
        data=_skill_detail(skill),
    )


@router.patch("/{name}/toggle")
async def toggle_skill(
    name: str,
    enabled: bool = Query(...),
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Enable or disable a local skill."""
    registry = get_skills_registry()
    await registry.initialize()
    updated = registry.set_skill_enabled(name, enabled)
    if not updated:
        raise HTTPException(status_code=404, detail="Skill not found")

    await _refresh_runtime_bridge_catalogs_if_connected()

    state = "enabled" if enabled else "disabled"
    message = f"Skill '{name}' {state}"
    return make_api_response(
        success=True,
        message=message,
        data={"message": message},
    )


@router.post("/reload")
async def reload_skills(
    _session: LocalSessionPayload = Depends(require_local_session),
):
    """Rescan configured local skill roots."""
    registry = get_skills_registry()
    await registry.refresh()
    await _refresh_runtime_bridge_catalogs_if_connected()
    message = "Skills reloaded"
    return make_api_response(
        success=True,
        message=message,
        data={"message": message},
    )
