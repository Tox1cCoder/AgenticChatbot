"""
Local skills management endpoints with server-compatible response envelopes.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from client_backend.api.common import make_api_response
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.services.local_skills_registry import get_skills_registry
from client_backend.services.runtime_bridge import get_runtime_bridge

router = APIRouter(prefix="/skills", tags=["skills"])


def _skill_summary(skill) -> dict:
    return {
        "name": skill.name,
        "description": skill.description,
        "enabled": skill.enabled,
        "folderPath": str(skill.path.parent),
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
