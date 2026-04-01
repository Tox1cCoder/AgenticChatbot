"""REST API endpoints for the Skills system — mirrors app/api/mcp.py."""

from fastapi import APIRouter, Depends, Query

from app.core.auth import get_current_user
from app.core.dependency_injection import AppAutoInjector
from app.schemas.responses import ApiResponse
from app.schemas.skills import (
    SkillDetail,
    SkillListResponse,
    SkillOperationResponse,
)
from app.services.skills_service import SkillsService

router = APIRouter(
    prefix="/skills",
    tags=["skills"],
    dependencies=[Depends(get_current_user)],
)


@router.get("")
@AppAutoInjector.auto_inject()
async def list_skills(
    skills_service: SkillsService,
) -> ApiResponse[SkillListResponse]:
    """List all skills with enabled state."""
    result = await skills_service.list_skills()
    return ApiResponse(
        success=True,
        message="Skills retrieved",
        data=SkillListResponse(**result),
    )


@router.get("/{name}")
@AppAutoInjector.auto_inject()
async def get_skill(
    name: str,
    skills_service: SkillsService,
) -> ApiResponse[SkillDetail]:
    """Get full detail (incl. Markdown content) for one skill."""
    result = await skills_service.get_skill(name)
    return ApiResponse(
        success=True,
        message=f"Skill '{name}' retrieved",
        data=SkillDetail(**result),
    )


@router.patch("/{name}/toggle")
@AppAutoInjector.auto_inject()
async def toggle_skill(
    name: str,
    skills_service: SkillsService,
    enabled: bool = Query(...),
) -> ApiResponse[SkillOperationResponse]:
    """Enable or disable a skill."""
    result = await skills_service.toggle_skill(name, enabled)
    return ApiResponse(
        success=True,
        message=result["message"],
        data=SkillOperationResponse(**result),
    )


@router.post("/reload")
@AppAutoInjector.auto_inject()
async def reload_skills(
    skills_service: SkillsService,
) -> ApiResponse[SkillOperationResponse]:
    """Trigger hot-reload from disk."""
    result = await skills_service.reload_skills()
    return ApiResponse(
        success=True,
        message=result["message"],
        data=SkillOperationResponse(**result),
    )
