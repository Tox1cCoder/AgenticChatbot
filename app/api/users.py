from uuid import UUID

from fastapi import APIRouter, Depends, Path

from app.core.auth import get_current_user, require_user_ownership
from app.core.dependency_injection import AppAutoInjector
from app.interfaces.user_service_interface import IUserService
from app.models.user import User
from app.schemas.responses import ApiResponse
from app.schemas.user import UserRead

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}", response_model=ApiResponse[UserRead])
@AppAutoInjector.auto_inject()
async def get_user(
    user_service: IUserService,
    user_id: UUID = Path(...),  # noqa: B008
    current_user: User = Depends(get_current_user),  # noqa: B008
) -> ApiResponse[UserRead]:
    """Get user by ID"""
    require_user_ownership(user_id, current_user.id)
    result = user_service.get_by_id(current_user.id)
    return ApiResponse(success=True, message="User retrieved successfully", data=result)
