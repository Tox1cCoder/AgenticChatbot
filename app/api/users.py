from uuid import UUID

from fastapi import APIRouter, Path

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.user_service_interface import IUserService
from app.schemas.responses import ApiResponse
from app.schemas.user import UserRead

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}", response_model=ApiResponse[UserRead])
@AppAutoInjector.auto_inject()
async def get_user(
    user_service: IUserService,
    user_id: UUID = Path(...),
) -> ApiResponse[UserRead]:
    """Get user by ID"""
    result = user_service.get_by_id(user_id)
    return ApiResponse(success=True, message="User retrieved successfully", data=result)
