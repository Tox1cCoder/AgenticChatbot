from uuid import UUID
from fastapi import APIRouter

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.user_service_interface import IUserService
from app.schemas.user import UserRead
from app.schemas.responses import ApiResponse

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}", response_model=ApiResponse[UserRead])
@AppAutoInjector.auto_inject()
async def get_user(
    user_id: UUID,
    user_service: IUserService,
) -> ApiResponse[UserRead]:
    """Get user by ID"""
    result = user_service.get_by_id(user_id)
    return ApiResponse(success=True, message="User retrieved successfully", data=result)
