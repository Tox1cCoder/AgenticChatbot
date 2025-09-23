from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, status

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.interfaces.user_service_interface import IUserService
from app.schemas.user import UserRead
from app.schemas.responses import ApiResponse

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}", response_model=ApiResponse[UserRead])
@inject
async def get_user(
    user_id: UUID,
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
) -> ApiResponse[UserRead]:
    """Get user by ID"""
    result = user_service.get_by_id(user_id)
    return ApiResponse(
        success=True,
        message="User retrieved successfully",
        data=result
    )
