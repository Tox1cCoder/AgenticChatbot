import logging
from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.interfaces.user_service_interface import IUserService
from app.schemas.user import UserCreate, UserUpdate, UserRead
from app.schemas.responses import ApiResponse

router = APIRouter(prefix="/users", tags=["users"])


@router.post(
    "/", response_model=ApiResponse[UserRead], status_code=status.HTTP_201_CREATED
)
@inject
async def create_user(
    user_data: UserCreate,
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
) -> ApiResponse[UserRead]:
    """Create a new user"""
    result = user_service.create_user(user_data)
    return ApiResponse(data=result, message="User created successfully")


@router.get("/{user_id}", response_model=ApiResponse[UserRead])
@inject
async def get_user(
    user_id: UUID,
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
) -> ApiResponse[UserRead]:
    """Get user by ID"""
    result = user_service.get_by_id(user_id)
    return ApiResponse(data=result, message="User retrieved successfully")


@router.get("/", response_model=ApiResponse[List[UserRead]])
@inject
async def get_users(
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
    page: int = 1,
    limit: int = 100,
) -> ApiResponse[List[UserRead]]:
    """Get all users"""
    skip = (page - 1) * limit
    result = user_service.get_all(skip=skip, limit=limit)
    return ApiResponse(data=result, message="Users retrieved successfully")
