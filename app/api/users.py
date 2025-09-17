from typing import List
from uuid import UUID
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status

from dependency_injector.wiring import Provide, inject

from app.core.container import Container
from app.interfaces.user_service_interface import IUserService
from app.schemas.user import UserCreate, UserUpdate, UserRead

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
@inject
async def create_user(
    user_data: UserCreate,
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
) -> UserRead:
    """Create a new user"""
    return user_service.create_user(user_data)


@router.get("/{user_id}", response_model=UserRead)
@inject
async def get_user(
    user_id: UUID,
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
) -> UserRead:
    """Get user by ID"""
    return user_service.get_by_id(user_id)


@router.get("/", response_model=List[UserRead])
@inject
async def get_users(
    user_service: Annotated[IUserService, Depends(Provide[Container.user_service])],
    page: int = 1,
    limit: int = 100,
) -> List[UserRead]:
    """Get all users"""
    skip = (page - 1) * limit
    return user_service.get_all(skip=skip, limit=limit)
