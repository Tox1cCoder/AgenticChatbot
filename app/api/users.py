from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.container import get_container, DIContainer
from app.services.user_service import UserService
from app.schemas.user import UserCreate, UserUpdate, UserRead

router = APIRouter(prefix="/users", tags=["users"])


def get_user_service(db: Session = Depends(get_db)) -> UserService:
    """Dependency to get UserService instance"""
    container = get_container()
    container.set_session(db)
    return container.get("user_service")


@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate, user_service: UserService = Depends(get_user_service)
) -> UserRead:
    """Create a new user"""
    return user_service.create_user(user_data)


@router.get("/{user_id}", response_model=UserRead)
async def get_user(
    user_id: UUID, user_service: UserService = Depends(get_user_service)
) -> UserRead:
    """Get user by ID"""
    return user_service.get_user_by_id(user_id)


@router.get("/", response_model=List[UserRead])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    user_service: UserService = Depends(get_user_service),
) -> List[UserRead]:
    """Get all users"""
    return user_service.get_all_users(skip=skip, limit=limit)


# @router.get("/email/{email}", response_model=UserRead)
# async def get_user_by_email(
#     email: str, user_service: UserService = Depends(get_user_service)
# ) -> UserRead:
#     """Get user by email"""
#     user = user_service.get_user_by_email(email)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
#         )
#     return user


# @router.get("/username/{username}", response_model=UserRead)
# async def get_user_by_username(
#     username: str, user_service: UserService = Depends(get_user_service)
# ) -> UserRead:
#     """Get user by username"""
#     user = user_service.get_user_by_username(username)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
#         )
#     return user
