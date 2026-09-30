from functools import cache
from uuid import UUID

from fastapi import HTTPException, status

from app.core.exceptions import AuthenticationException
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.core.security.token_version import with_token_version
from app.interfaces.auth_service_interface import IAuthService
from app.interfaces.user_service_interface import IUserService
from app.schemas.responses.token_response import LoginRequest
from app.services.jwt_service import JwtService


@cache
def _dummy_password_hash() -> str:
    """Compared against when the email is unknown, so a miss costs one bcrypt
    check like a hit does. Without it the response time told a caller which
    emails are registered."""
    return hash_password("dummy-password-for-timing-parity")


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )


class AuthService(IAuthService):
    def __init__(self, user_service: IUserService, jwt_service: JwtService) -> None:
        self.user_service = user_service
        self.jwt_service = jwt_service

    def authenticate_user(self, login_data: LoginRequest) -> dict:
        user = self.user_service.get_by_email_with_password(login_data.email)
        stored_hash = user.password_hash if user else _dummy_password_hash()
        password_ok = verify_password(login_data.password, stored_hash)
        # A soft-deleted account keeps its row (and email) but must not log in.
        if user is None or user.deleted_at is not None or not password_ok:
            raise _invalid_credentials()

        token_data = with_token_version({"sub": str(user.id)}, user.token_version)
        access_token = create_access_token(token_data, self.jwt_service)
        refresh_token = create_refresh_token(token_data, self.jwt_service)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user_id": str(user.id),
        }

    def refresh_access_token(self, user_id: UUID) -> str:
        """A new access token stamped with the user's current token version.

        The refresh dependency has already refused a revoked refresh token; this
        re-reads the row so the new token never carries a stale version.
        """
        state = self.user_service.get_token_state(user_id)
        if state is None or state.deleted:
            raise AuthenticationException(
                detail="Authenticated user no longer exists",
                error_code="AUTHENTICATED_USER_NOT_FOUND",
            )
        token_data = with_token_version({"sub": str(user_id)}, state.version)
        return create_access_token(token_data, self.jwt_service)
