"""
Skills-related exception classes.
"""

from fastapi import status
from .http import CustomHTTPException


class SkillNotFoundError(CustomHTTPException):
    """Raised when a requested skill does not exist."""

    def __init__(self, name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{name}' not found",
            error_code="SKILL_NOT_FOUND",
        )
        self.skill_name = name
