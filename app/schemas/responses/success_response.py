"""
Success Response Schema
"""

from pydantic import BaseModel, Field, ConfigDict


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class SuccessResponse(BaseModel):
    """Success Response Schema"""

    success: bool = Field(True, description="Indicates the request was successful")
    message: str = Field(
        "Operation completed successfully", description="Success message"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
