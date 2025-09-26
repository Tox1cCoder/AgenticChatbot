"""
Success Response Schema
"""

from pydantic import BaseModel, Field, ConfigDict
from app.utils.case_conversion import to_camel_case as to_camel


class SuccessResponse(BaseModel):
    """Success Response Schema"""

    success: bool = Field(True, description="Indicates the request was successful")
    message: str = Field(
        "Operation completed successfully", description="Success message"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
