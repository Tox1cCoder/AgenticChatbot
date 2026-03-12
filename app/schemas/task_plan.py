"""
TaskPlan Pydantic schemas for API request/response handling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
from app.models.enums import TaskStatus
from app.utils.case_conversion import to_camel_case as to_camel


class TaskPlanCreate(BaseModel):
    """Schema for creating a new task plan."""

    conversation_id: UUID = Field(
        ..., description="Conversation ID this task plan belongs to"
    )
    task_order: int = Field(..., ge=0, description="Order/sequence of the task")
    description: str = Field(
        ..., min_length=1, description="Clear, actionable task description"
    )

    task_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the task (e.g., notes)",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class TaskPlanUpdate(BaseModel):
    """Schema for updating an existing task plan."""

    description: Optional[str] = Field(
        None, min_length=1, description="Updated task description"
    )
    status: Optional[TaskStatus] = Field(
        None,
        description="Task status: pending, in_progress, completed, skipped",
    )

    task_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Updated metadata for the task"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Timestamp when the task was completed"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class TaskPlanRead(BaseModel):
    """Schema for reading a task plan (API response)."""

    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    conversation_id: UUID = Field(
        ..., description="Conversation ID this task belongs to"
    )
    task_order: int = Field(..., description="Order/sequence of the task")
    description: str = Field(..., description="Task description")
    status: TaskStatus = Field(
        ..., description="Task status: pending, in_progress, completed, skipped"
    )

    task_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional task metadata"
    )
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = Field(
        None, description="Timestamp when the task was completed"
    )


class TaskPlanInDB(BaseModel):
    """Internal schema matching database structure."""

    model_config = ConfigDict(
        from_attributes=True, alias_generator=to_camel, populate_by_name=True
    )

    id: UUID
    conversation_id: UUID
    task_order: int
    description: str
    status: TaskStatus

    task_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class TaskPlanGenerateRequest(BaseModel):
    """Request schema for generating a task plan from user message."""

    user_message: str = Field(
        ..., min_length=1, description="The user's request to generate a plan for"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class TaskPlanManualCreateRequest(BaseModel):
    """Request schema for manually creating task plans from a list of descriptions."""

    task_descriptions: List[str] = Field(
        ...,
        min_length=1,
        description="List of task descriptions to create",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class PlanningStatusResponse(BaseModel):
    """Response schema for planning status."""

    planning_mode_enabled: bool = Field(
        ..., description="Whether planning mode is enabled for the conversation"
    )
    plan_lifecycle: Optional[str] = Field(
        None,
        description="Explicit plan lifecycle state: draft, ready, executing, paused, completed",
    )
    total_tasks: int = Field(..., ge=0, description="Total number of tasks")
    pending_tasks: int = Field(..., ge=0, description="Number of pending tasks")
    in_progress_tasks: int = Field(..., ge=0, description="Number of in-progress tasks")
    completed_tasks: int = Field(..., ge=0, description="Number of completed tasks")
    skipped_tasks: int = Field(..., ge=0, description="Number of skipped tasks")
    progress_percentage: float = Field(
        ..., ge=0, le=100, description="Percentage of tasks completed"
    )
    next_task: Optional[TaskPlanRead] = Field(
        None, description="The next task to work on (if any)"
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
