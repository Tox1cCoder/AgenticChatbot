"""
Task Plans API endpoints.
"""

from typing import Any, List
from uuid import UUID
from fastapi import APIRouter, status

from app.core.dependency_injection import AppAutoInjector
from app.interfaces.task_plan_service_interface import ITaskPlanService
from app.schemas.task_plan import (
    TaskPlanRead,
    TaskPlanUpdate,
    TaskPlanGenerateRequest,
    TaskPlanManualCreateRequest,
    PlanningStatusResponse,
)
from app.schemas.responses import ApiResponse

router = APIRouter(tags=["task-plans"])


@router.post(
    "/conversations/{conversation_id}/task-plans",
    response_model=ApiResponse[List[TaskPlanRead]],
    status_code=status.HTTP_201_CREATED,
)
@AppAutoInjector.auto_inject()
async def create_task_plan(
    conversation_id: UUID,
    request_data: TaskPlanGenerateRequest,
    task_plan_service: ITaskPlanService,
    user_id: UUID,
) -> ApiResponse[List[TaskPlanRead]]:
    """
    Create a task plan from a user's request using the planning agent.
    The planning agent will analyze the request and generate a structured plan
    with ordered tasks.
    """
    result = await task_plan_service.create_task_plan(
        conversation_id=conversation_id,
        user_message=request_data.user_message,
        user_id=user_id,
    )
    return ApiResponse(
        success=True,
        message=f"Task plan created successfully with {len(result)} tasks",
        data=result,
    )


@router.post(
    "/conversations/{conversation_id}/task-plans/manual",
    response_model=ApiResponse[List[TaskPlanRead]],
    status_code=status.HTTP_201_CREATED,
)
@AppAutoInjector.auto_inject()
async def create_task_plan_manual(
    conversation_id: UUID,
    request_data: TaskPlanManualCreateRequest,
    task_plan_service: ITaskPlanService,
    user_id: UUID,
) -> ApiResponse[List[TaskPlanRead]]:
    """
    Create a task plan from a manual list of task descriptions.
    Tasks will be created in order.
    """
    result = task_plan_service.create_task_plan_from_list(
        conversation_id=conversation_id,
        task_descriptions=request_data.task_descriptions,
        user_id=user_id,
    )
    return ApiResponse(
        success=True,
        message=f"Task plan created successfully with {len(result)} tasks",
        data=result,
    )


@router.get(
    "/conversations/{conversation_id}/task-plans",
    response_model=ApiResponse[List[TaskPlanRead]],
)
@AppAutoInjector.auto_inject()
async def get_conversation_task_plans(
    conversation_id: UUID,
    task_plan_service: ITaskPlanService,
    user_id: UUID,
    include_completed: bool = False,
) -> ApiResponse[List[TaskPlanRead]]:
    """
    Get all task plans for a conversation.
    By default, completed and skipped tasks are excluded.
    Set include_completed=true to include all tasks.
    """
    result = task_plan_service.get_conversation_tasks(
        conversation_id=conversation_id,
        user_id=user_id,
        include_completed=include_completed,
    )
    return ApiResponse(
        success=True,
        message="Task plans retrieved successfully",
        data=result,
    )


@router.get(
    "/task-plans/{task_id}",
    response_model=ApiResponse[TaskPlanRead],
)
@AppAutoInjector.auto_inject()
async def get_task_plan(
    task_id: UUID,
    task_plan_service: ITaskPlanService,
    user_id: UUID,
) -> ApiResponse[TaskPlanRead]:
    """Get a specific task plan by ID."""
    result = task_plan_service.get_by_id(task_id=task_id, user_id=user_id)
    return ApiResponse(
        success=True,
        message="Task plan retrieved successfully",
        data=result,
    )


@router.patch(
    "/task-plans/{task_id}",
    response_model=ApiResponse[TaskPlanRead],
)
@AppAutoInjector.auto_inject()
async def update_task_plan(
    task_id: UUID,
    task_update_data: TaskPlanUpdate,
    task_plan_service: ITaskPlanService,
    user_id: UUID,
) -> ApiResponse[TaskPlanRead]:
    """
    Update a task plan.
    Can update description, status, and metadata.
    """
    result = task_plan_service.update_task(
        task_id=task_id,
        user_id=user_id,
        task_update_data=task_update_data,
    )
    return ApiResponse(
        success=True,
        message="Task plan updated successfully",
        data=result,
    )


@router.post(
    "/task-plans/{task_id}/complete",
    response_model=ApiResponse[TaskPlanRead],
)
@AppAutoInjector.auto_inject()
async def complete_task_plan(
    task_id: UUID,
    task_plan_service: ITaskPlanService,
    user_id: UUID,
) -> ApiResponse[TaskPlanRead]:
    """Mark a task plan as completed."""
    result = task_plan_service.mark_task_completed(task_id=task_id, user_id=user_id)
    return ApiResponse(
        success=True,
        message="Task plan marked as completed",
        data=result,
    )


@router.delete(
    "/task-plans/{task_id}",
    response_model=ApiResponse[Any],
)
@AppAutoInjector.auto_inject()
async def delete_task_plan(
    task_id: UUID,
    task_plan_service: ITaskPlanService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Delete a task plan."""
    task_plan_service.delete_task(task_id=task_id, user_id=user_id)
    return ApiResponse(
        success=True,
        message="Task plan deleted successfully",
        data=None,
    )


@router.get(
    "/conversations/{conversation_id}/planning-status",
    response_model=ApiResponse[PlanningStatusResponse],
)
@AppAutoInjector.auto_inject()
async def get_planning_status(
    conversation_id: UUID,
    task_plan_service: ITaskPlanService,
    user_id: UUID,
) -> ApiResponse[PlanningStatusResponse]:
    """
    Get planning mode status and progress for a conversation.
    Returns task counts, progress percentage, and next task information.
    """
    result = task_plan_service.get_planning_status(
        conversation_id=conversation_id,
        user_id=user_id,
    )
    return ApiResponse(
        success=True,
        message="Planning status retrieved successfully",
        data=result,
    )
