from abc import ABC, abstractmethod

from app.schemas.task_plan import PlanningRuntimeRequest, PlanningRuntimeResult


class IPlanningRuntimeService(ABC):
    """Service-layer port for planning-agent interactions."""

    @abstractmethod
    async def generate_plan(self, request: PlanningRuntimeRequest) -> PlanningRuntimeResult:
        """Generate a todo plan from a user request."""
        pass

    @abstractmethod
    async def modify_plan(self, request: PlanningRuntimeRequest) -> PlanningRuntimeResult:
        """Modify an existing todo plan from a user request."""
        pass
