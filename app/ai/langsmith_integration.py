"""
LangSmith Integration for Multi-Agent System Tracking

This module provides comprehensive logging and tracking capabilities using LangSmith
to monitor agent execution, performance metrics, and workflow orchestration.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from contextlib import contextmanager
import json
import functools

# LangSmith imports
try:
    from langsmith import Client, RunTree
    from langsmith.run_helpers import traceable

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

    # Create dummy decorators if LangSmith is not available
    def traceable(func):
        return func

    class Client:
        def __init__(self, *args, **kwargs):
            pass

    class RunTree:
        def __init__(self, *args, **kwargs):
            pass


from ..schemas import AgentMessage, AgentResponse, WorkflowConfig

# Configure logging
logger = logging.getLogger(__name__)


class LangSmithTracker:
    """
    LangSmith integration for tracking multi-agent system execution.

    Provides comprehensive tracking of agent selection, execution times,
    tool usage, and workflow performance with detailed tracing.
    """

    def __init__(
        self,
        project_name: str = "multi-agent-chatbot",
        api_key: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize LangSmith tracker.

        Args:
            project_name: LangSmith project name
            api_key: LangSmith API key (defaults to environment variable)
            enabled: Whether tracking is enabled
        """
        self.project_name = project_name
        self.enabled = enabled and LANGSMITH_AVAILABLE

        if self.enabled:
            # Initialize LangSmith client
            self.client = Client(
                api_key=api_key or os.getenv("LANGSMITH_API_KEY"),
                api_url=os.getenv(
                    "LANGSMITH_API_URL", "https://api.smith.langchain.com"
                ),
            )

            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = project_name
            if api_key:
                os.environ["LANGSMITH_API_KEY"] = api_key

            logger.info(f"LangSmith tracking enabled for project: {project_name}")
        else:
            self.client = None
            if not LANGSMITH_AVAILABLE:
                logger.warning(
                    "LangSmith not available - install with 'pip install langsmith'"
                )
            else:
                logger.info("LangSmith tracking disabled")

        # Track active runs
        self.active_runs: Dict[str, RunTree] = {}
        self.session_stats = {
            "total_requests": 0,
            "agent_usage": {},
            "tool_usage": {},
            "error_count": 0,
            "average_response_time": 0.0,
        }

    @traceable
    def start_workflow_run(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Start tracking a new workflow run.

        Args:
            message: Input message
            conversation_id: Optional conversation ID
            user_id: Optional user ID

        Returns:
            str: Run ID for tracking
        """
        if not self.enabled:
            return "disabled"

        try:
            run_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Create root run for the workflow
            run = RunTree(
                name="multi_agent_workflow",
                project_name=self.project_name,
                run_type="chain",
                inputs={
                    "message": message.content,
                    "message_type": (
                        message.message_type.value
                        if hasattr(message.message_type, "value")
                        else str(message.message_type)
                    ),
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                },
                tags=["workflow", "multi-agent", "chatbot"],
                extra={
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "message_metadata": message.metadata,
                },
            )

            self.active_runs[run_id] = run
            self.session_stats["total_requests"] += 1

            logger.debug(f"Started LangSmith workflow run: {run_id}")
            return run_id

        except Exception as e:
            logger.error(f"Failed to start LangSmith workflow run: {str(e)}")
            return "error"

    @traceable
    def track_agent_selection(
        self,
        run_id: str,
        selected_agent: str,
        confidence: float,
        routing_scores: Dict[str, float],
        routing_time: float,
    ) -> None:
        """
        Track agent selection and routing decision.

        Args:
            run_id: Workflow run ID
            selected_agent: Selected agent identifier
            confidence: Selection confidence score
            routing_scores: All agent scores
            routing_time: Time taken for routing
        """
        if not self.enabled or run_id not in self.active_runs:
            return

        try:
            parent_run = self.active_runs[run_id]

            # Create child run for routing
            routing_run = parent_run.create_child(
                name="agent_routing",
                run_type="chain",
                inputs={
                    "routing_scores": routing_scores,
                    "routing_time_ms": routing_time * 1000,
                },
                outputs={"selected_agent": selected_agent, "confidence": confidence},
                tags=["routing", "agent-selection"],
                extra={
                    "routing_algorithm": "content_based",
                    "available_agents": list(routing_scores.keys()),
                },
            )

            routing_run.end()

            # Update session stats
            self.session_stats["agent_usage"][selected_agent] = (
                self.session_stats["agent_usage"].get(selected_agent, 0) + 1
            )

            logger.debug(
                f"Tracked agent selection: {selected_agent} (confidence: {confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to track agent selection: {str(e)}")

    @traceable
    def track_agent_execution(
        self,
        run_id: str,
        agent_id: str,
        execution_time: float,
        success: bool,
        response: Optional[AgentResponse] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Track agent execution performance.

        Args:
            run_id: Workflow run ID
            agent_id: Agent identifier
            execution_time: Execution time in seconds
            success: Whether execution was successful
            response: Agent response if successful
            error: Error message if failed
        """
        if not self.enabled or run_id not in self.active_runs:
            return

        try:
            parent_run = self.active_runs[run_id]

            # Prepare execution data
            inputs = {"agent_id": agent_id, "execution_time_ms": execution_time * 1000}

            outputs = {}
            if success and response:
                outputs.update(
                    {
                        "response_content": (
                            response.content[:200] + "..."
                            if len(response.content) > 200
                            else response.content
                        ),
                        "response_type": (
                            response.response_type.value
                            if hasattr(response.response_type, "value")
                            else str(response.response_type)
                        ),
                        "confidence": response.confidence,
                        "agent_id": response.agent_id,
                    }
                )

                if hasattr(response, "metadata") and response.metadata:
                    outputs["metadata"] = response.metadata

            if error:
                outputs["error"] = error

            # Create child run for agent execution
            agent_run = parent_run.create_child(
                name=f"agent_execution_{agent_id}",
                run_type="llm" if agent_id in ["chat", "rag"] else "tool",
                inputs=inputs,
                outputs=outputs,
                tags=["agent-execution", agent_id, "success" if success else "error"],
                extra={
                    "agent_type": agent_id,
                    "execution_success": success,
                    "performance_metrics": {
                        "execution_time": execution_time,
                        "tokens_used": (
                            response.metadata.get("tokens_used", 0)
                            if response
                            and hasattr(response, "metadata")
                            and response.metadata
                            else 0
                        ),
                    },
                },
            )

            if not success:
                agent_run.end(error=error)
                self.session_stats["error_count"] += 1
            else:
                agent_run.end()

            logger.debug(
                f"Tracked agent execution: {agent_id} ({'success' if success else 'error'})"
            )

        except Exception as e:
            logger.error(f"Failed to track agent execution: {str(e)}")

    @traceable
    def track_tool_execution(
        self,
        run_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        execution_time: float,
        success: bool,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Track tool execution performance.

        Args:
            run_id: Workflow run ID
            tool_name: Tool name
            parameters: Tool parameters
            execution_time: Execution time in seconds
            success: Whether execution was successful
            result: Tool result if successful
            error: Error message if failed
        """
        if not self.enabled or run_id not in self.active_runs:
            return

        try:
            parent_run = self.active_runs[run_id]

            # Create child run for tool execution
            tool_run = parent_run.create_child(
                name=f"tool_execution_{tool_name}",
                run_type="tool",
                inputs={
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "execution_time_ms": execution_time * 1000,
                },
                outputs=(
                    {
                        "result": (
                            str(result)[:500] + "..."
                            if result and len(str(result)) > 500
                            else result
                        ),
                        "success": success,
                        "error": error,
                    }
                    if success
                    else {"error": error, "success": False}
                ),
                tags=["tool-execution", tool_name, "success" if success else "error"],
                extra={
                    "tool_type": tool_name,
                    "parameter_count": len(parameters),
                    "execution_success": success,
                },
            )

            if not success:
                tool_run.end(error=error)
            else:
                tool_run.end()

            # Update session stats
            self.session_stats["tool_usage"][tool_name] = (
                self.session_stats["tool_usage"].get(tool_name, 0) + 1
            )

            logger.debug(
                f"Tracked tool execution: {tool_name} ({'success' if success else 'error'})"
            )

        except Exception as e:
            logger.error(f"Failed to track tool execution: {str(e)}")

    @traceable
    def end_workflow_run(
        self,
        run_id: str,
        success: bool,
        final_response: Optional[AgentResponse] = None,
        error: Optional[str] = None,
        total_time: Optional[float] = None,
    ) -> None:
        """
        End a workflow run with final results.

        Args:
            run_id: Workflow run ID
            success: Whether workflow was successful
            final_response: Final response if successful
            error: Error message if failed
            total_time: Total execution time in seconds
        """
        if not self.enabled or run_id not in self.active_runs:
            return

        try:
            run = self.active_runs[run_id]

            # Prepare final outputs
            outputs = {}
            if success and final_response:
                outputs.update(
                    {
                        "final_response": final_response.content,
                        "response_type": (
                            final_response.response_type.value
                            if hasattr(final_response.response_type, "value")
                            else str(final_response.response_type)
                        ),
                        "agent_id": final_response.agent_id,
                        "confidence": final_response.confidence,
                    }
                )

                if hasattr(final_response, "metadata") and final_response.metadata:
                    outputs["metadata"] = final_response.metadata

            if error:
                outputs["error"] = error

            if total_time:
                outputs["total_execution_time_ms"] = total_time * 1000

                # Update average response time
                current_avg = self.session_stats["average_response_time"]
                total_requests = self.session_stats["total_requests"]
                self.session_stats["average_response_time"] = (
                    current_avg * (total_requests - 1) + total_time
                ) / total_requests

            # Add session statistics
            outputs["session_stats"] = self.session_stats.copy()

            # End the run
            if success:
                run.end(outputs=outputs)
            else:
                run.end(outputs=outputs, error=error)

            # Clean up
            del self.active_runs[run_id]

            logger.debug(
                f"Ended LangSmith workflow run: {run_id} ({'success' if success else 'error'})"
            )

        except Exception as e:
            logger.error(f"Failed to end LangSmith workflow run: {str(e)}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        return {
            **self.session_stats,
            "langsmith_enabled": self.enabled,
            "project_name": self.project_name,
            "active_runs": len(self.active_runs),
        }

    def reset_session_stats(self) -> None:
        """Reset session statistics."""
        self.session_stats = {
            "total_requests": 0,
            "agent_usage": {},
            "tool_usage": {},
            "error_count": 0,
            "average_response_time": 0.0,
        }
        logger.info("LangSmith session statistics reset")


# Global tracker instance
_global_tracker: Optional[LangSmithTracker] = None


def initialize_langsmith_tracking(
    project_name: str = "multi-agent-chatbot",
    api_key: Optional[str] = None,
    enabled: bool = True,
) -> LangSmithTracker:
    """
    Initialize global LangSmith tracking.

    Args:
        project_name: LangSmith project name
        api_key: LangSmith API key
        enabled: Whether tracking is enabled

    Returns:
        LangSmithTracker: Initialized tracker instance
    """
    global _global_tracker
    _global_tracker = LangSmithTracker(
        project_name=project_name, api_key=api_key, enabled=enabled
    )
    return _global_tracker


def get_tracker() -> Optional[LangSmithTracker]:
    """Get the global LangSmith tracker instance."""
    return _global_tracker


def langsmith_trace(name: str, run_type: str = "chain", tags: List[str] = None):
    """
    Decorator for tracing functions with LangSmith.

    Args:
        name: Run name
        run_type: Type of run (chain, llm, tool, etc.)
        tags: Optional tags for the run
    """

    def decorator(func):
        if not LANGSMITH_AVAILABLE:
            return func

        @traceable(run_type=run_type, name=name, tags=tags or [])
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @traceable(run_type=run_type, name=name, tags=tags or [])
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


@contextmanager
def langsmith_context(name: str, inputs: Dict[str, Any] = None, tags: List[str] = None):
    """
    Context manager for manual LangSmith tracing.

    Args:
        name: Run name
        inputs: Input data
        tags: Optional tags
    """
    if not LANGSMITH_AVAILABLE or not _global_tracker or not _global_tracker.enabled:
        yield None
        return

    try:
        run = RunTree(
            name=name,
            project_name=_global_tracker.project_name,
            inputs=inputs or {},
            tags=tags or [],
        )
        yield run
        run.end()
    except Exception as e:
        logger.error(f"LangSmith context error: {str(e)}")
        yield None


# Export key components
__all__ = [
    "LangSmithTracker",
    "initialize_langsmith_tracking",
    "get_tracker",
    "langsmith_trace",
    "langsmith_context",
    "LANGSMITH_AVAILABLE",
]
