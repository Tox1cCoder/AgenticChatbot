"""
Shared utility functions for AI agents.
"""

from typing import Any, Dict, List, Tuple
import json
import asyncio
import logging

logger = logging.getLogger(__name__)


def coerce_response_text(content: Any) -> str:
    """
    Convert various content types to plain text string.
    
    Handles strings, lists, dictionaries, and other types to ensure
    consistent text output from agent responses.
    
    Args:
        content: The content to convert (str, list, dict, or other)
        
    Returns:
        Plain text string representation of the content
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    text_parts.append(str(item))
            else:
                text_parts.append(str(item))
        return " ".join(text_parts)
    elif isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return str(content)
    else:
        return str(content) if content is not None else ""


def format_tool_result(value: Any) -> str:
    """
    Format tool execution results as JSON or string.
    
    Args:
        value: The tool result to format
        
    Returns:
        Formatted string representation of the tool result
    """
    if value is None:
        return ""
    
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def make_json_safe(value: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable format.
    
    Handles Pydantic models, objects with dict() methods, nested structures,
    and other non-serializable types.
    
    Args:
        value: The value to make JSON-safe
        
    Returns:
        JSON-serializable version of the value
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    
    if hasattr(value, "model_dump"):
        return make_json_safe(value.model_dump())
    
    if hasattr(value, "dict") and callable(value.dict):
        return make_json_safe(value.dict())
    
    return str(value)


def extract_agent_execution_info(agent_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract execution information from agent executor response.
    
    Parses the response from create_agent invocation and extracts:
    - Final response text
    - List of tools used
    - Tool artifacts (calls with arguments and outputs)
    - Reasoning steps if available
    
    Args:
        agent_response: The response dictionary from agent.invoke()
        
    Returns:
        Dictionary with keys: response_text, tools_used, tool_artifacts, reasoning_steps
    """
    result = {
        "response_text": "",
        "tools_used": [],
        "tool_artifacts": [],
        "reasoning_steps": []
    }
    
    # Extract messages from response
    messages = agent_response.get("messages", [])
    if not messages:
        return result
    
    # Find the final AI message
    final_message = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            final_message = msg
            break
    
    if not final_message:
        return result
    
    # Extract response text
    result["response_text"] = coerce_response_text(final_message.content)
    
    # Extract tools used and artifacts from all messages
    tools_used_set = set()
    for msg in messages:
        # Check for tool calls in AI messages
        if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "tool_calls"):
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                if tool_name:
                    tools_used_set.add(tool_name)
                    
                    # Find corresponding tool result
                    tool_id = tool_call.get("id", "")
                    tool_result = None
                    for result_msg in messages:
                        if hasattr(result_msg, "type") and result_msg.type == "tool":
                            if hasattr(result_msg, "tool_call_id") and result_msg.tool_call_id == tool_id:
                                tool_result = result_msg.content
                                break
                    
                    result["tool_artifacts"].append({
                        "tool": tool_name,
                        "args": make_json_safe(tool_call.get("args", {})),
                        "output": format_tool_result(tool_result) if tool_result else None
                    })
    
    result["tools_used"] = list(tools_used_set)
    
    # Extract reasoning steps (messages leading to final response)
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "ai":
            content = coerce_response_text(msg.content)
            if content and content != result["response_text"]:
                result["reasoning_steps"].append(content)
    
    return result


async def execute_tools_concurrently(
    tool_calls: List[Dict[str, Any]], 
    tools: List[Any]
) -> List[Tuple[str, Dict[str, Any], Any, bool]]:
    """
    Execute multiple tool calls concurrently using asyncio.
    
    This enables parallel tool execution when the LLM returns multiple tool calls,
    leveraging Gemini's native parallel function calling capability.
    
    Args:
        tool_calls: List of tool call dictionaries from ai_message.tool_calls
        tools: List of available BaseTool instances
        
    Returns:
        List of tuples: (tool_name, tool_args, result_or_error, success_flag)
    """
    if not tool_calls:
        return []
    
    # Create a mapping of tool names to tool instances
    tool_map = {tool.name: tool for tool in tools}
    
    async def execute_single_tool(tool_call: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Any, bool]:
        """Execute a single tool call with error handling."""
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        
        try:
            tool = tool_map.get(tool_name)
            if not tool:
                return (tool_name, tool_args, f"Tool '{tool_name}' not found", False)
            
            # Execute tool (handle both sync and async tools)
            if asyncio.iscoroutinefunction(tool.ainvoke):
                result = await tool.ainvoke(tool_args)
            else:
                # Run sync tool in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tool.invoke, tool_args)
            
            return (tool_name, tool_args, result, True)
        
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return (tool_name, tool_args, f"Error: {str(e)}", False)
    
    # Execute all tool calls concurrently
    results = await asyncio.gather(
        *[execute_single_tool(tc) for tc in tool_calls],
        return_exceptions=False
    )
    
    return results
