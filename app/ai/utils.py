"""
Shared utility functions for AI agents.
"""

from typing import Any
import json


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
    
    Attempts to serialize the value as JSON for structured data,
    falls back to string representation for simple values.
    
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
