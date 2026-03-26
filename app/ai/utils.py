"""
Shared utility functions for AI agents.
"""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage


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
                item_type = item.get("type", "")
                # Skip thinking/reasoning blocks - they should not be in text output
                if item_type == "thinking":
                    continue
                # Extract text from text blocks
                if "text" in item:
                    text_parts.append(item["text"])
                elif item_type == "text":
                    text_parts.append(item.get("text", ""))
                # Skip other known non-text types (tool_call_chunk, etc.)
                elif item_type in ("tool_call_chunk", "tool_use", "tool_result"):
                    continue
                else:
                    # Only stringify unknown items if they don't look like structured content
                    if not item_type:
                        text_parts.append(str(item))
            elif hasattr(item, "text"):
                # Handle objects with text attribute
                text_parts.append(item.text)
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        return "".join(text_parts)
    elif isinstance(content, dict):
        content_type = content.get("type", "")
        # Skip thinking blocks
        if content_type == "thinking":
            return ""
        if "text" in content:
            return content["text"]
        if content_type == "text":
            return content.get("text", "")
        # Skip other non-text types
        if content_type in ("tool_call_chunk", "tool_use", "tool_result"):
            return ""
        return str(content)
    else:
        return str(content) if content is not None else ""


def extract_openai_reasoning_summary(content: Any) -> str | None:
    """
    Extract OpenAI reasoning *summary* text from LangChain content blocks.

    When `reasoning={"summary": ...}` is set, ChatOpenAI may return content like:
      [
        {"type": "reasoning", "summary": [{"type": "text", "text": "..."}]},
        {"type": "text", "text": "final answer"}
      ]
    """
    if not content:
        return None

    blocks: list[Any]
    if isinstance(content, list):
        blocks = content
    elif isinstance(content, dict):
        blocks = [content]
    else:
        return None

    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "").strip().lower() != "reasoning":
            continue

        summary = block.get("summary")
        if isinstance(summary, str):
            parts.append(summary)
            continue

        if isinstance(summary, dict):
            text = summary.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
            continue

        if isinstance(summary, list):
            for item in summary:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                elif isinstance(item, str) and item.strip():
                    parts.append(item)
            continue

    cleaned = "\n".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
    return cleaned or None


def _get_nested(data: Any, path: Sequence[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_openai_reasoning_tokens(message: Any) -> int | None:
    """
    Best-effort extraction of OpenAI reasoning token count from LangChain metadata.
    """
    candidates: list[dict[str, Any]] = []
    for attr in ("usage_metadata", "response_metadata", "additional_kwargs"):
        meta = getattr(message, attr, None)
        if isinstance(meta, dict):
            candidates.append(meta)

    paths: list[Sequence[str]] = [
        ("usage", "output_tokens_details", "reasoning_tokens"),
        ("usage", "completion_tokens_details", "reasoning_tokens"),
        ("token_usage", "completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    ]

    for meta in candidates:
        for path in paths:
            value = _get_nested(meta, path)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)

    return None


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


def _decode_tool_args(args: Any) -> Any:
    """
    Best-effort decode for tool-call arguments returned as JSON strings.

    Some model/tool-calling adapters return argument payloads as serialized JSON
    text. Decoding them here ensures tools receive structured arguments and
    Unicode escapes such as ``\\u00e1`` are converted back to normal text.
    """
    if not isinstance(args, str):
        return args

    stripped = args.strip()
    if not stripped:
        return {}

    looks_like_json = (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
        or (stripped.startswith('"') and stripped.endswith('"'))
    )
    if not looks_like_json:
        return args

    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return args


def normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    """
    Normalize tool call data to a consistent dictionary format.

    Args:
        tool_call: Tool call data in dict or object format. Can be:
            - dict with keys: name/action/tool, args/tool_input/arguments, id/tool_call_id
            - object with attributes: name, args, id, tool_call_id
            - other representations with similar structures

    Returns:
        Normalized dict with keys:
            - name (str): The tool/function name
            - args (dict): The tool arguments/parameters
            - id (str): Unique identifier for the tool call
            - tool_call_id (str): Alternative ID field (same as id)
    """
    if isinstance(tool_call, dict):
        # Extract name from various possible keys
        name = (
            tool_call.get("name") or tool_call.get("action") or tool_call.get("tool") or "unknown"
        )

        # Extract args from various possible keys
        args = (
            tool_call.get("args") or tool_call.get("tool_input") or tool_call.get("arguments") or {}
        )

        # Extract ID from various possible keys
        tool_id = (
            tool_call.get("id") or tool_call.get("tool_call_id") or tool_call.get("task_id") or ""
        )
    else:
        # Handle object with attributes
        name = (
            getattr(tool_call, "name", None)
            or getattr(tool_call, "action", None)
            or getattr(tool_call, "tool", None)
            or "unknown"
        )

        args = (
            getattr(tool_call, "args", None)
            or getattr(tool_call, "tool_input", None)
            or getattr(tool_call, "arguments", None)
            or {}
        )

        tool_id = (
            getattr(tool_call, "id", None)
            or getattr(tool_call, "tool_call_id", None)
            or getattr(tool_call, "task_id", None)
            or ""
        )

    args = _decode_tool_args(args)

    return {
        "name": name,
        "args": args,
        "id": tool_id,
        "tool_call_id": tool_id,
    }


def find_pending_tool_call_message(
    messages: Sequence[Any],
) -> tuple[int, AIMessage] | None:
    """
    Find the most recent AIMessage that still has pending tool calls.

    The approval node may append rejection ToolMessages after the rewritten
    AIMessage. In that case the pending tool-call message is no longer the last
    entry, so consumers must scan backward until they reach the most recent
    AIMessage boundary.
    """
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None):
                return idx, message
            return None

    return None


def extract_rejection_reason(decision: Any) -> str | None:
    """Extract a free-form rejection reason from a HITL decision payload."""
    if not isinstance(decision, dict):
        return None

    decision_args = decision.get("args")
    if isinstance(decision_args, str):
        trimmed = decision_args.strip()
        if trimmed:
            return trimmed
    elif isinstance(decision_args, dict):
        for key in ("message", "reason", "feedback", "content", "text"):
            value = decision_args.get(key)
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    return trimmed

    for key in ("message", "reason", "feedback"):
        value = decision.get(key)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return trimmed

    return None


def build_rejection_tool_message(
    tool_call: Any,
    decision: dict[str, Any] | None = None,
    *,
    source: str,
) -> str:
    """
    Serialize rejection context so the agent can inspect the real decision data.
    """
    normalized_tool_call = normalize_tool_call(tool_call)
    payload: dict[str, Any] = {
        "status": "rejected",
        "source": source,
        "tool_name": normalized_tool_call.get("name"),
        "tool_call_id": normalized_tool_call.get("id"),
        "tool_args": make_json_safe(normalized_tool_call.get("args", {})),
    }

    if isinstance(decision, dict):
        decision_type = decision.get("type")
        if decision_type:
            payload["decision_type"] = str(decision_type)

        action = decision.get("action")
        if action:
            payload["action"] = str(action)

        decision_args = decision.get("args")
        if decision_args not in (None, "", [], {}):
            payload["decision_args"] = make_json_safe(decision_args)

    rejection_reason = extract_rejection_reason(decision)
    if rejection_reason:
        payload["reason"] = rejection_reason

    return json.dumps(payload, ensure_ascii=False, default=str)


def extract_agent_execution_info(agent_response: dict[str, Any]) -> dict[str, Any]:
    """
    Extract execution information from agent executor response.

    Parses the response from create_agent invocation and extracts:
    - Final response text
    - List of tools used
    - Tool artifacts (calls with arguments and outputs)

    Args:
        agent_response: The response dictionary from agent.invoke()

    Returns:
        Dictionary with keys: response_text, tools_used, tool_artifacts
    """
    result = {
        "response_text": "",
        "tools_used": [],
        "tool_artifacts": [],
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
                        if (
                            hasattr(result_msg, "type")
                            and result_msg.type == "tool"
                            and hasattr(result_msg, "tool_call_id")
                            and result_msg.tool_call_id == tool_id
                        ):
                            tool_result = result_msg.content
                            break

                    result["tool_artifacts"].append(
                        {
                            "tool": tool_name,
                            "args": make_json_safe(tool_call.get("args", {})),
                            "output": (format_tool_result(tool_result) if tool_result else None),
                        }
                    )

    result["tools_used"] = list(tools_used_set)

    return result


def get_error_recovery_hint(error: Exception, tool_name: str, tool_args: dict[str, Any]) -> str:
    """
    Analyze an exception and provide a recovery hint for the LLM.

    Args:
        error: The exception that occurred
        tool_name: Name of the tool that failed
        tool_args: Arguments passed to the tool

    Returns:
        A helpful hint string for the LLM on how to recover
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()

    # Missing argument errors
    if "missing" in error_msg and (
        "argument" in error_msg or "parameter" in error_msg or "required" in error_msg
    ):
        return (
            "Missing required argument: check the tool's schema and provide all required parameters"
        )

    # Type errors
    if isinstance(error, TypeError):
        if "got an unexpected keyword argument" in error_msg:
            return "Invalid argument name: verify the argument names match the tool's schema"
        if "takes" in error_msg and "positional argument" in error_msg:
            return "Wrong number of arguments: check the tool's parameter requirements"
        return "Type mismatch: ensure argument types match the tool's expected types"

    # Value errors
    if isinstance(error, ValueError):
        if "invalid" in error_msg or "format" in error_msg:
            return "Invalid argument format: check the expected format/structure for this argument"
        return "Invalid value: verify the argument values are within acceptable ranges"

    # Key errors
    if isinstance(error, KeyError):
        return "Missing key in arguments: verify all required parameters are provided with correct names"

    # Connection/Network errors
    if "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
        return "Network issue: retry the operation or use an alternative tool if available"

    # Permission/Auth errors
    if "permission" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
        return "Permission denied: this tool may require additional credentials or access rights"

    # Not found errors
    if "not found" in error_msg or isinstance(error, (FileNotFoundError, AttributeError)):
        return "Resource not found: verify the resource exists or try alternative search terms"

    # Generic fallback
    return f"Unexpected {error_type}: review the error message and adjust arguments or try a different approach"


def extract_content_from_result(result: Any) -> Any:
    """
    Extract actual content from LangChain Content objects.

    MCP tools often return results wrapped in Content format:
    [{'type': 'text', 'text': '...', 'id': '...'}]

    This function unwraps such content to extract the actual text values.

    Args:
        result: The tool result which may be wrapped in Content format

    Returns:
        Unwrapped content - either pure text or cleaned structure
    """
    if isinstance(result, list):
        cleaned = []
        for item in result:
            if isinstance(item, dict):
                if "type" in item and item.get("type") == "text" and "text" in item:
                    cleaned.append(item["text"])
                else:
                    cleaned.append(item)
            else:
                cleaned.append(item)
        if len(cleaned) == 1:
            return cleaned[0]
        return cleaned

    if (
        isinstance(result, dict)
        and "type" in result
        and result.get("type") == "text"
        and "text" in result
    ):
        return result["text"]

    return result


def apply_hitl_decisions(
    tool_calls: list[dict[str, Any]],
    human_decisions: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Apply HITL human decisions to a list of tool calls.

    Args:
        tool_calls: List of normalised tool-call dicts, each with an ``"id"`` key.
        human_decisions: A single decision dict or list of decision dicts.  Each
            decision must have ``"type"`` (approve/edit/reject) and
            ``"task_id"`` (matching a tool-call ``"id"``).

    Returns:
        approved: tool calls to execute (with modified args for 'edit' decisions)
        rejected_feedback: mapping of tool_call_id -> rejection reason message
    """
    decisions = human_decisions if isinstance(human_decisions, list) else [human_decisions]
    decision_map: dict[str, Any] = {}
    for d in decisions:
        if isinstance(d, dict):
            task_id = d.get("task_id") or d.get("tool_call_id")
            if task_id:
                decision_map[task_id] = d

    approved: list[dict[str, Any]] = []
    rejected_feedback: dict[str, str] = {}
    for tc in tool_calls:
        tool_call_id = tc.get("id")
        tool_name = tc.get("name")
        decision = decision_map.get(tool_call_id, {})
        decision_type = decision.get("type", "reject")

        if decision_type == "approve":
            approved.append(tc)
        elif decision_type == "edit":
            modified_args = decision.get("args", tc.get("args", {}))
            approved.append(
                {
                    "name": tool_name,
                    "args": modified_args,
                    "id": tool_call_id,
                }
            )
        else:  # reject / unknown
            feedback = build_rejection_tool_message(
                tc,
                decision=decision if decision else None,
                source="human_decision" if decision else "missing_decision",
            )
            if tool_call_id:
                rejected_feedback[tool_call_id] = feedback

    return approved, rejected_feedback
