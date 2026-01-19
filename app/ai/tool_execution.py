from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .utils import extract_content_from_result, normalize_tool_call


def extract_images_from_tool_result(result_text: str) -> List[Dict[str, str]]:
    if not result_text:
        return []

    try:
        parsed = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(parsed, dict):
        return []

    images = parsed.get("images")
    if not isinstance(images, list):
        return []

    extracted: List[Dict[str, str]] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        url = image.get("url")
        if not url:
            continue
        extracted.append(
            {
                "url": str(url),
                "description": str(image.get("description") or ""),
            }
        )
    return extracted


def build_tool_artifact(
    *,
    tool_call_id: Optional[str],
    tool_name: str,
    tool_args: Any,
    output_text: Optional[str],
    error: Optional[str],
    max_output_chars: int = 1000,
) -> Dict[str, Any]:
    artifact: Dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "tool": tool_name,
        "args": tool_args,
        "output": None,
        "error": error,
    }

    if output_text is not None:
        artifact["output"] = (
            output_text[:max_output_chars]
            if len(output_text) > max_output_chars
            else output_text
        )

    return artifact


async def ensure_agent_tool_map(agent: Any) -> Dict[str, Any]:
    if not agent:
        return {}

    tools = getattr(agent, "tools", None) or []
    if not tools:
        if hasattr(agent, "_init_mcp"):
            await agent._init_mcp()
        elif hasattr(agent, "_init_tools"):
            await agent._init_tools()
        tools = getattr(agent, "tools", None) or []

    return {t.name: t for t in tools if getattr(t, "name", None)}


async def invoke_tool(tool: Any, tool_args: Any) -> Any:
    if getattr(tool, "coroutine", None):
        return await tool.ainvoke(tool_args)

    invoke = getattr(tool, "invoke", None)
    if callable(invoke):
        return invoke(tool_args)

    ainvoke = getattr(tool, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(tool_args)

    if callable(tool):
        return tool(tool_args)

    raise TypeError("Tool has no invoke/ainvoke and is not callable")


async def execute_tool_calls(
    *,
    tool_calls: List[Any],
    tool_map: Dict[str, Any],
    capture_images: bool = True,
    artifact_max_output_chars: int = 1000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    outputs: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    images: List[Dict[str, str]] = []

    for raw_tool_call in tool_calls:
        tool_call = normalize_tool_call(raw_tool_call)
        tool_name = tool_call.get("name")
        tool_id = tool_call.get("id")
        tool_args = tool_call.get("args", {})

        if not tool_name:
            error_msg = "Error: Tool name missing"
            outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name or "unknown",
                    "content": error_msg,
                }
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name or "unknown",
                    tool_args=tool_args,
                    output_text=None,
                    error=error_msg,
                    max_output_chars=artifact_max_output_chars,
                )
            )
            continue

        tool = tool_map.get(tool_name)
        if not tool:
            error_msg = f"Error: Tool {tool_name} not found"
            outputs.append(
                {"tool_call_id": tool_id, "name": tool_name, "content": error_msg}
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=None,
                    error=error_msg,
                    max_output_chars=artifact_max_output_chars,
                )
            )
            continue

        try:
            result = await invoke_tool(tool, tool_args)
            result = extract_content_from_result(result)
            result_text = str(result)

            outputs.append(
                {"tool_call_id": tool_id, "name": tool_name, "content": result_text}
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=result_text,
                    error=None,
                    max_output_chars=artifact_max_output_chars,
                )
            )
            if capture_images:
                images.extend(extract_images_from_tool_result(result_text))
        except Exception as exc:
            error_msg = f"Error: {exc}"
            outputs.append(
                {"tool_call_id": tool_id, "name": tool_name, "content": error_msg}
            )
            artifacts.append(
                build_tool_artifact(
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    output_text=None,
                    error=str(exc),
                    max_output_chars=artifact_max_output_chars,
                )
            )

    return outputs, artifacts, images

