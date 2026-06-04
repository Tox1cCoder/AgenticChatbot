from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .text_normalization import split_identifier_tokens, tokenize_text


@dataclass(frozen=True)
class QueryIntent:
    raw_query: str
    tokens: set[str]
    capabilities: set[str] = field(default_factory=set)
    action_verbs: set[str] = field(default_factory=set)
    target_terms: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ToolCapabilityProfile:
    tool_name: str
    server_name: str
    name_tokens: set[str]
    arg_tokens: set[str]
    required_arg_tokens: set[str]
    description_tokens: set[str]
    capabilities: set[str]
    purpose: str


_SHELL_TERMS = {"run", "execute", "shell", "command", "terminal", "process", "python", "script"}
_FILE_SEARCH_TERMS = {"search", "find", "grep", "pattern", "contents", "content"}
_FILE_EDIT_TERMS = {"edit", "patch", "apply", "replace", "modify", "surgical"}
_FILE_WRITE_TERMS = {"write", "create", "append", "save"}
_CONFIG_TERMS = {"config", "configuration", "settings", "blockedcommands"}


def infer_query_intent(query: str | None) -> QueryIntent:
    raw_query = str(query or "").strip()
    tokens = set(tokenize_text(raw_query))
    capabilities: set[str] = set()
    action_verbs: set[str] = set()
    target_terms: set[str] = set(tokens)

    if tokens & _SHELL_TERMS and (
        tokens & {"run", "execute", "command", "shell", "python", "script"}
    ):
        capabilities.add("shell_exec")
    if tokens & _FILE_SEARCH_TERMS and tokens & {"file", "files", "contents", "content", "pattern"}:
        capabilities.add("file_search")
    if tokens & _FILE_EDIT_TERMS:
        capabilities.add("file_edit")
    if tokens & _FILE_WRITE_TERMS and tokens & {"file", "files", "content", "contents"}:
        capabilities.add("file_write")
    if tokens & _CONFIG_TERMS:
        capabilities.add("config_read")

    action_verbs.update(
        tokens & (_SHELL_TERMS | _FILE_SEARCH_TERMS | _FILE_EDIT_TERMS | _FILE_WRITE_TERMS)
    )
    return QueryIntent(
        raw_query=raw_query,
        tokens=tokens,
        capabilities=capabilities,
        action_verbs=action_verbs,
        target_terms=target_terms,
    )


def infer_tool_profile(
    *,
    tool_name: str,
    server_name: str = "",
    description: str = "",
    arg_names: list[str] | None = None,
    required_arg_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolCapabilityProfile:
    arg_names = arg_names or []
    required_arg_names = required_arg_names or []
    name_tokens = set(split_identifier_tokens(tool_name))
    arg_tokens = set(token for arg in arg_names for token in split_identifier_tokens(arg))
    required_arg_tokens = set(
        token for arg in required_arg_names for token in split_identifier_tokens(arg)
    )
    description_tokens = set(tokenize_text(description))
    all_tokens = name_tokens | arg_tokens | required_arg_tokens | description_tokens
    capabilities: set[str] = set()

    if (
        {"start", "process"} <= name_tokens
        or "command" in required_arg_tokens
        or ("shell" in arg_tokens and "command" in arg_tokens)
    ):
        capabilities.add("shell_exec")
    if "search" in name_tokens and ({"path", "pattern"} & required_arg_tokens):
        capabilities.add("file_search")
    if "edit" in name_tokens or "patch" in name_tokens or {"old", "new", "string"} <= arg_tokens:
        capabilities.add("file_edit")
    if ("write" in name_tokens and "content" in required_arg_tokens) or (
        "path" in required_arg_tokens and "content" in required_arg_tokens
    ):
        capabilities.add("file_write")
    if "config" in name_tokens and not ({"set", "write"} & name_tokens):
        capabilities.add("config_read")
    if "set" in name_tokens and "config" in name_tokens:
        capabilities.add("config_write")
    if "interact" in name_tokens and "process" in name_tokens:
        capabilities.add("process_interaction")

    purpose = _compact_purpose(tool_name, capabilities, description, all_tokens)
    return ToolCapabilityProfile(
        tool_name=tool_name,
        server_name=server_name,
        name_tokens=name_tokens,
        arg_tokens=arg_tokens,
        required_arg_tokens=required_arg_tokens,
        description_tokens=description_tokens,
        capabilities=capabilities,
        purpose=purpose,
    )


def _compact_purpose(
    tool_name: str,
    capabilities: set[str],
    description: str,
    all_tokens: set[str],
) -> str:
    if "shell_exec" in capabilities:
        return "Start a shell command or local process."
    if "file_search" in capabilities:
        return "Search file contents by path and pattern."
    if "file_edit" in capabilities:
        return "Apply focused edits to existing file text."
    if "file_write" in capabilities:
        return "Write or append file contents."
    if "config_read" in capabilities:
        return "Read server configuration."
    if "process_interaction" in capabilities:
        return "Send input to an already running process."
    first_line = " ".join(str(description or "").strip().split())
    if first_line:
        return first_line[:120]
    return f"Use {tool_name}."
