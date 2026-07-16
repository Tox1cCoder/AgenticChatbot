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
_EXTERNAL_OPEN_ACTION_TERMS = {"open", "launch", "play", "show"}
_EXTERNAL_OPEN_TARGET_TERMS = {
    "app",
    "application",
    "browser",
    "link",
    "media",
    "url",
    "urls",
    "video",
    "webpage",
    "website",
}
_CONTENT_READ_ACTION_TERMS = {
    "analyze",
    "analyse",
    "extract",
    "fetch",
    "inspect",
    "read",
    "summarize",
    "summarise",
}
_WEB_RESOURCE_TERMS = {
    "article",
    "content",
    "page",
    "source",
    "url",
    "urls",
    "webpage",
    "website",
}
_WEB_SEARCH_ACTION_TERMS = {"search", "find", "lookup"}
_WEB_SEARCH_CONTEXT_TERMS = {"web", "internet", "online", "news", "current", "recent"}
_WEB_MAP_TERMS = {"map", "sitemap", "site", "structure", "pages", "urls", "discover"}
_WEB_CRAWL_TERMS = {"crawl", "site", "website", "docs", "documentation", "section", "pages"}
_URL_ARGUMENT_TERMS = {"link", "uri", "url", "urls"}


def infer_query_intent(query: str | None) -> QueryIntent:
    raw_query = str(query or "").strip()
    tokens = set(tokenize_text(raw_query))
    capabilities: set[str] = set()
    action_verbs: set[str] = set()

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

    external_open_actions = tokens & _EXTERNAL_OPEN_ACTION_TERMS
    external_open_targets = tokens & _EXTERNAL_OPEN_TARGET_TERMS
    content_read_actions = tokens & _CONTENT_READ_ACTION_TERMS
    web_resource_targets = tokens & _WEB_RESOURCE_TERMS

    if external_open_actions and external_open_targets:
        capabilities.add("external_open")
    if content_read_actions and web_resource_targets:
        capabilities.add("web_extract")
    if tokens & _WEB_SEARCH_ACTION_TERMS and tokens & _WEB_SEARCH_CONTEXT_TERMS:
        capabilities.add("web_search")
    if tokens & _WEB_MAP_TERMS and tokens & {
        "map",
        "sitemap",
        "structure",
        "discover",
        "urls",
    }:
        capabilities.add("web_map")
    if tokens & _WEB_CRAWL_TERMS and tokens & {
        "crawl",
        "site",
        "website",
        "docs",
        "documentation",
    }:
        capabilities.add("web_crawl")

    action_verbs.update(
        tokens
        & (
            _SHELL_TERMS
            | _FILE_SEARCH_TERMS
            | _FILE_EDIT_TERMS
            | _FILE_WRITE_TERMS
            | _EXTERNAL_OPEN_ACTION_TERMS
            | _CONTENT_READ_ACTION_TERMS
        )
    )
    target_terms = external_open_targets | web_resource_targets
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
    url_arg_tokens = (arg_tokens | required_arg_tokens) & _URL_ARGUMENT_TERMS
    direct_open_signal = bool(
        ({"open", "launch"} & name_tokens)
        and (url_arg_tokens or {"browser", "link", "url"} & description_tokens)
    )
    if direct_open_signal:
        capabilities.add("external_open")

    if "extract" in name_tokens and url_arg_tokens:
        capabilities.add("web_extract")
    if (
        "search" in name_tokens
        and "query" in (arg_tokens | required_arg_tokens)
        and {"internet", "news", "online", "source", "web"} & description_tokens
    ):
        capabilities.add("web_search")
    if "map" in name_tokens and url_arg_tokens:
        capabilities.add("web_map")
    if "crawl" in name_tokens and url_arg_tokens:
        capabilities.add("web_crawl")

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
    if "external_open" in capabilities:
        return "Open an external URL or resource in a local application."
    if "web_search" in capabilities:
        return "Search the web for current facts, news, and source discovery."
    if "web_extract" in capabilities:
        return "Extract content from one or more known web page URLs."
    if "web_map" in capabilities:
        return "Discover URLs and structure for a website."
    if "web_crawl" in capabilities:
        return "Crawl a bounded site section and return page content."
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
