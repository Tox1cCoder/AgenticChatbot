from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.ai.mcp_tool_catalog import ToolDescriptor  # noqa: E402
from app.ai.tool_search_scoring import rank_tool_candidates  # noqa: E402


@dataclass(frozen=True)
class GoldenCase:
    query: str
    expected_top: str
    expected_confidence: str = "high"


CASES = [
    GoldenCase("run shell command", "start_process"),
    GoldenCase("run python script", "start_process"),
    GoldenCase("search file contents", "start_search"),
    GoldenCase("write file", "write_file"),
    GoldenCase("apply_patch", "edit_block"),
    GoldenCase("get server config", "get_config"),
    GoldenCase("open browser url", "start_process"),
    GoldenCase("open link", "start_process"),
    GoldenCase("open application or URL on desktop", "start_process"),
    GoldenCase("browser open", "start_process"),
    GoldenCase("open youtube in browser", "start_process"),
    GoldenCase("open url in browser play youtube video", "start_process"),
    GoldenCase("extract content from known URL", "extract_url"),
    GoldenCase("search the web for current news", "web_search"),
]


def _candidate_tools() -> list[ToolDescriptor]:
    return [
        ToolDescriptor(
            "start_search",
            "desktop_commander",
            "Search files by path and pattern.",
            ["path", "pattern"],
            ["path", "pattern"],
            "fp1",
        ),
        ToolDescriptor(
            "get_config",
            "desktop_commander",
            "Get server configuration and blocked shell commands.",
            [],
            [],
            "fp2",
        ),
        ToolDescriptor(
            "interact_with_process",
            "desktop_commander",
            "Send input to a running process.",
            ["pid", "input"],
            ["pid", "input"],
            "fp3",
        ),
        ToolDescriptor(
            "start_process",
            "desktop_commander",
            "Start a terminal process.",
            ["command", "timeout_ms", "shell"],
            ["command"],
            "fp4",
        ),
        ToolDescriptor(
            "edit_block",
            "desktop_commander",
            "Apply surgical edits to files.",
            ["file_path", "old_string", "new_string"],
            ["file_path"],
            "fp5",
        ),
        ToolDescriptor(
            "write_file",
            "desktop_commander",
            "Write or append to file contents.",
            ["path", "content", "mode"],
            ["path", "content"],
            "fp6",
        ),
        ToolDescriptor(
            "extract_url",
            "web_content",
            "Extract page content from known URLs.",
            ["urls"],
            ["urls"],
            "fp7",
        ),
        ToolDescriptor(
            "web_search",
            "web_content",
            "Search the web for current news and sources.",
            ["query"],
            ["query"],
            "fp8",
        ),
    ]


def main() -> int:
    tools = _candidate_tools()
    failures = []
    for case in CASES:
        ranked = rank_tool_candidates(query=case.query, candidates=tools)
        top = ranked[0].tool.tool_name if ranked else None
        confidence = ranked[0].confidence if ranked else "none"
        print(
            f"{case.query}: top={top} confidence={confidence} "
            f"expected={case.expected_top}/{case.expected_confidence}"
        )
        if top != case.expected_top or confidence != case.expected_confidence:
            failures.append(
                (
                    case.query,
                    f"{top}/{confidence}",
                    f"{case.expected_top}/{case.expected_confidence}",
                )
            )
    if failures:
        print("FAILURES:")
        for query, top, expected in failures:
            print(f"- {query}: got {top}, expected {expected}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
