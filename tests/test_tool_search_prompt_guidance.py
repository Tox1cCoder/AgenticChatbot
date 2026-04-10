"""Tests for generic prompt invariants (Phase 4).

Verifies that TOOL_EXPLORATION_SUFFIX:
- Contains no hard-coded MCP server names
- Contains the phrase 'tool_search' (forces discovery)
- Explicitly states prompt text is not inventory
"""

from __future__ import annotations

from app.ai.prompts import TOOL_EXPLORATION_SUFFIX


# ---------------------------------------------------------------------------
# Generic prompt invariants
# ---------------------------------------------------------------------------


def test_tool_exploration_suffix_contains_no_hard_coded_server_names():
    """TOOL_EXPLORATION_SUFFIX must not hard-code specific MCP server names as examples."""
    forbidden = ["tavily", "desktop-commander", "desktop_commander"]
    lower_suffix = TOOL_EXPLORATION_SUFFIX.lower()
    for name in forbidden:
        assert name not in lower_suffix, (
            f"TOOL_EXPLORATION_SUFFIX hard-codes server name '{name}'. "
            "Server names must not appear in shared prompt text."
        )


def test_tool_exploration_suffix_mentions_tool_search():
    """TOOL_EXPLORATION_SUFFIX must reference 'tool_search' to force discovery."""
    assert "tool_search" in TOOL_EXPLORATION_SUFFIX, (
        "TOOL_EXPLORATION_SUFFIX must mention 'tool_search' so agents know to use it."
    )


def test_tool_exploration_suffix_says_prompt_is_not_inventory():
    """TOOL_EXPLORATION_SUFFIX must explicitly state that the prompt is not inventory."""
    lower = TOOL_EXPLORATION_SUFFIX.lower()
    assert "not a tool inventory" in lower or "not an inventory" in lower or "not inventory" in lower, (
        "TOOL_EXPLORATION_SUFFIX must state that prompt text is not a tool inventory."
    )


def test_tool_exploration_suffix_mentions_inventory_mode():
    """TOOL_EXPLORATION_SUFFIX must mention how to list all servers (inventory mode)."""
    assert "tool_search()" in TOOL_EXPLORATION_SUFFIX, (
        "TOOL_EXPLORATION_SUFFIX must show the no-argument form of tool_search() "
        "for listing all available servers."
    )
