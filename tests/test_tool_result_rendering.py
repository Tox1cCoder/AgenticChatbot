from __future__ import annotations

from app.ai.tool_result_rendering import normalize_tool_result_for_rendering


def test_normalizes_apps_sdk_template_result():
    raw_result = {
        "content": [{"type": "text", "text": "Created presentation: Quarterly Roadmap"}],
        "structuredContent": {
            "presentation_id": "deck_123",
            "slides": [{"title": "Overview"}, {"title": "Timeline"}],
        },
        "_meta": {
            "openai/outputTemplate": "ui://canva/presentation-viewer.html",
            "title": "Quarterly Roadmap",
        },
    }

    normalized = normalize_tool_result_for_rendering(
        raw_result,
        tool_name="canva_create_presentation",
    )

    assert normalized.model_content == "Created presentation: Quarterly Roadmap"
    assert normalized.output_preview == "Created presentation: Quarterly Roadmap"
    assert normalized.render["version"] == 1
    assert normalized.render["type"] == "mcp_app"
    assert normalized.render["template_uri"] == "ui://canva/presentation-viewer.html"
    assert normalized.render["structured_content"]["presentation_id"] == "deck_123"
    assert normalized.render["ui_meta"]["openai/outputTemplate"] == (
        "ui://canva/presentation-viewer.html"
    )


def test_normalizes_snake_case_structured_content_and_resource_uri():
    raw_result = {
        "content": [{"type": "text", "text": "Dashboard ready"}],
        "structured_content": {"chart_type": "bar", "labels": ["Q1"], "datasets": []},
        "_meta": {"ui": {"resourceUri": "ui://charts/dashboard.html"}},
    }

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="chart_tool")

    assert normalized.render["type"] == "mcp_app"
    assert normalized.render["template_uri"] == "ui://charts/dashboard.html"
    assert normalized.render["structured_content"]["chart_type"] == "bar"


def test_infers_chart_without_template_uri():
    raw_result = {
        "chart_type": "line",
        "labels": ["Mon", "Tue"],
        "datasets": [{"label": "Visits", "data": [10, 12]}],
    }

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="chart_tool")

    assert normalized.model_content.startswith("{")
    assert normalized.render["type"] == "chart"
    assert normalized.render["structured_content"]["labels"] == ["Mon", "Tue"]


def test_infers_table_without_template_uri():
    raw_result = {
        "columns": ["Name", "Score"],
        "rows": [["Ada", 99], ["Grace", 98]],
    }

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="table_tool")

    assert normalized.render["type"] == "table"
    assert normalized.render["structured_content"]["rows"][0] == ["Ada", 99]


def test_unwraps_langchain_text_content_blocks():
    raw_result = [{"type": "text", "text": "The answer is 42."}]

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="answer_tool")

    assert normalized.model_content == "The answer is 42."
    assert normalized.output_preview == "The answer is 42."
    assert normalized.render["type"] == "text"
    assert normalized.render["content"] == [{"type": "text", "text": "The answer is 42."}]


def test_preserves_image_content_as_image_render():
    raw_result = {
        "content": [
            {
                "type": "image",
                "mimeType": "image/png",
                "data": "iVBORw0KGgo=",
            }
        ]
    }

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="image_tool")

    assert normalized.model_content == "[image/png image]"
    assert normalized.render["type"] == "image"
    assert normalized.render["content"][0]["type"] == "image"


def test_error_normalization_uses_error_render_type():
    normalized = normalize_tool_result_for_rendering(
        "Error: permission denied",
        tool_name="dangerous_tool",
        error="permission denied",
    )

    assert normalized.model_content == "Error: permission denied"
    assert normalized.render["type"] == "error"
    assert normalized.render["error"] == "permission denied"


def test_error_normalization_does_not_double_prefix():
    """Caller already-prefixed error strings must not become 'Error: Error: ...'."""
    normalized = normalize_tool_result_for_rendering(
        "Error: Tool name missing",
        tool_name="unknown",
        error="Error: Tool name missing",
    )

    assert normalized.model_content == "Error: Tool name missing"
    assert normalized.render["error"] == "Tool name missing"


def test_mixed_text_and_image_content_blocks_produce_image_render():
    """Mixed content lists must be recognized as image content, not collapsed to JSON."""
    raw_result = [
        {"type": "text", "text": "Here is the chart."},
        {"type": "image", "mimeType": "image/png", "data": "abc"},
    ]

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="chart_tool")

    assert normalized.render["type"] == "image"
    assert normalized.render["content"][0]["type"] == "text"
    assert normalized.render["content"][1]["type"] == "image"
    assert normalized.model_content == "Here is the chart."


def test_bare_text_content_block_dict_is_unwrapped_to_text():
    """A single MCP text content block dict should render as text, not json."""
    raw_result = {"type": "text", "text": "The answer is 42."}

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="answer_tool")

    assert normalized.render["type"] == "text"
    assert normalized.model_content == "The answer is 42."
    assert "structured_content" not in normalized.render


def test_large_inline_image_data_is_redacted():
    """Base64 image data over the threshold is redacted to keep render bounded."""
    large_data = "A" * (128 * 1024)
    raw_result = {"content": [{"type": "image", "mimeType": "image/png", "data": large_data}]}

    normalized = normalize_tool_result_for_rendering(raw_result, tool_name="image_tool")

    block = normalized.render["content"][0]
    assert block["data"] == ""
    assert block["_truncated"] is True
    assert block["_original_size"] == len(large_data)
