"""Tests for the widget_quality module (objective state validation)."""

from __future__ import annotations

from app.services.widget_quality import assess_widget_state


def test_chart_requires_labels_and_numeric_datasets():
    result = assess_widget_state("chart", {"chart_type": "bar", "labels": [], "datasets": []})

    assert result.allowed is False
    assert "chart requires at least two labels" in result.messages


def test_chart_requires_dataset_values_for_every_label():
    result = assess_widget_state(
        "chart",
        {
            "chart_type": "bar",
            "labels": ["A", "B", "C"],
            "datasets": [{"label": "X", "data": [1, 2]}],
        },
    )

    assert result.allowed is False
    assert any("numeric values for every label" in m for m in result.messages)


def test_chart_bar_with_two_points_is_allowed():
    result = assess_widget_state(
        "chart",
        {
            "chart_type": "bar",
            "labels": ["A", "B"],
            "datasets": [{"label": "Score", "data": [1, 2]}],
        },
    )

    assert result.allowed is True


def test_donut_rejects_negative_values():
    result = assess_widget_state(
        "chart",
        {
            "chart_type": "donut",
            "labels": ["A", "B"],
            "datasets": [{"label": "Share", "data": [5, -1]}],
        },
    )

    assert result.allowed is False
    assert "donut charts require non-negative values" in result.messages


def test_donut_rejects_multiple_series():
    result = assess_widget_state(
        "chart",
        {
            "chart_type": "pie",
            "labels": ["A", "B"],
            "datasets": [
                {"label": "S1", "data": [1, 2]},
                {"label": "S2", "data": [3, 4]},
            ],
        },
    )

    assert result.allowed is False
    assert any("part-to-whole" in m for m in result.messages)


def test_line_chart_requires_ordered_x_kind():
    result = assess_widget_state(
        "chart",
        {
            "chart_type": "line",
            "labels": ["A", "B", "C"],
            "datasets": [{"label": "X", "data": [1, 2, 3]}],
        },
    )

    assert result.allowed is False
    assert any("x_kind" in m for m in result.messages)


def test_line_chart_with_time_x_kind_is_allowed():
    result = assess_widget_state(
        "chart",
        {
            "chart_type": "line",
            "labels": ["t0", "t1", "t2"],
            "datasets": [{"label": "Trajectory", "data": [0.1, 0.4, 0.9]}],
            "presentation": {"x_kind": "time"},
        },
    )

    assert result.allowed is True


def test_chart_soft_messages_for_missing_caption_and_labels():
    result = assess_widget_state(
        "chart",
        {
            "chart_type": "bar",
            "labels": ["A", "B"],
            "datasets": [{"label": "S", "data": [1, 2]}],
        },
    )

    assert result.allowed is True
    soft = " ".join(result.soft_messages)
    assert "caption" in soft
    assert "axis labels" in soft


def test_table_rejects_no_rows_or_columns():
    result = assess_widget_state("table", {})

    assert result.allowed is False


def test_table_with_empty_state_message_is_allowed():
    result = assess_widget_state("table", {"empty_state": "No results yet."})

    assert result.allowed is True


def test_table_with_rows_and_columns_is_allowed():
    result = assess_widget_state(
        "table",
        {"columns": ["Name"], "rows": [["Alice"]]},
    )

    assert result.allowed is True


def test_dashboard_rejects_empty_state():
    result = assess_widget_state("dashboard", {})

    assert result.allowed is False


def test_dashboard_with_panels_is_allowed():
    result = assess_widget_state(
        "dashboard",
        {"panels": [{"type": "metric", "title": "Total", "value": 42}]},
    )

    assert result.allowed is True


def test_dashboard_with_fallback_metrics_is_allowed():
    result = assess_widget_state("dashboard", {"total": 42, "label": "Revenue"})

    assert result.allowed is True


def test_html_widget_requires_content():
    result = assess_widget_state("html", {"html": "", "height": 400})

    assert result.allowed is False
    assert any("html content" in m for m in result.messages)


def test_html_widget_rejects_out_of_range_height():
    result = assess_widget_state("html", {"html": "<div>hi</div>", "height": 100})

    assert result.allowed is False
    assert any("height" in m for m in result.messages)


def test_html_widget_accepts_in_range_height():
    result = assess_widget_state(
        "html",
        {"html": "<div>hi</div>", "height": 540},
    )

    assert result.allowed is True


def test_non_object_state_for_structured_widget_rejected():
    result = assess_widget_state("chart", "not-a-dict")

    assert result.allowed is False
    assert any("JSON object" in m for m in result.messages)


def test_form_and_list_widgets_are_allowed_by_default():
    assert assess_widget_state("form", {"fields": []}).allowed is True
    assert assess_widget_state("list", {"items": []}).allowed is True
