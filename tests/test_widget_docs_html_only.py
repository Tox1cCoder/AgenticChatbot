"""Doc-contract tests: frontend-facing widget docs must describe HTML-only widgets.

These guard against silently re-advertising structured widget renderers in the
documents shipped to frontend developers.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# plans/AI_SDK_FE_CONTRACT.md
# ---------------------------------------------------------------------------
def test_ai_sdk_contract_live_widget_examples_use_html():
    doc = _read("plans/AI_SDK_FE_CONTRACT.md")
    assert '"widget_type": "html"' in doc
    assert '"widget_type": "table"' not in doc
    assert '"widget_type": "chart"' not in doc


def test_ai_sdk_contract_documents_html_state_and_iframe():
    doc = _read("plans/AI_SDK_FE_CONTRACT.md")
    assert "sandboxed iframe" in doc
    assert "state.html" in doc
    assert '"height": 620' in doc


# ---------------------------------------------------------------------------
# plans/live-widgets-frontend-integration.md
# ---------------------------------------------------------------------------
def test_frontend_integration_doc_is_html_only():
    doc = _read("plans/live-widgets-frontend-integration.md")
    assert '"widget_type": "html"' in doc
    assert '"widget_type": "chart"' not in doc
    assert '"widget_type": "table"' not in doc
    assert "sandboxed iframe" in doc


def test_frontend_integration_doc_drops_structured_view_state():
    doc = _read("plans/live-widgets-frontend-integration.md")
    assert "table_ui" not in doc
    assert "chart_ui" not in doc


# ---------------------------------------------------------------------------
# README.md Live Widgets section
# ---------------------------------------------------------------------------
def test_readme_drops_widget_quality_module_reference():
    doc = _read("README.md")
    assert "widget_quality.py" not in doc
    assert "test_widget_quality.py" not in doc
