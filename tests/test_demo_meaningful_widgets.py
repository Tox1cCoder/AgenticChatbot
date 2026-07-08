"""Tests that the demo.py live widget component is an HTML-only iframe renderer.

After the HTML-only migration the Streamlit component must contain only the
sandboxed-iframe rendering path: no structured renderers (table/chart/dashboard/
form/list) and no state-field aliases.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest


class _SessionState(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class _CacheDecorator:
    def __call__(self, *args: Any, **kwargs: Any):
        return lambda func: func

    def clear(self) -> None:
        return None


class _StreamlitStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.query_params: dict[str, str] = {}
        self.cache_data = _CacheDecorator()
        self.cache_resource = _CacheDecorator()

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        return None

    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def divider(self, *args: Any, **kwargs: Any) -> None:
        return None

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any):
            return None

        return _noop


def _import_demo_with_ui_stubs(monkeypatch: pytest.MonkeyPatch):
    streamlit_stub = _StreamlitStub()
    components_module = types.ModuleType("streamlit.components")
    components_v1_module = types.ModuleType("streamlit.components.v1")
    components_v1_module.html = lambda *args, **kwargs: None
    components_v1_module.declare_component = (
        lambda *args, **kwargs: (lambda **_component_kwargs: _component_kwargs.get("default"))
    )
    components_module.v1 = components_v1_module
    streamlit_stub.components = components_module
    markdown_stub = types.ModuleType("markdown")
    markdown_stub.markdown = lambda text, **_kwargs: text

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "streamlit.components", components_module)
    monkeypatch.setitem(sys.modules, "streamlit.components.v1", components_v1_module)
    monkeypatch.setitem(sys.modules, "markdown", markdown_stub)
    sys.modules.pop("demo", None)
    return importlib.import_module("demo"), streamlit_stub


def _sample_widget() -> dict[str, Any]:
    return {
        "widget_id": "w-1",
        "session_id": "conv-1",
        "widget_type": "html",
        "title": "Harmonic Oscillation",
        "status": "active",
        "version": 1,
        "connection_endpoint": "/widgets/w-1/connection",
    }


def test_live_widget_component_renders_sandboxed_iframe(monkeypatch):
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    markup = demo._build_live_widget_component_html(_sample_widget(), "token")

    assert "renderHtmlWidget" in markup
    assert "<iframe" in markup
    assert 'sandbox="allow-scripts' in markup
    assert "srcdoc=" in markup


def test_live_widget_component_keeps_websocket_connection_path(monkeypatch):
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    markup = demo._build_live_widget_component_html(_sample_widget(), "token")

    assert "WebSocket" in markup
    assert "widget_state_sync" in markup
    assert "connect(" in markup


def test_live_widget_component_has_no_structured_renderers(monkeypatch):
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    markup = demo._build_live_widget_component_html(_sample_widget(), "token")

    for removed in (
        "renderTable",
        "renderChart",
        "renderDashboard",
        "renderForm",
        "renderList",
        "normalizeChartPayload",
        "renderControls",
    ):
        assert removed not in markup, f"structured renderer {removed} should be gone"


def test_live_widget_component_reads_only_contract_html_fields(monkeypatch):
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    markup = demo._build_live_widget_component_html(_sample_widget(), "token")

    # The HTML renderer must read only the contract keys, not the old aliases.
    for alias in ("data.document", "data.content", "data.srcdoc", "data.min_height", "data.minHeight"):
        assert alias not in markup, f"alias {alias} must not be read by the HTML renderer"


def test_live_widget_component_js_strings_have_no_raw_newlines(monkeypatch):
    """Regression: a Python `\\n` inside a JS double-quoted string becomes a raw
    newline at render time, which is a JS parse error and prevents the bootstrap
    `connect()` call from ever firing. Asserting that no quoted JS string in the
    rendered template spans a newline catches that class of bug.
    """
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    markup = demo._build_live_widget_component_html(_sample_widget(), "token")

    # Strip away CSS — the <style> block legitimately has rules spanning lines.
    script_start = markup.find("<script>")
    script_end = markup.rfind("</script>")
    assert script_start != -1 and script_end != -1
    script = markup[script_start:script_end]

    # Walk through the JS source and verify no `"..."` literal contains a raw
    # newline character. Template literals (backticks) are allowed multi-line.
    i = 0
    line = 1
    while i < len(script):
        ch = script[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < len(script):
                c = script[j]
                if c == "\\":
                    j += 2
                    continue
                if c == "\n":
                    raise AssertionError(
                        f"Raw newline inside JS double-quoted string at line ~{line}: "
                        f"{script[max(0, i - 30) : j + 10]!r}"
                    )
                if c == '"':
                    break
                j += 1
            i = j + 1
            continue
        i += 1
