from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext
from pathlib import Path
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


def _seed_chat_state(stub: _StreamlitStub, conversation_id: str) -> None:
    stub.session_state.clear()
    stub.session_state.current_conversation_id = conversation_id
    stub.session_state.conversation_messages_page = 1
    stub.session_state.has_more_messages = False
    stub.session_state.messages = [
        {
            "id": "msg-1",
            "sender": "user",
            "content": "hello before the plan existed",
            "metadata": {},
        }
    ]
    stub.session_state.pending_image_attachments = []
    stub.session_state.pending_file_attachments = []
    stub.session_state.conversation_messages_meta = {}


def test_chat_view_renders_plan_widget_when_conversation_cache_is_stale(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    _seed_chat_state(streamlit_stub, "conv-1")
    streamlit_stub.session_state.conversations_list = []

    calls: list[dict[str, Any] | None] = []

    def record_plan_widget(conversation_id: str, *, current_conv: dict[str, Any] | None) -> None:
        calls.append(current_conv)
        raise RuntimeError("stop after plan widget")

    monkeypatch.setattr(demo, "_render_plan_progress_widget", record_plan_widget)

    with pytest.raises(RuntimeError, match="stop after plan widget"):
        demo.render_chat_view()

    assert calls == [None]


def test_chat_view_renders_plan_widget_when_cached_flag_is_false(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    _seed_chat_state(streamlit_stub, "conv-2")
    cached_conversation = {
        "id": "conv-2",
        "title": "Existing chat",
        "planningModeEnabled": False,
    }
    streamlit_stub.session_state.conversations_list = [cached_conversation]

    calls: list[dict[str, Any] | None] = []

    def record_plan_widget(conversation_id: str, *, current_conv: dict[str, Any] | None) -> None:
        calls.append(current_conv)
        raise RuntimeError("stop after plan widget")

    monkeypatch.setattr(demo, "_render_plan_progress_widget", record_plan_widget)

    with pytest.raises(RuntimeError, match="stop after plan widget"):
        demo.render_chat_view()

    assert calls == [cached_conversation]


def test_live_rich_renderer_places_resolved_widget_between_markdown(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    calls: list[tuple[str, Any]] = []

    class MarkdownSlot:
        def markdown(self, text: str) -> None:
            calls.append(("markdown", text))

    class RootPlaceholder:
        def container(self):
            return nullcontext()

    streamlit_stub.empty = lambda: MarkdownSlot()
    streamlit_stub.caption = lambda text: calls.append(("caption", text))
    monkeypatch.setattr(
        demo,
        "_render_inline_rich_item",
        lambda item, **kwargs: calls.append(("rich", (item["id"], kwargs["auto_mount"]))),
    )

    renderer = demo._StreamingRichResponseRenderer(
        RootPlaceholder(),
        message_key="active-response",
    )
    renderer.append_text("Before\n\n<!--rich:widget:w-1-->\n\nAfter")
    calls.clear()
    renderer.apply_rich_items_upsert(
        [
            {
                "id": "widget:w-1",
                "type": "live_widget",
                "display_policy": "inline_or_append",
                "payload": {
                    "widget_id": "w-1",
                    "session_id": "conv-1",
                    "widget_type": "chart",
                    "status": "active",
                    "version": 1,
                    "connection_endpoint": "/widgets/w-1/connection",
                },
            }
        ]
    )

    assert calls == [
        ("markdown", "Before"),
        ("rich", ("widget:w-1", True)),
        ("markdown", "After"),
    ]


def test_live_rich_renderer_only_auto_mounts_first_repeated_widget_marker(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    auto_mounts: list[bool] = []

    class MarkdownSlot:
        def markdown(self, _text: str) -> None:
            return None

    class RootPlaceholder:
        def container(self):
            return nullcontext()

    streamlit_stub.empty = lambda: MarkdownSlot()
    streamlit_stub.caption = lambda _text: None
    monkeypatch.setattr(
        demo,
        "_render_inline_rich_item",
        lambda _item, **kwargs: auto_mounts.append(kwargs["auto_mount"]),
    )

    renderer = demo._StreamingRichResponseRenderer(
        RootPlaceholder(),
        message_key="active-response",
    )
    renderer.append_text("<!--rich:widget:w-1-->\n\nText\n\n<!--rich:widget:w-1-->")
    renderer.apply_rich_items_upsert(
        [
            {
                "id": "widget:w-1",
                "type": "live_widget",
                "display_policy": "inline_or_append",
                "payload": {
                    "widget_id": "w-1",
                    "session_id": "conv-1",
                    "widget_type": "chart",
                    "status": "active",
                    "version": 1,
                    "connection_endpoint": "/widgets/w-1/connection",
                },
            }
        ]
    )

    assert auto_mounts == [True, False]


def test_live_rich_renderer_does_not_append_unplaced_widget_before_completion(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[str] = []

    class MarkdownSlot:
        def markdown(self, _text: str) -> None:
            return None

    class RootPlaceholder:
        def container(self):
            return nullcontext()

    streamlit_stub.empty = lambda: MarkdownSlot()
    monkeypatch.setattr(
        demo,
        "_render_inline_rich_item",
        lambda item, **_kwargs: rendered.append(item["id"]),
    )

    renderer = demo._StreamingRichResponseRenderer(
        RootPlaceholder(),
        message_key="active-response",
    )
    renderer.append_text("Intro paragraph.\n\nMore explanation.")
    renderer.apply_rich_items_upsert(
        [
            {
                "id": "widget:w-1",
                "type": "live_widget",
                "display_policy": "inline_or_append",
                "payload": {
                    "widget_id": "w-1",
                    "session_id": "conv-1",
                    "widget_type": "chart",
                    "status": "active",
                    "version": 1,
                    "connection_endpoint": "/widgets/w-1/connection",
                },
            }
        ]
    )

    assert rendered == []


def test_live_rich_renderer_shows_pending_widget_slot_until_upsert(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[tuple[str, str]] = []

    class MarkdownSlot:
        def markdown(self, _text: str) -> None:
            return None

    class RootPlaceholder:
        def container(self):
            return nullcontext()

    streamlit_stub.empty = lambda: MarkdownSlot()
    monkeypatch.setattr(
        demo,
        "_render_pending_rich_placeholder",
        lambda item_id: rendered.append(("pending", item_id)),
        raising=False,
    )
    monkeypatch.setattr(
        demo,
        "_render_inline_rich_item",
        lambda item, **_kwargs: rendered.append(("rich", item["id"])),
    )

    renderer = demo._StreamingRichResponseRenderer(
        RootPlaceholder(),
        message_key="active-response",
    )
    renderer.append_text("Before\n\n<!--rich:widget:w-1-->\n\nAfter")
    assert rendered == [("pending", "widget:w-1")]

    rendered.clear()
    renderer.apply_rich_items_upsert(
        [
            {
                "id": "widget:w-1",
                "type": "live_widget",
                "display_policy": "inline_or_append",
                "payload": {
                    "widget_id": "w-1",
                    "session_id": "conv-1",
                    "widget_type": "chart",
                    "status": "active",
                    "version": 1,
                    "connection_endpoint": "/widgets/w-1/connection",
                },
            }
        ]
    )

    assert rendered == [("rich", "widget:w-1")]


def test_inline_live_widget_mount_omits_legacy_attachment_chrome(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    markdown: list[str] = []
    details: list[str] = []
    mounts: list[str] = []

    streamlit_stub.session_state.auth_token = "token"
    streamlit_stub.session_state.live_widget_mounts = {}
    streamlit_stub.markdown = lambda text, **_kwargs: markdown.append(text)

    def capture_expander(label: str, **_kwargs):
        details.append(label)
        return nullcontext()

    streamlit_stub.expander = capture_expander
    monkeypatch.setattr(
        demo._stc,
        "html",
        lambda markup, **_kwargs: mounts.append(markup),
    )

    demo.render_live_widgets(
        {
            "live_widgets": [
                {
                    "widget_id": "w-inline",
                    "session_id": "conv-1",
                    "widget_type": "chart",
                    "title": "Growth",
                    "status": "active",
                    "version": 1,
                }
            ]
        },
        message_key="inline",
        auto_mount=True,
        inline=True,
    )

    assert mounts
    assert all("Live Widget" not in text for text in markdown)
    assert details == []


def test_rich_item_widget_uses_inline_widget_mode(monkeypatch):
    demo, _streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        demo,
        "render_live_widgets",
        lambda _metadata, **kwargs: calls.append(kwargs),
    )

    demo._render_inline_rich_item(
        {
            "id": "widget:w-inline",
            "type": "live_widget",
            "display_policy": "inline_or_append",
            "payload": {"widget_id": "w-inline", "widget_type": "chart"},
        },
        message_metadata={},
        message_key="answer",
        auto_mount=True,
    )

    assert calls[0]["inline"] is True


def test_inline_tool_render_uses_existing_demo_renderer(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[dict[str, Any]] = []
    json_fallbacks: list[dict[str, Any]] = []
    payload = {"type": "table", "structured_content": [{"name": "Ada"}]}

    monkeypatch.setattr(
        demo,
        "render_tool_render_payload",
        lambda render: rendered.append(render) or True,
    )
    streamlit_stub.json = lambda render: json_fallbacks.append(render)

    demo._render_inline_rich_item(
        {
            "id": "tool:call-1",
            "type": "tool_render",
            "display_policy": "inline_or_append",
            "payload": {"render": payload},
        },
        message_metadata={},
        message_key="response",
        auto_mount=False,
    )

    assert rendered == [payload]
    assert json_fallbacks == []


def test_image_tool_render_displays_text_and_base64_image(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    markdown_calls: list[tuple[str, dict[str, Any]]] = []
    streamlit_stub.markdown = lambda text, **kwargs: markdown_calls.append((text, kwargs))
    streamlit_stub.expander = lambda *_args, **_kwargs: nullcontext()
    monkeypatch.setattr(demo, "render_tool_result_payload", lambda *_args, **_kwargs: None)

    rendered = demo.render_tool_render_payload(
        {
            "type": "image",
            "content": [
                {"type": "text", "text": "Generated chart"},
                {"type": "image", "mimeType": "image/png", "data": "YWJj"},
            ],
        }
    )

    assert rendered is True
    assert any(text == "Generated chart" for text, _kwargs in markdown_calls)
    assert any("data:image/png;base64,YWJj" in text for text, _kwargs in markdown_calls)


def test_chart_tool_render_preserves_labels_as_x_axis(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    chart_calls: list[tuple[Any, dict[str, Any]]] = []
    streamlit_stub.line_chart = lambda data, **kwargs: chart_calls.append((data, kwargs))
    streamlit_stub.expander = lambda *_args, **_kwargs: nullcontext()
    streamlit_stub.dataframe = lambda *_args, **_kwargs: None
    monkeypatch.setattr(demo, "render_tool_result_payload", lambda *_args, **_kwargs: None)

    assert demo.render_tool_render_payload(
        {
            "type": "chart",
            "structured_content": {
                "chart_type": "line",
                "labels": ["Mon", "Tue"],
                "datasets": [{"label": "Visits", "data": [10, 12]}],
            },
        }
    )

    assert chart_calls == [
        (
            {"label": ["Mon", "Tue"], "Visits": [10, 12]},
            {"x": "label", "y": ["Visits"]},
        )
    ]


def test_chart_tool_render_preserves_duplicate_labels_and_series_names(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    chart_calls: list[tuple[Any, dict[str, Any]]] = []
    streamlit_stub.line_chart = lambda data, **kwargs: chart_calls.append((data, kwargs))
    streamlit_stub.expander = lambda *_args, **_kwargs: nullcontext()
    streamlit_stub.dataframe = lambda *_args, **_kwargs: None
    monkeypatch.setattr(demo, "render_tool_result_payload", lambda *_args, **_kwargs: None)

    demo.render_tool_render_payload(
        {
            "type": "chart",
            "structured_content": {
                "chart_type": "line",
                "labels": ["Jan", "Jan"],
                "datasets": [
                    {"label": "Revenue", "data": [1, 2]},
                    {"label": "Revenue", "data": [3, 4]},
                ],
            },
        }
    )

    assert chart_calls == [
        (
            {
                "label": ["Jan", "Jan"],
                "Revenue": [1, 2],
                "Revenue (2)": [3, 4],
            },
            {"x": "label", "y": ["Revenue", "Revenue (2)"]},
        )
    ]


def test_both_demo_stream_loops_use_segmented_rich_renderer():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    assert source.count("_StreamingRichResponseRenderer(") >= 2
    assert source.count("stream_renderer.append_text(content)") >= 2
    assert source.count("stream_renderer.apply_rich_items_upsert") >= 2


def test_render_citations_escapes_legacy_source_html(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[str] = []
    streamlit_stub.expander = lambda *_args, **_kwargs: nullcontext()
    streamlit_stub.markdown = lambda text, **_kwargs: rendered.append(text)

    demo.render_citations(
        {
            "citations": [
                {
                    "source": '<img src=x onerror="alert(1)">',
                    "score": 0.9,
                }
            ]
        }
    )

    html_output = "\n".join(rendered)
    assert "<img src=x" not in html_output
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html_output


def test_render_citations_escapes_grouped_source_html(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    rendered: list[str] = []
    streamlit_stub.session_state.message_chunks = {}
    streamlit_stub.expander = lambda *_args, **_kwargs: nullcontext()
    streamlit_stub.markdown = lambda text, **_kwargs: rendered.append(text)

    demo.render_citations(
        {
            "documents_cited": [
                {
                    "document_number": "<b>1</b>",
                    "source": "<script>alert(1)</script>",
                    "total_chunks": 0,
                    "avg_score": 0.9,
                    "chunks": [],
                }
            ]
        },
        "msg-1",
    )

    html_output = "\n".join(rendered)
    assert "<script>" not in html_output
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_output
    assert "&lt;b&gt;1&lt;/b&gt;" in html_output


def test_render_citations_can_suppress_v1_image_gallery(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    gallery_calls: list[list[dict[str, str]]] = []
    streamlit_stub.session_state.message_chunks = {}
    streamlit_stub.expander = lambda *_args, **_kwargs: nullcontext()
    monkeypatch.setattr(
        demo,
        "_render_thumbnail_gallery",
        lambda items, **_kwargs: gallery_calls.append(items),
    )

    demo.render_citations(
        {
            "documents_cited": [
                {
                    "document_id": "doc-1",
                    "document_number": 1,
                    "source": "report.pdf",
                    "total_chunks": 0,
                    "avg_score": 0.9,
                    "chunks": [],
                }
            ],
            "images": [
                {
                    "name": "doc-1-page-1",
                    "url": "https://img.test/selected.png",
                    "mime": "image/png",
                }
            ],
        },
        "msg-1",
        include_images=False,
    )

    assert gallery_calls == []


def test_v1_message_bubble_disables_citation_image_gallery():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    assert "include_images=not view.is_v1" in source
