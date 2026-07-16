from __future__ import annotations

from typing import Any

from tests.test_demo_plan_widget import _import_demo_with_ui_stubs


def _install_form_widget_defaults(streamlit_stub: Any) -> list[dict[str, Any]]:
    selectbox_calls: list[dict[str, Any]] = []

    streamlit_stub.checkbox = lambda _label, **kwargs: kwargs["value"]
    streamlit_stub.number_input = lambda _label, **kwargs: kwargs["value"]
    streamlit_stub.text_input = lambda _label, **kwargs: kwargs.get("value", "")
    streamlit_stub.text_area = lambda _label, **kwargs: kwargs.get("value", "")

    def selectbox(_label: str, **kwargs: Any) -> Any:
        selectbox_calls.append(kwargs)
        index = kwargs.get("index", 0)
        return None if index is None else kwargs["options"][index]

    streamlit_stub.selectbox = selectbox
    return selectbox_calls


def test_parameter_form_omits_untouched_optional_fields_and_honors_defaults(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    selectbox_calls = _install_form_widget_defaults(streamlit_stub)
    schema = {
        "type": "object",
        "properties": {
            "optional_flag": {"type": "boolean"},
            "optional_count": {"type": "integer"},
            "optional_ratio": {"type": "number"},
            "optional_label": {"type": "string"},
            "optional_object": {"type": "object"},
            "optional_array": {"type": "array"},
            "optional_mode": {"type": "string", "enum": ["fast", "safe"]},
            "defaulted_mode": {
                "type": "string",
                "enum": ["fast", "safe"],
                "default": "safe",
            },
            "defaulted_count": {"type": "integer", "default": 0},
        },
    }

    parameters, errors = demo.render_tool_parameter_form(schema, key_prefix="alpha::inspect")

    assert errors == []
    assert parameters == {"defaulted_mode": "safe", "defaulted_count": 0}
    enum_calls = [call for call in selectbox_calls if call["options"] == ["fast", "safe"]]
    assert [call["index"] for call in enum_calls] == [None, 1]


def test_tester_execution_payload_and_results_are_qualified_by_server(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_request(method: str, endpoint: str, payload: dict[str, Any]):
        calls.append((method, endpoint, payload))
        return {"data": {"success": True, "serverName": payload["serverName"]}}

    monkeypatch.setattr(demo, "make_api_request", fake_request)
    alpha = demo.execute_mcp_tool(
        "inspect",
        {"value": 1},
        server_name="alpha",
        qualified_tool_id="alpha::inspect",
    )
    beta = {"success": True, "serverName": "beta"}
    demo._remember_mcp_tool_execution_result("alpha::inspect", alpha)
    demo._remember_mcp_tool_execution_result("beta::inspect", beta)

    assert calls == [
        (
            "POST",
            "/mcp/tools/inspect/execute",
            {
                "arguments": {"value": 1},
                "serverName": "alpha",
                "qualifiedToolId": "alpha::inspect",
            },
        )
    ]
    assert streamlit_stub.session_state["mcp_tool_execution_results"] == {
        "alpha::inspect": {"success": True, "serverName": "alpha"},
        "beta::inspect": beta,
    }

    demo._clear_mcp_tool_execution_result("alpha::inspect")

    assert streamlit_stub.session_state["mcp_tool_execution_results"] == {"beta::inspect": beta}


def test_tester_renders_mcp_image_content_as_rich_image_not_base64_json(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    markdown_calls: list[tuple[str, dict[str, Any]]] = []
    code_calls: list[tuple[Any, dict[str, Any]]] = []
    streamlit_stub.markdown = lambda value, **kwargs: markdown_calls.append((value, kwargs))
    streamlit_stub.code = lambda value, **kwargs: code_calls.append((value, kwargs))

    demo.render_tool_result_payload(
        [
            {"type": "text", "text": "preview"},
            {"type": "image", "data": "QUJD", "mimeType": "image/png"},
        ]
    )

    assert any("preview" in value for value, _kwargs in markdown_calls)
    assert any("data:image/png;base64,QUJD" in value for value, _kwargs in markdown_calls)
    assert any(kwargs.get("unsafe_allow_html") is True for _value, kwargs in markdown_calls)
    assert code_calls == []
