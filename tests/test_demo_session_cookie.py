"""The demo keeps its session across reloads without putting the token in a URL.

It used to restore the token from localStorage by redirecting to
``?__t=<token>&__u=<user>``, which wrote the token into browser history and
the Streamlit server's request log. The browser now keeps it in a
``SameSite=Strict`` cookie that Streamlit reads back through
``st.context.cookies`` when a reload opens a new session.
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

DEMO_SOURCE = Path(__file__).resolve().parent.parent / "demo.py"
TOKEN = "eyJhbGciOiJIUzI1NiJ9.local-session.signature"


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
    def __init__(self, *, cookies: dict[str, str], query_params: dict[str, str]) -> None:
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.query_params = dict(query_params)
        self.context = types.SimpleNamespace(cookies=dict(cookies))
        self.cache_data = _CacheDecorator()
        self.cache_resource = _CacheDecorator()

    def __getattr__(self, _name: str):
        return lambda *_args, **_kwargs: None


def _import_demo(monkeypatch, *, cookies=None, query_params=None):
    streamlit_stub = _StreamlitStub(cookies=cookies or {}, query_params=query_params or {})
    scripts: list[str] = []
    components_module = types.ModuleType("streamlit.components")
    components_v1_module = types.ModuleType("streamlit.components.v1")
    components_v1_module.html = lambda html, *_args, **_kwargs: scripts.append(html)
    components_v1_module.declare_component = lambda *args, **kwargs: (
        lambda **component_kwargs: component_kwargs.get("default")
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
    return importlib.import_module("demo"), streamlit_stub, scripts


def test_a_reload_restores_the_session_from_the_cookie(monkeypatch):
    _demo, st, _scripts = _import_demo(monkeypatch, cookies={"cbtoken": TOKEN, "cbuid": "user-1"})

    assert st.session_state.auth_token == TOKEN
    assert st.session_state.current_user_id == "user-1"
    assert st.session_state.show_login is False


def test_a_token_in_the_url_is_never_restored_and_is_scrubbed(monkeypatch):
    """An old bookmark or history entry must not sign anyone in, or keep the token."""
    _demo, st, _scripts = _import_demo(
        monkeypatch, query_params={"__restore": "1", "__t": TOKEN, "__u": "user-1", "tab": "x"}
    )

    assert not st.session_state.get("auth_token")
    assert st.query_params == {"tab": "x"}


def test_the_cookie_is_read_once_so_signing_out_sticks(monkeypatch):
    """``st.context.cookies`` holds the reload's cookies for the whole session."""
    demo, st, _scripts = _import_demo(monkeypatch, cookies={"cbtoken": TOKEN, "cbuid": "user-1"})
    st.session_state.auth_token = None
    st.session_state.current_user_id = None

    demo._restore_session_from_cookie()

    assert st.session_state.auth_token is None


def test_signing_in_stores_the_token_in_a_strict_cookie_not_the_url(monkeypatch):
    demo, st, scripts = _import_demo(monkeypatch)
    st.session_state._ls_op = {"token": TOKEN, "uid": "user-1"}

    demo._flush_session_cookie_op()

    [script] = scripts[-1:]
    assert "SameSite=Strict" in script
    assert "Path=/" in script
    assert TOKEN in script
    assert not re.search(r"location\.(replace|assign|href|search)|searchParams", script)
    # What earlier versions left in localStorage is removed, not kept alongside.
    assert "localStorage.removeItem('cbtoken')" in script
    assert st.session_state._ls_op is None


def test_signing_out_expires_the_cookie(monkeypatch):
    demo, st, scripts = _import_demo(monkeypatch)
    st.session_state._ls_op = "clear"

    demo._flush_session_cookie_op()

    assert "cbtoken=; Path=/; SameSite=Strict; Max-Age=0" in scripts[-1]
    assert TOKEN not in scripts[-1]


def test_no_code_path_puts_the_token_into_query_params():
    source = DEMO_SOURCE.read_text(encoding="utf-8")

    assert "searchParams.set" not in source
    assert not re.search(r"query_params\s*\[[^\]]*\]\s*=(?!=)", source)
    assert not re.search(r"query_params\.(update|from_dict|setdefault)\(", source)
    assert "__t" not in source.replace('"__t"', "").replace("'__t'", "")


@pytest.mark.parametrize("value", [TOKEN, "user-1"])
def test_no_restored_value_is_left_in_the_url(monkeypatch, value):
    _demo, st, _scripts = _import_demo(
        monkeypatch,
        cookies={"cbtoken": TOKEN, "cbuid": "user-1"},
        query_params={"__t": TOKEN, "__u": "user-1"},
    )

    assert value not in st.query_params.values()
