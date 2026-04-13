from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import requests


module_spec = importlib.util.spec_from_file_location(
    "take100_api_under_test",
    Path(__file__).resolve().parents[1] / "skills" / "take100" / "take100_api.py",
)
assert module_spec is not None and module_spec.loader is not None
take100_api = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(take100_api)

BASE_URL = take100_api.BASE_URL
Take100Client = take100_api.Take100Client


class FakeResponse:
    def __init__(
        self,
        *,
        text: str = "",
        url: str | None = None,
        status_code: int = 200,
        json_data: dict | None = None,
    ) -> None:
        self.text = text
        self.url = url or f"{BASE_URL}/wt-applications"
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self) -> dict:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class FakeSession:
    def __init__(self, *, request_responses: list[FakeResponse] | None = None) -> None:
        self.cookies = requests.cookies.RequestsCookieJar()
        self.cookies.set("XSRF-TOKEN", "cached-xsrf")
        self.request_responses = list(request_responses or [])
        self.login_get_calls = 0
        self.login_post_calls = 0
        self.request_calls: list[tuple[str, str, dict]] = []

    def clear(self) -> None:
        self.cookies.clear()

    def get(self, url: str, **kwargs) -> FakeResponse:
        if url == f"{BASE_URL}/login":
            self.login_get_calls += 1
            return FakeResponse(text='<input name="_token" value="csrf-token">', url=url)

        self.request_calls.append(("GET", url, kwargs))
        if self.request_responses:
            return self.request_responses.pop(0)
        return FakeResponse(url=url)

    def post(self, url: str, **kwargs) -> FakeResponse:
        if url == f"{BASE_URL}/login":
            self.login_post_calls += 1
            self.cookies.set("XSRF-TOKEN", "fresh-xsrf")
            return FakeResponse(url=f"{BASE_URL}/wt-applications")

        self.request_calls.append(("POST", url, kwargs))
        if self.request_responses:
            return self.request_responses.pop(0)
        return FakeResponse(url=url, json_data={"ok": True})

    def delete(self, url: str, **kwargs) -> FakeResponse:
        self.request_calls.append(("DELETE", url, kwargs))
        if self.request_responses:
            return self.request_responses.pop(0)
        return FakeResponse(url=url, json_data={"ok": True})

    def request(self, method: str, url: str, **kwargs) -> FakeResponse:
        self.request_calls.append((method.upper(), url, kwargs))
        if self.request_responses:
            return self.request_responses.pop(0)
        return FakeResponse(url=url)


def test_login_persists_session_cache(tmp_path: Path) -> None:
    session_cache_path = tmp_path / "take100-session.json"
    client = Take100Client("user@example.com", "secret", session_cache_path=session_cache_path)
    client.session = FakeSession()

    assert client.login() is True

    saved = json.loads(session_cache_path.read_text(encoding="utf-8"))
    assert saved["email"] == "user@example.com"
    assert saved["cookies"]["XSRF-TOKEN"] == "fresh-xsrf"


def test_ensure_authenticated_reuses_valid_cached_session_without_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_cache_path = tmp_path / "take100-session.json"
    session_cache_path.write_text(
        json.dumps(
            {
                "email": "user@example.com",
                "cookies": {"XSRF-TOKEN": "persisted-xsrf"},
            }
        ),
        encoding="utf-8",
    )

    client = Take100Client("user@example.com", "secret", session_cache_path=session_cache_path)

    monkeypatch.setattr(client, "_session_is_authenticated", lambda: True)
    monkeypatch.setattr(
        client,
        "login",
        lambda: pytest.fail("login() should not run when persisted session is still valid"),
    )

    assert client.ensure_authenticated() is True
    assert client.session.cookies.get("XSRF-TOKEN") == "persisted-xsrf"


def test_request_reauthenticates_once_after_login_redirect(tmp_path: Path) -> None:
    session_cache_path = tmp_path / "take100-session.json"
    session_cache_path.write_text(
        json.dumps(
            {
                "email": "user@example.com",
                "cookies": {"XSRF-TOKEN": "persisted-xsrf"},
            }
        ),
        encoding="utf-8",
    )

    client = Take100Client("user@example.com", "secret", session_cache_path=session_cache_path)
    client.session = FakeSession(
        request_responses=[
            FakeResponse(url=f"{BASE_URL}/login"),
            FakeResponse(url=f"{BASE_URL}/wt-applications"),
        ]
    )

    response = client._request("GET", f"{BASE_URL}/wt-applications")

    assert response.url == f"{BASE_URL}/wt-applications"
    assert client.session.login_get_calls == 1
    assert client.session.login_post_calls == 1


def test_main_list_action_does_not_force_eager_login(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class StubClient:
        def __init__(self, email: str, password: str) -> None:
            self.email = email
            self.password = password

        def login(self) -> bool:
            pytest.fail("main() should not force login before dispatching the action")

        def list_applications(self) -> list[dict]:
            return [{"id": "100", "no": "100-100", "process": "draft"}]

    monkeypatch.setattr(take100_api, "Take100Client", StubClient)
    monkeypatch.setattr(
        sys,
        "argv",
        ["take100_api.py", "--action", "list", "--email", "user@example.com", "--password", "secret"],
    )

    take100_api.main()

    result = json.loads(capsys.readouterr().out)
    assert result["success"] is True
    assert result["applications"][0]["no"] == "100-100"
