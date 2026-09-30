"""The HTTP session the Streamlit frontend uses to call the local sidecar.

The sidecar refuses requests that lack this launch's token in ``X-Kani-Client``
(see ``client_backend/core/request_trust.py``). The token is read from the file
the sidecar writes at startup, and re-read once when the sidecar answers 401:
it rotates every launch, so a sidecar restart must not strand a running UI.

The token is attached only to requests for the sidecar's own base URL, and is
stripped from anything else -- including a redirect off that URL -- so it never
reaches another host.
"""

from __future__ import annotations

from collections.abc import Callable

import requests

from client_backend.core.launch_token import HEADER_NAME, read_launch_token

__all__ = ["SidecarSession"]


class SidecarSession(requests.Session):
    def __init__(
        self,
        base_url: str,
        token_reader: Callable[[], str | None] = read_launch_token,
    ) -> None:
        super().__init__()
        # Normalised the way requests prepares URLs, so the prefix test matches.
        self._base_url = (requests.Request("GET", base_url).prepare().url or "").rstrip("/")
        self._read_token = token_reader
        self._token: str | None = None
        self._token_loaded = False

    def send(self, request: requests.PreparedRequest, **kwargs) -> requests.Response:
        if not self._is_sidecar_url(request.url or ""):
            request.headers.pop(HEADER_NAME, None)
            return super().send(request, **kwargs)

        token = self._current_token()
        _stamp(request, token)
        response = super().send(request, **kwargs)
        if response.status_code != 401:
            return response

        # A 401 is either a bearer-session refusal or a token from an earlier
        # launch. Only a changed token file tells them apart; resend just then.
        fresh = self._reload_token()
        if not fresh or fresh == token:
            return response
        response.close()
        retry = request.copy()
        _stamp(retry, fresh)
        return super().send(retry, **kwargs)

    def _is_sidecar_url(self, url: str) -> bool:
        return url == self._base_url or url.startswith(self._base_url + "/")

    def _current_token(self) -> str | None:
        if not self._token_loaded:
            return self._reload_token()
        return self._token

    def _reload_token(self) -> str | None:
        self._token = self._read_token()
        self._token_loaded = True
        return self._token


def _stamp(request: requests.PreparedRequest, token: str | None) -> None:
    if token:
        request.headers[HEADER_NAME] = token
    else:
        request.headers.pop(HEADER_NAME, None)
