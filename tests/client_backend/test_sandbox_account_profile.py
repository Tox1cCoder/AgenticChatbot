"""Finding the sandbox account's profile folder, creating it when the account has none.

A freshly set-up account has no profile until something signs in with one, and
``LogonUser`` does not create it. Windows' own ``GetUserProfileDirectoryW`` is
replaced here by a stand-in that answers the way it does for an account with
and without a profile; creating a real profile is verified on a real machine.
"""

from __future__ import annotations

import ctypes
import sys

import pytest

from client_backend.services.sandbox import windows

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows sandbox only")

PROFILE = r"C:\Users\KaniSb84cfd2fb7333"
_ERROR_FILE_NOT_FOUND = 2
# How the size query fails while there is no profile. A new account in the
# field answered "too small" without saying what size would do, which made
# the buffer zero-sized and the launch fail with WinError 122.
_NO_PROFILE_ANSWERS = {
    "too-small-without-a-size": windows._ERROR_INSUFFICIENT_BUFFER,
    "not-found": _ERROR_FILE_NOT_FOUND,
}


class _Userenv:
    """``GetUserProfileDirectoryW`` for an account whose profile may not exist yet."""

    def __init__(self, profile: str | None, no_profile_error: int = _ERROR_FILE_NOT_FOUND):
        self.profile = profile
        self.no_profile_error = no_profile_error

    def GetUserProfileDirectoryW(self, _token, buffer, size_ref) -> int:  # noqa: N802
        size = size_ref._obj
        if self.profile is None:
            ctypes.set_last_error(self.no_profile_error)
            return 0
        needed = len(self.profile) + 1
        if buffer is None or size.value < needed:
            size.value = needed
            ctypes.set_last_error(windows._ERROR_INSUFFICIENT_BUFFER)
            return 0
        buffer.value = self.profile
        return 1


def test_an_existing_profile_is_found():
    assert windows._profile_directory(_Userenv(PROFILE), token=None) == PROFILE


@pytest.mark.parametrize("error", _NO_PROFILE_ANSWERS.values(), ids=_NO_PROFILE_ANSWERS.keys())
def test_a_new_account_gets_its_profile_created_once(monkeypatch, error):
    userenv = _Userenv(None, no_profile_error=error)
    created: list[str] = []

    def create_profile(username, _password):
        created.append(username)
        userenv.profile = PROFILE

    monkeypatch.setattr(windows, "_create_profile", create_profile)

    assert windows._account_profile(userenv, None, "KaniSb84cfd2fb7333", "pw") == PROFILE
    assert created == ["KaniSb84cfd2fb7333"]


def test_an_account_with_a_profile_is_not_signed_in_again(monkeypatch):
    monkeypatch.setattr(windows, "_create_profile", lambda *_: pytest.fail("signed in again"))

    assert windows._account_profile(_Userenv(PROFILE), None, "KaniSandbox", "pw") == PROFILE


def test_a_profile_windows_will_not_create_is_reported(monkeypatch):
    monkeypatch.setattr(windows, "_create_profile", lambda *_: None)

    with pytest.raises(OSError) as raised:
        windows._account_profile(_Userenv(None), None, "KaniSb84cfd2fb7333", "pw")

    assert raised.value.winerror == _ERROR_FILE_NOT_FOUND
    assert "has no Windows profile" in windows.explain_windows_error(raised.value)
