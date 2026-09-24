"""Let the sandbox account resolve paths into a workspace inside the user's profile.

PowerShell and node read the attributes of every folder on the way to a path,
and folders in the user's profile are closed to other accounts: PowerShell then
silently starts at ``C:\\`` and node cannot load a script. Each folder strictly
between the profile and a workspace root gets one entry for the account --
read attributes, on that folder alone. It cannot list the folder, read its
files, or reach anything below except the granted workspace. (The profile
folder itself needs nothing: ``C:\\Users`` is listable, and that already lets
any account read the attributes of the folders in it.)

The entry is written with ``SetFileSecurity``, which changes only the folder
named. ``icacls`` and ``SetNamedSecurityInfo`` re-stamp every file below,
which on a profile takes minutes and touches every file the user has.
Granted folders are recorded so ``sandbox remove`` can take the entries out.
"""

from __future__ import annotations

import ctypes
import json
import re
from ctypes import wintypes
from pathlib import Path

from client_backend.core.config import client_settings
from client_backend.services.sandbox.account import SANDBOX_USERNAME

_DACL_SECURITY_INFORMATION = 0x00000004
_SDDL_REVISION_1 = 1
_SE_DACL_AUTO_INHERIT_REQ = 0x0100
_SE_DACL_AUTO_INHERITED = 0x0400
_SE_DACL_PROTECTED = 0x1000
_CONTROL_MASK = _SE_DACL_AUTO_INHERIT_REQ | _SE_DACL_AUTO_INHERITED | _SE_DACL_PROTECTED
# FILE_READ_ATTRIBUTES | SYNCHRONIZE, with no inheritance flags: this folder only.
_RIGHTS = "0x100080"
_ACE = re.compile(r"\(([^()]*)\)")


def grant_path_resolution(folder: Path, *, principal_sid: str | None = None) -> None:
    """Add the read-attributes entry on each profile folder above ``folder``."""

    between = _folders_between_profile_and(folder)
    if not between:
        return
    sid = principal_sid or account_sid(SANDBOX_USERNAME)
    _, granted = _recorded()
    for ancestor in between:
        _add_entry(ancestor, sid)
        granted.add(str(ancestor))
    _record(sid, granted)


def revoke_path_resolution(*, principal_sid: str | None = None) -> None:
    """Remove every entry ``grant_path_resolution`` added.

    Uses the SID recorded at grant time, so it works even after the account
    has been deleted and its name no longer resolves.
    """

    recorded_sid, granted = _recorded()
    sid = principal_sid or recorded_sid
    if not sid or not granted:
        return
    entry = _entry(sid)
    for folder in sorted(granted):
        path = Path(folder)
        if path.is_dir():
            sddl = read_dacl(path)
            if entry in sddl:
                write_dacl(path, sddl.replace(entry, ""))
    _record_path().unlink(missing_ok=True)


def _folders_between_profile_and(folder: Path) -> list[Path]:
    home = Path.home().resolve()
    target = folder.resolve()
    if not target.is_relative_to(home) or target == home:
        return []
    return [
        ancestor
        for ancestor in reversed(target.parents)
        if ancestor.is_relative_to(home) and ancestor != home
    ]


def _entry(sid: str) -> str:
    """The entry exactly as Windows writes it back.

    Windows renders well-known SIDs as short aliases (``S-1-5-32-545`` reads
    back as ``BU``), so the entry is round-tripped through Windows' own
    conversion before it is searched for or removed.
    """

    descriptor = ctypes.c_void_p()
    advapi32 = _advapi32()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        f"D:(A;;{_RIGHTS};;;{sid})", _SDDL_REVISION_1, ctypes.byref(descriptor), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    text = wintypes.LPWSTR()
    try:
        if not advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor, _SDDL_REVISION_1, _DACL_SECURITY_INFORMATION, ctypes.byref(text), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return text.value.removeprefix("D:")
    finally:
        ctypes.windll.kernel32.LocalFree(descriptor)
        if text:
            ctypes.windll.kernel32.LocalFree(text)


def _add_entry(folder: Path, sid: str) -> None:
    sddl = read_dacl(folder)
    entry = _entry(sid)
    if entry in sddl:
        return
    write_dacl(folder, _insert_explicit(sddl, entry))


def _insert_explicit(sddl: str, entry: str) -> str:
    """Put an allow entry after explicit denies and before every inherited entry.

    That is Windows' canonical order; out of it, access checks can misbehave.
    """

    matches = list(_ACE.finditer(sddl))
    position = matches[0].start() if matches else len(sddl)
    for match in matches:
        kind, flags = match.group(1).split(";")[:2]
        if "ID" in flags or kind != "D":
            break
        position = match.end()
    return sddl[:position] + entry + sddl[position:]


def read_dacl(path: Path) -> str:
    """The folder's DACL as SDDL."""

    advapi32 = _advapi32()
    needed = wintypes.DWORD(0)
    advapi32.GetFileSecurityW(str(path), _DACL_SECURITY_INFORMATION, None, 0, ctypes.byref(needed))
    buffer = ctypes.create_string_buffer(needed.value)
    if not advapi32.GetFileSecurityW(
        str(path), _DACL_SECURITY_INFORMATION, buffer, needed, ctypes.byref(needed)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    text = wintypes.LPWSTR()
    if not advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
        buffer, _SDDL_REVISION_1, _DACL_SECURITY_INFORMATION, ctypes.byref(text), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return text.value
    finally:
        ctypes.windll.kernel32.LocalFree(text)


def write_dacl(path: Path, sddl: str) -> None:
    """Set the folder's DACL, and only that folder's: nothing below is touched.

    ``SetFileSecurity`` drops the DACL's auto-inherited and protected marks
    unless the descriptor also asks for auto-inheritance. Asked, Windows keeps
    the marks and recomputes this folder's inherited entries from its parent
    -- this folder only, no walk through the files below.
    """

    advapi32 = _advapi32()
    descriptor = ctypes.c_void_p()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, _SDDL_REVISION_1, ctypes.byref(descriptor), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        flags = sddl.partition("(")[0].removeprefix("D:")
        control = _SE_DACL_AUTO_INHERIT_REQ
        control |= _SE_DACL_AUTO_INHERITED if "AI" in flags else 0
        control |= _SE_DACL_PROTECTED if "P" in flags else 0
        if not advapi32.SetSecurityDescriptorControl(descriptor, _CONTROL_MASK, control):
            raise ctypes.WinError(ctypes.get_last_error())
        if not advapi32.SetFileSecurityW(str(path), _DACL_SECURITY_INFORMATION, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        ctypes.windll.kernel32.LocalFree(descriptor)


def account_sid(name: str) -> str:
    """The account's SID as a string, e.g. ``S-1-5-21-...``."""

    advapi32 = _advapi32()
    sid_size, domain_size = wintypes.DWORD(0), wintypes.DWORD(0)
    use = wintypes.DWORD()
    sizes = (ctypes.byref(sid_size), ctypes.byref(domain_size), ctypes.byref(use))
    advapi32.LookupAccountNameW(None, name, None, sizes[0], None, sizes[1], sizes[2])
    sid = ctypes.create_string_buffer(sid_size.value)
    domain = ctypes.create_unicode_buffer(domain_size.value)
    if not advapi32.LookupAccountNameW(None, name, sid, sizes[0], domain, sizes[1], sizes[2]):
        raise ctypes.WinError(ctypes.get_last_error())
    text = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return text.value
    finally:
        ctypes.windll.kernel32.LocalFree(text)


def _advapi32():
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.GetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    advapi32.SetSecurityDescriptorControl.argtypes = [
        ctypes.c_void_p,
        wintypes.WORD,
        wintypes.WORD,
    ]
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.c_void_p,
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    return advapi32


def _record_path() -> Path:
    return Path(client_settings.profile_root) / "sandbox" / "path-grants.json"


def _recorded() -> tuple[str | None, set[str]]:
    try:
        payload = json.loads(_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, set()
    return payload.get("sid"), set(payload.get("folders") or [])


def _record(sid: str, folders: set[str]) -> None:
    path = _record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sid": sid, "folders": sorted(folders)}), encoding="utf-8")
