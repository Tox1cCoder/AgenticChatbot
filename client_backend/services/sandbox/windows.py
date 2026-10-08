"""The Win32 calls the sandbox needs, through ctypes.

Kept in one module so the rest of the sandbox reads as ordinary Python, and so
nothing outside Windows imports ``ctypes.windll``.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from ctypes import wintypes

from client_backend.core.windows_job import (
    EXTENDED_LIMIT_INFORMATION,
    KILL_ON_JOB_CLOSE,
    JobExtendedLimitInformation,
)

_SEE_MASK_NOCLOSEPROCESS = 0x00000040
_SEE_MASK_NOASYNC = 0x00000100
_SW_HIDE = 0
_INFINITE = 0xFFFFFFFF
_ERROR_CANCELLED = 1223
_ERROR_INSUFFICIENT_BUFFER = 122
_LOGON32_LOGON_INTERACTIVE = 2
_LOGON32_PROVIDER_DEFAULT = 0


class ElevationDeclinedError(RuntimeError):
    """The user answered No to the Windows administrator prompt."""


class _ShellExecuteInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("fMask", ctypes.c_ulong),
        ("hwnd", wintypes.HWND),
        ("lpVerb", wintypes.LPCWSTR),
        ("lpFile", wintypes.LPCWSTR),
        ("lpParameters", wintypes.LPCWSTR),
        ("lpDirectory", wintypes.LPCWSTR),
        ("nShow", ctypes.c_int),
        ("hInstApp", wintypes.HINSTANCE),
        ("lpIDList", ctypes.c_void_p),
        ("lpClass", wintypes.LPCWSTR),
        ("hkeyClass", wintypes.HKEY),
        ("dwHotKey", wintypes.DWORD),
        ("hIconOrMonitor", wintypes.HANDLE),
        ("hProcess", wintypes.HANDLE),
    ]


def run_elevated(arguments: list[str]) -> int:
    """Run ``python -m client_backend <arguments>`` as administrator and wait for it.

    Windows shows its administrator prompt; the elevated process runs hidden,
    as the same user, so DPAPI data it writes stays readable here.
    """

    info = _ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = _SEE_MASK_NOCLOSEPROCESS | _SEE_MASK_NOASYNC
    info.lpVerb = "runas"
    info.lpFile = sys.executable
    info.lpParameters = subprocess.list2cmdline(["-m", "client_backend", *arguments])
    info.lpDirectory = os.getcwd()
    info.nShow = _SW_HIDE
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        error = ctypes.get_last_error()
        if error == _ERROR_CANCELLED:
            raise ElevationDeclinedError()
        raise ctypes.WinError(error)

    kernel32 = ctypes.windll.kernel32
    try:
        kernel32.WaitForSingleObject(info.hProcess, _INFINITE)
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(exit_code)):
            raise ctypes.WinError()
        return int(exit_code.value)
    finally:
        kernel32.CloseHandle(info.hProcess)


_LOGON_WITH_PROFILE = 0x00000001
_CREATE_SUSPENDED = 0x00000004
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_STARTF_USESHOWWINDOW = 0x00000001
_STARTF_USESTDHANDLES = 0x00000100
_STD_HANDLES = (-10, -11, -12)  # input, output, error


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


def account_environment(username: str, password: str) -> tuple[dict[str, str], str]:
    """The environment Windows builds for the account's token, and its profile folder.

    The account's registry hive is not loaded here, so the profile-derived
    variables in it are the Default user's; the caller corrects them from the
    folder (see ``launcher.profile_variables``).
    """

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    userenv = _userenv()
    token = wintypes.HANDLE()
    if not advapi32.LogonUserW(
        username,
        ".",
        password,
        _LOGON32_LOGON_INTERACTIVE,
        _LOGON32_PROVIDER_DEFAULT,
        ctypes.byref(token),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        profile = _account_profile(userenv, token, username, password)
        return _token_environment(userenv, token), profile
    finally:
        ctypes.windll.kernel32.CloseHandle(token)


def _userenv():
    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    userenv.CreateEnvironmentBlock.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        wintypes.HANDLE,
        wintypes.BOOL,
    ]
    userenv.DestroyEnvironmentBlock.argtypes = [ctypes.c_void_p]
    userenv.GetUserProfileDirectoryW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    return userenv


def _token_environment(userenv, token) -> dict[str, str]:
    """``CreateEnvironmentBlock`` for ``token``, without this process's variables."""

    block = ctypes.c_void_p()
    if not userenv.CreateEnvironmentBlock(ctypes.byref(block), token, False):
        raise ctypes.WinError(ctypes.get_last_error())
    env: dict[str, str] = {}
    try:
        address = block.value
        while entry := ctypes.wstring_at(address):
            address += (len(entry) + 1) * ctypes.sizeof(ctypes.c_wchar)
            # "=C:=C:\\..." entries are per-drive working directories, not variables.
            if entry.startswith("="):
                continue
            name, separator, value = entry.partition("=")
            if separator:
                env[name] = value
    finally:
        userenv.DestroyEnvironmentBlock(block)
    return env


def _account_profile(userenv, token, username: str, password: str) -> str:
    """The account's profile folder, created first if the account has none yet.

    ``LogonUser`` does not create a profile, so a freshly set-up account has
    none until something signs in with one. Starting a throwaway program the
    way real launches do (``LOGON_WITH_PROFILE``) makes Windows create it.
    """

    try:
        return _profile_directory(userenv, token)
    except OSError:
        print(
            f"Creating the Windows profile for {username}; the first launch takes a moment.",
            file=sys.stderr,
        )
        _create_profile(username, password)
    try:
        return _profile_directory(userenv, token)
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"The account has no Windows profile and signing in once did not create one: "
            f"{exc.strerror}",
            None,
            exc.winerror,
        ) from exc


def _profile_directory(userenv, token) -> str:
    """The profile folder, or the error that says why there is none.

    The first call only asks for the size; any failure other than "buffer too
    small" is the real error. Ignoring it sized the buffer from a zero and
    reported "the data area passed to a system call is too small" instead.
    """

    size = wintypes.DWORD(0)
    if not userenv.GetUserProfileDirectoryW(token, None, ctypes.byref(size)):
        error = ctypes.get_last_error()
        if error != _ERROR_INSUFFICIENT_BUFFER or size.value == 0:
            raise ctypes.WinError(error)
    buffer = ctypes.create_unicode_buffer(size.value)
    if not userenv.GetUserProfileDirectoryW(token, buffer, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.value


def _create_profile(username: str, password: str) -> None:
    """Start and wait for a program that does nothing, as the account, with its profile."""

    system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
    cmd = os.path.join(system_root, "System32", "cmd.exe")
    environment = f"SystemRoot={system_root}\0\0"
    run_as_user(username, password, f'"{cmd}" /c exit 0', system_root, environment)


def run_as_user(
    username: str,
    password: str,
    command_line: str,
    cwd: str,
    environment: str,
    *,
    stderr_handle: int | None = None,
) -> int:
    """Run a command line as another local account, on this process's stdio, and wait.

    The child gets exactly ``environment`` (a Unicode environment block) and
    this process's standard handles, so whoever holds our pipes talks to it
    directly; ``stderr_handle``, an inheritable handle, replaces its stderr.
    It starts suspended inside a kill-on-close job and only then runs: when
    this process exits, for any reason, the child and everything it started
    go too.
    """

    kernel32 = _kernel32()
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    job = _kill_on_close_job(kernel32)

    startup = _StartupInfo()
    startup.cb = ctypes.sizeof(startup)
    startup.dwFlags = _STARTF_USESTDHANDLES | _STARTF_USESHOWWINDOW
    startup.wShowWindow = _SW_HIDE
    startup.hStdInput, startup.hStdOutput, startup.hStdError = (
        kernel32.GetStdHandle(handle) for handle in _STD_HANDLES
    )
    if stderr_handle is not None:
        startup.hStdError = stderr_handle
    process = _ProcessInformation()
    command_buffer = ctypes.create_unicode_buffer(command_line)
    # Raw UTF-16 bytes: a str buffer would stop at the first of the block's NULs.
    block = environment.encode("utf-16-le")
    environment_buffer = ctypes.create_string_buffer(block, len(block))
    if not advapi32.CreateProcessWithLogonW(
        username,
        ".",
        password,
        _LOGON_WITH_PROFILE,
        None,
        command_buffer,
        _CREATE_SUSPENDED | _CREATE_UNICODE_ENVIRONMENT,
        environment_buffer,
        cwd,
        ctypes.byref(startup),
        ctypes.byref(process),
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise ctypes.WinError(error)
    try:
        if not kernel32.AssignProcessToJobObject(job, process.hProcess):
            error = ctypes.get_last_error()
            kernel32.TerminateProcess(process.hProcess, 1)
            raise ctypes.WinError(error)
        kernel32.ResumeThread(process.hThread)
        kernel32.WaitForSingleObject(process.hProcess, _INFINITE)
        exit_code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(process.hProcess, ctypes.byref(exit_code))
        return int(exit_code.value)
    finally:
        kernel32.CloseHandle(process.hThread)
        kernel32.CloseHandle(process.hProcess)
        # Closing the job ends anything the child left running.
        kernel32.CloseHandle(job)


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = wintypes.HANDLE
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetStdHandle.restype = handle
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = handle
    kernel32.SetInformationJobObject.argtypes = [
        handle,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [handle, handle]
    kernel32.ResumeThread.argtypes = [handle]
    kernel32.TerminateProcess.argtypes = [handle, wintypes.UINT]
    kernel32.WaitForSingleObject.argtypes = [handle, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [handle, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [handle]
    return kernel32


def _kill_on_close_job(kernel32):
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = JobExtendedLimitInformation()
    limits.basic_limit_information.limit_flags = KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise ctypes.WinError(error)
    return job


# What to do about the errors that stop a program starting as the account.
_ERROR_ADVICE = {
    1058: (
        "The Secondary Logon service (seclogon) is disabled, and Windows needs it to "
        "start a program as another account. As administrator, run: "
        "sc config seclogon start= demand"
    ),
    1326: (
        "The stored password no longer matches the account. Run: "
        "python -m client_backend sandbox setup"
    ),
    1331: "The account is disabled. Run: python -m client_backend sandbox setup",
    1385: (
        "Windows policy does not let the account sign in locally (the 'Allow log on "
        "locally' / 'Deny log on locally' rights); an administrator has to allow it"
    ),
}


def explain_windows_error(error: OSError) -> str:
    """The error as Windows words it, plus what to do when that is known."""

    code = getattr(error, "winerror", None)
    text = (error.strerror or str(error)).strip()
    described = f"[WinError {code}] {text}" if code else text
    advice = _ERROR_ADVICE.get(code) if code else None
    return f"{described}. {advice}" if advice else described


def logon_problem(username: str, password: str) -> str | None:
    """Why Windows would refuse to start a process as this account, or None."""

    token = wintypes.HANDLE()
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    if advapi32.LogonUserW(
        username,
        ".",
        password,
        _LOGON32_LOGON_INTERACTIVE,
        _LOGON32_PROVIDER_DEFAULT,
        ctypes.byref(token),
    ):
        ctypes.windll.kernel32.CloseHandle(token)
        return None
    return explain_windows_error(ctypes.WinError(ctypes.get_last_error()))
