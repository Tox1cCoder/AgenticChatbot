"""Start one program as the sandbox account, on this process's stdio.

    python -m client_backend.services.sandbox.launcher --cwd DIR
        [--profile-root DIR] -- PROGRAM [ARG ...]

The MCP SDK spawns this as the signed-in user, exactly as it would spawn the
server itself; this starts the server as the sandbox account on the same pipes
and waits. The MCP SDK's own kill-on-close job ends this process, and this
process's job ends the server, so no sandboxed process outlives the session.

The MCP SDK hands this process only a dozen of the sidecar's variables, so the
sidecar's profile root arrives as ``--profile-root``; read from the
environment, it could name a profile holding no sandbox credentials. When the
start fails, the reason, with the program's last lines of stderr, is written
to ``launch_report`` as well as to stderr, which the sidecar cannot capture.

Variables for the program arrive in this process's environment, under
``ENV_VARIABLE``, not as arguments, and leave it in the program's environment
block, never on a command line: a command line is readable by administrators
and recorded by process-audit logs. Stdin is not an option, because it is the
MCP pipe the program itself talks on.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import ntpath
import os
import re
import subprocess
import sys
import threading
from collections.abc import Mapping

_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ENV_VARIABLE = "KANI_SANDBOX_ENV"


def encode_environment(env: dict[str, str]) -> str:
    """``env`` as one opaque value for ``ENV_VARIABLE``.

    Base64 because the MCP adapter expands ``${NAME}`` in environment values;
    the encoded text can never contain one, so every value arrives verbatim.
    """

    return base64.b64encode(json.dumps(env).encode("utf-8")).decode("ascii")


def decode_environment(encoded: str) -> dict[str, str]:
    """The variables ``encode_environment`` packed; ``{}`` for an empty value."""

    if not encoded:
        return {}
    env = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
    if not isinstance(env, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in env.items()
    ):
        raise ValueError(f"{ENV_VARIABLE} must hold a map of names to string values")
    return env


def build_command_line(argv: list[str]) -> str:
    """The command line that starts ``argv`` directly, with no shell in between."""

    if any("\0" in argument for argument in argv):
        raise ValueError("An argument contains a NUL character")
    return subprocess.list2cmdline(argv)


def profile_variables(profile: str) -> dict[str, str]:
    """The variables Windows derives from a profile folder when it loads the profile.

    ``CreateEnvironmentBlock`` fills these from the account's registry hive, which
    is loaded only while the account is signed in; before that it falls back to
    the Default user's folder and the system TEMP. The launcher cannot load the
    hive itself (that needs administrator privileges), so it sets them from the
    account's real profile folder instead, as a fresh local account has them.
    """

    drive, path = ntpath.splitdrive(profile)
    local = ntpath.join(profile, "AppData", "Local")
    return {
        "USERPROFILE": profile,
        "HOMEDRIVE": drive,
        "HOMEPATH": path or "\\",
        "APPDATA": ntpath.join(profile, "AppData", "Roaming"),
        "LOCALAPPDATA": local,
        "TEMP": ntpath.join(local, "Temp"),
        "TMP": ntpath.join(local, "Temp"),
    }


def overlay_environment(base: Mapping[str, str], added: Mapping[str, str]) -> dict[str, str]:
    """``base`` with ``added`` on top. Names compare case-insensitively, as on Windows."""

    for name, value in added.items():
        if not _VARIABLE_NAME.match(name):
            raise ValueError(f"Environment variable name {name!r} is not allowed")
        if "\0" in value:
            raise ValueError(f"Environment value for {name} contains a NUL character")
    merged = {name.upper(): (name, value) for name, value in base.items()}
    merged.update({name.upper(): (name, value) for name, value in added.items()})
    return dict(merged.values())


def environment_block(env: Mapping[str, str]) -> str:
    """``env`` as a Windows Unicode environment block: sorted, NUL-separated, NUL-ended."""

    entries = sorted(env.items(), key=lambda item: item[0].upper())
    return "".join(f"{name}={value}\0" for name, value in entries) + "\0"


class StderrRelay:
    """A pipe for the program's stderr: passed through to ours, its tail kept.

    The tail is what the failure report carries; everything still reaches
    this process's stderr, the sidecar's console, as before.
    """

    def __init__(self, tail_lines: int) -> None:
        self._read_fd, self._write_fd = os.pipe()
        os.set_inheritable(self._write_fd, True)
        self._tail: collections.deque[str] = collections.deque(maxlen=tail_lines)
        self._thread = threading.Thread(target=self._relay, name="stderr-relay", daemon=True)
        self._thread.start()

    @property
    def handle(self) -> int:
        """The pipe's write end, as the inheritable Windows handle the program gets."""

        import msvcrt  # Windows only, like everything this launcher starts

        return msvcrt.get_osfhandle(self._write_fd)

    def close(self, timeout: float = 2.0) -> list[str]:
        """Stop relaying once the program is gone, and return the tail."""

        os.close(self._write_fd)
        self._thread.join(timeout)
        return list(self._tail)

    def _relay(self) -> None:
        pending = b""
        with os.fdopen(self._read_fd, "rb", buffering=0) as pipe:
            while chunk := pipe.read(4096):
                sys.stderr.buffer.write(chunk)
                sys.stderr.flush()
                *lines, pending = (pending + chunk).split(b"\n")
                self._tail.extend(_decode(line) for line in lines)
        if pending:
            self._tail.append(_decode(pending))


def _decode(line: bytes) -> str:
    return line.decode("utf-8", errors="replace").rstrip("\r")


def _fail(code: int, message: str, stderr: list[str] | None = None) -> int:
    """Say why the start failed, on stderr and in the report the sidecar reads."""

    from client_backend.services.sandbox.launch_report import write_launch_failure

    print(message, file=sys.stderr)
    try:
        write_launch_failure(message, stderr or [])
    except Exception as exc:  # the settings themselves may be what failed
        print(f"Could not record the launch failure: {exc!r}", file=sys.stderr)
    return code


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sandbox-launcher")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--profile-root")
    parser.add_argument("program", nargs=argparse.REMAINDER)
    options = parser.parse_args(arguments)
    argv = options.program[1:] if options.program[:1] == ["--"] else options.program
    if not argv:
        parser.error("a program to run is required after --")
    if options.profile_root:
        # Before anything reads the settings, which are built on first use.
        os.environ["CLIENT_PROFILE_ROOT"] = options.profile_root

    try:
        return _launch(argv, options.cwd)
    except Exception as exc:
        _fail(1, f"The sandbox launcher failed unexpectedly: {exc!r}")
        raise


def _launch(argv: list[str], cwd: str) -> int:
    from client_backend.services.sandbox.account import load_credentials
    from client_backend.services.sandbox.launch_report import STDERR_TAIL_LINES
    from client_backend.services.sandbox.windows import (
        account_environment,
        explain_windows_error,
        run_as_user,
    )

    credentials = load_credentials()
    if credentials is None:
        return _fail(
            127,
            "The sandbox account is not set up for this profile. "
            "Run: python -m client_backend sandbox setup",
        )
    try:
        added = decode_environment(os.environ.pop(ENV_VARIABLE, ""))
        command_line = build_command_line(argv)
        windows_env, profile = account_environment(credentials.username, credentials.password)
        account_env = overlay_environment(windows_env, profile_variables(profile))
        env = overlay_environment(account_env, added)
    except ValueError as exc:
        return _fail(126, f"Could not start the program as {credentials.username}: {exc}")
    except OSError as exc:
        reason = explain_windows_error(exc)
        return _fail(126, f"Could not sign in as {credentials.username}: {reason}")

    relay = StderrRelay(STDERR_TAIL_LINES)
    try:
        code = run_as_user(
            credentials.username,
            credentials.password,
            command_line,
            cwd,
            environment_block(env),
            stderr_handle=relay.handle,
        )
    except OSError as exc:
        relay.close()
        reason = explain_windows_error(exc)
        return _fail(126, f"Could not start the program as {credentials.username}: {reason}")
    stderr = relay.close()
    if code != 0:
        return _fail(code, f"{argv[0]} exited with code {code} as {credentials.username}", stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
