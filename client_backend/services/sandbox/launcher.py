"""Start one program as the sandbox account, on this process's stdio.

    python -m client_backend.services.sandbox.launcher --cwd DIR -- PROGRAM [ARG ...]

The MCP SDK spawns this as the signed-in user, exactly as it would spawn the
server itself; this starts the server as the sandbox account on the same pipes
and waits. The MCP SDK's own kill-on-close job ends this process, and this
process's job ends the server, so no sandboxed process outlives the session.

Variables for the program arrive in this process's environment, under
``ENV_VARIABLE``, not as arguments, and leave it in the program's environment
block, never on a command line: a command line is readable by administrators
and recorded by process-audit logs. Stdin is not an option, because it is the
MCP pipe the program itself talks on.
"""

from __future__ import annotations

import argparse
import base64
import json
import ntpath
import os
import re
import subprocess
import sys
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


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sandbox-launcher")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("program", nargs=argparse.REMAINDER)
    options = parser.parse_args(arguments)
    argv = options.program[1:] if options.program[:1] == ["--"] else options.program
    if not argv:
        parser.error("a program to run is required after --")

    from client_backend.services.sandbox.account import load_credentials
    from client_backend.services.sandbox.windows import account_environment, run_as_user

    credentials = load_credentials()
    if credentials is None:
        print(
            "The sandbox account is not set up. Run: python -m client_backend sandbox setup",
            file=sys.stderr,
        )
        return 127
    try:
        added = decode_environment(os.environ.pop(ENV_VARIABLE, ""))
        command_line = build_command_line(argv)
        windows_env, profile = account_environment(credentials.username, credentials.password)
        account_env = overlay_environment(windows_env, profile_variables(profile))
        env = overlay_environment(account_env, added)
        return run_as_user(
            credentials.username,
            credentials.password,
            command_line,
            options.cwd,
            environment_block(env),
        )
    except (ValueError, OSError) as exc:
        print(f"Could not start the program as {credentials.username}: {exc}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
