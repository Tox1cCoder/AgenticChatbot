"""Start one program as the sandbox account, on this process's stdio.

    python -m client_backend.services.sandbox.launcher \
        --cwd DIR [--env NAME=VALUE ...] -- PROGRAM [ARG ...]

The MCP SDK spawns this as the signed-in user, exactly as it would spawn the
server itself; this starts the server as the sandbox account on the same pipes
and waits. The MCP SDK's own kill-on-close job ends this process, and this
process's job ends the server, so no sandboxed process outlives the session.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Text cmd.exe would act on rather than pass through: command separators,
# redirection, its escape character, variable expansion, and quotes that
# would end the quoted command early.
_CMD_METACHARACTERS = re.compile(r'[&|<>^%"\r\n]')
_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_command_line(argv: list[str], env: dict[str, str]) -> str:
    """A ``cmd.exe`` line that sets ``env`` on top of the profile environment, then runs ``argv``.

    Anything ``cmd.exe`` would interpret is refused rather than escaped: the
    values here are fixed settings and file paths, and a refusal is safer
    than an escaping mistake in a line that runs as another account.
    """

    for name, value in env.items():
        if not _VARIABLE_NAME.match(name):
            raise ValueError(f"Environment variable name {name!r} is not allowed")
        if _CMD_METACHARACTERS.search(value):
            raise ValueError(f"Environment value for {name} contains characters cmd.exe interprets")
    for argument in argv:
        if _CMD_METACHARACTERS.search(argument):
            raise ValueError(f"Argument {argument!r} contains characters cmd.exe interprets")

    steps = [f'set "{name}={value}"' for name, value in env.items()]
    steps.append(subprocess.list2cmdline(argv))
    return f'cmd.exe /d /s /c "{"&&".join(steps)}"'


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sandbox-launcher")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("program", nargs=argparse.REMAINDER)
    options = parser.parse_args(arguments)
    argv = options.program[1:] if options.program[:1] == ["--"] else options.program
    if not argv:
        parser.error("a program to run is required after --")

    from client_backend.services.sandbox.account import load_credentials
    from client_backend.services.sandbox.windows import run_as_user

    credentials = load_credentials()
    if credentials is None:
        print(
            "The sandbox account is not set up. Run: python -m client_backend sandbox setup",
            file=sys.stderr,
        )
        return 127
    env = dict(item.split("=", 1) for item in options.env)
    try:
        command_line = build_command_line(argv, env)
        return run_as_user(credentials.username, credentials.password, command_line, options.cwd)
    except (ValueError, OSError) as exc:
        print(f"Could not start the program as {credentials.username}: {exc}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
