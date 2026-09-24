"""Create, reset, or remove the sandbox account. Needs administrator rights.

The account is a standard local user in the built-in Users group (which holds
the "log on locally" right that starting a process as it requires), with a
password that never expires. Group membership goes by SID because group names
are translated on non-English Windows.

It is deliberately left visible on the sign-in screen and says what it is for:
a hidden, tool-created account is what a backdoor looks like, to the user and
to their antivirus.
"""

from __future__ import annotations

import base64
import subprocess

from client_backend.services.sandbox.account import (
    SANDBOX_USERNAME,
    SandboxCredentials,
    forget_credentials,
    generate_password,
    save_credentials,
)

_USERS_GROUP_SID = "S-1-5-32-545"
# Windows caps a local account's description at 48 characters.
_DESCRIPTION = "Runs commands for the Kani assistant"

# The password is read from stdin: a command line is visible to every process.
_PROVISION_SCRIPT = rf"""
$ErrorActionPreference = 'Stop'
$name = '{SANDBOX_USERNAME}'
$password = ConvertTo-SecureString ([Console]::In.ReadLine()) -AsPlainText -Force
if (Get-LocalUser -Name $name -ErrorAction SilentlyContinue) {{
    Set-LocalUser -Name $name -Password $password -PasswordNeverExpires $true `
        -AccountNeverExpires -Description '{_DESCRIPTION}'
    Enable-LocalUser -Name $name
}} else {{
    New-LocalUser -Name $name -Password $password -PasswordNeverExpires `
        -UserMayNotChangePassword -AccountNeverExpires `
        -FullName 'Kani assistant sandbox' -Description '{_DESCRIPTION}' | Out-Null
}}
try {{
    Add-LocalGroupMember -SID '{_USERS_GROUP_SID}' -Member $name
}} catch [Microsoft.PowerShell.Commands.MemberExistsException] {{ }}
"""

_REMOVE_SCRIPT = rf"""
$ErrorActionPreference = 'Stop'
$name = '{SANDBOX_USERNAME}'
$user = Get-LocalUser -Name $name -ErrorAction SilentlyContinue
if ($user) {{
    $sid = $user.SID.Value
    Remove-LocalUser -Name $name
    Get-CimInstance Win32_UserProfile | Where-Object {{ $_.SID -eq $sid }} | Remove-CimInstance
}}
"""


class SandboxSetupError(RuntimeError):
    """Windows refused a step of setting up or removing the sandbox account."""


def provision_account() -> SandboxCredentials:
    """Create the account, or give an existing one a new password."""

    password = generate_password()
    _run_powershell(_PROVISION_SCRIPT, stdin=password + "\n")
    save_credentials(SANDBOX_USERNAME, password)
    return SandboxCredentials(SANDBOX_USERNAME, password)


def remove_account() -> None:
    """Delete the account and its profile folder. Files it wrote elsewhere remain."""

    _run_powershell(_REMOVE_SCRIPT, stdin="")
    forget_credentials()


def _run_powershell(script: str, *, stdin: str) -> None:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise SandboxSetupError(detail or f"PowerShell exited with {completed.returncode}")
