"""
Shell runner service for executing local shell commands.

Provides subprocess execution with timeout, output capture, and basic safety controls.
"""

import asyncio
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import validate_workspace_path
from client_backend.core.security import redact_env_for_audit, redact_path_for_audit

logger = get_logger(__name__)


class ShellExecutionResult:
    """Result of a shell command execution."""

    def __init__(
        self,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        execution_time_ms: int,
        truncated: bool = False,
        working_dir: str | None = None,
    ):
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time_ms = execution_time_ms
        self.truncated = truncated
        self.working_dir = working_dir

    @property
    def success(self) -> bool:
        """Check if the command succeeded."""
        return self.exit_code == 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time_ms": self.execution_time_ms,
            "truncated": self.truncated,
            "success": self.success,
            "working_dir": self.working_dir,
        }


class ShellExecutionError(Exception):
    """Raised when shell execution fails."""

    def __init__(self, message: str, exit_code: int | None = None):
        super().__init__(message)
        self.exit_code = exit_code


class ShellRunnerService:
    """
    Service for executing shell commands with basic safety controls.

    Features:
    - Timeout enforcement
    - Output size limits
    - Optional working directory validation
    - Shell allowlist
    - Environment variable filtering
    - Audit logging
    """

    def __init__(self):
        self.allowed_shells = client_settings.allowed_shells
        self.default_timeout = client_settings.shell_timeout_seconds
        self.max_output_bytes = client_settings.shell_max_output_bytes
        self._base_env_allowlist = {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "COMSPEC",
            "WINDIR",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
            "HOMEDRIVE",
            "HOMEPATH",
            "LANG",
            "TERM",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
            "PROGRAMDATA",
            "PROGRAMFILES",
            "PROGRAMFILES(X86)",
            "LOCALAPPDATA",
            "APPDATA",
        }

    async def execute(
        self,
        command: str | list[str],
        working_dir: str | Path | None = None,
        timeout: int | None = None,
        shell: str = "bash",
        env: dict[str, str] | None = None,
        capture_output: bool = True,
    ) -> ShellExecutionResult:
        """
        Execute a shell command.

        Args:
            command: Command string or list of arguments.
            working_dir: Optional working directory.
            timeout: Timeout in seconds (defaults to config).
            shell: Shell to use (must be in allowed list).
            env: Additional environment variables.
            capture_output: Whether to capture stdout/stderr.

        Returns:
            ShellExecutionResult with command output and metadata.

        Raises:
            ShellExecutionError: If execution fails.
            ValueError: If parameters are invalid.
        """
        # Validate shell
        if shell not in self.allowed_shells:
            raise ValueError(f"Shell '{shell}' not allowed. Allowed: {self.allowed_shells}")

        cwd: str | None = None
        if working_dir:
            try:
                resolved_cwd = validate_workspace_path(working_dir)
            except Exception as e:
                raise ValueError(f"Invalid working directory: {e}")

            if not resolved_cwd.exists():
                raise ValueError(f"Working directory does not exist: {working_dir}")
            if not resolved_cwd.is_dir():
                raise ValueError(f"Working directory is not a directory: {working_dir}")
            cwd = str(resolved_cwd)

        # Prepare timeout
        exec_timeout = timeout if timeout is not None else self.default_timeout

        # Prepare environment
        exec_env = self._build_execution_env()
        if env:
            exec_env.update({str(key): str(value) for key, value in env.items()})

        # Log execution (with redacted paths and env)
        audit_command = command if isinstance(command, str) else " ".join(command)
        audit_dir = redact_path_for_audit(cwd) if cwd else None
        audit_env = redact_env_for_audit(env) if env else {}

        logger.info(
            f"Executing shell command: {audit_command[:100]} "
            f"(cwd={audit_dir}, timeout={exec_timeout}s)"
        )
        logger.debug(f"Environment overrides: {audit_env}")

        # Execute
        start_time = datetime.now(timezone.utc)

        try:
            # Parse command if it's a string
            cmd_args = self._build_command_args(command, shell)

            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                cwd=cwd,
                env=exec_env,
            )

            # Wait for completion with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=exec_timeout,
                )
            except asyncio.TimeoutError:
                # Kill the process
                process.kill()
                await process.wait()
                raise ShellExecutionError(
                    f"Command timed out after {exec_timeout}s",
                    exit_code=-1,
                )

            # Measure execution time
            end_time = datetime.now(timezone.utc)
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Decode output
            stdout_str = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr_str = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            # Check output size and truncate if needed
            truncated = False
            total_size = len(stdout_str) + len(stderr_str)

            if total_size > self.max_output_bytes:
                truncated = True
                available_bytes = self.max_output_bytes
                stdout_portion = int(available_bytes * 0.7)  # Give more to stdout
                stderr_portion = available_bytes - stdout_portion

                stdout_str = stdout_str[:stdout_portion]
                stderr_str = stderr_str[:stderr_portion]

                logger.warning(
                    f"Command output truncated: {total_size} -> {self.max_output_bytes} bytes"
                )

            result = ShellExecutionResult(
                command=audit_command,
                exit_code=process.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time_ms=execution_time_ms,
                truncated=truncated,
                working_dir=audit_dir,
            )

            if not result.success:
                logger.warning(
                    f"Command failed with exit code {result.exit_code}: {audit_command[:50]}"
                )
            else:
                logger.debug(f"Command succeeded ({execution_time_ms}ms): {audit_command[:50]}")

            return result

        except asyncio.CancelledError:
            logger.warning(f"Command execution cancelled: {audit_command[:50]}")
            raise ShellExecutionError("Command execution was cancelled", exit_code=-2)

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise ShellExecutionError(f"Execution error: {e}")

    async def execute_script(
        self,
        script_content: str,
        working_dir: str | Path | None = None,
        timeout: int | None = None,
        shell: str = "bash",
        env: dict[str, str] | None = None,
    ) -> ShellExecutionResult:
        """
        Execute a multi-line script.

        Args:
            script_content: The script content to execute.
            working_dir: Optional working directory.
            timeout: Timeout in seconds (defaults to config).
            shell: Shell to use (must be in allowed list).
            env: Additional environment variables.

        Returns:
            ShellExecutionResult with script output and metadata.
        """
        return await self.execute(
            command=script_content,
            working_dir=working_dir,
            timeout=timeout,
            shell=shell,
            env=env,
            capture_output=True,
        )

    def validate_command(
        self,
        command: str,
        *,
        working_dir: str | Path | None = None,
    ) -> tuple[bool, str]:
        """
        Validate a command for potential security issues.

        Args:
            command: The command string to validate.

        Returns:
            (is_valid, message) tuple.
        """
        if not command.strip():
            return False, "Command cannot be empty"

        if working_dir:
            try:
                resolved_cwd = validate_workspace_path(working_dir)
            except Exception as exc:
                return False, f"Invalid working_dir: {exc}"

            if not resolved_cwd.exists():
                return False, f"Working directory does not exist: {working_dir}"
            if not resolved_cwd.is_dir():
                return False, f"Working directory is not a directory: {working_dir}"

        # Check for dangerous patterns
        dangerous_patterns = [
            "rm -rf /",
            "mkfs",
            "dd if=/dev/zero",
            "> /dev/sda",
            "fork bomb",
            ":(){ :|:& };:",
        ]

        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return False, f"Command contains dangerous pattern: {pattern}"

        return True, "Command appears safe"

    @staticmethod
    def _build_command_args(command: str | list[str], shell: str) -> list[str]:
        """Build subprocess arguments for the selected shell."""
        if not isinstance(command, str):
            return list(command)

        normalized_shell = shell.lower()
        if normalized_shell == "cmd":
            return [shell, "/d", "/s", "/c", command]
        if normalized_shell in {"powershell", "pwsh"}:
            return [shell, "-NoProfile", "-NonInteractive", "-Command", command]
        return [shell, "-c", command]

    def _build_execution_env(self) -> dict[str, str]:
        """Build a minimal environment for subprocess execution."""
        env: dict[str, str] = {}
        for key in self._base_env_allowlist:
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    async def test_shell_availability(self, shell: str) -> bool:
        """
        Test if a shell is available on the system.

        Args:
            shell: Shell name to test.

        Returns:
            True if available, False otherwise.
        """
        try:
            normalized_shell = shell.lower()
            if normalized_shell == "cmd":
                cmd_args = [shell, "/d", "/c", "ver"]
            elif normalized_shell in {"powershell", "pwsh"}:
                cmd_args = [shell, "-NoProfile", "-NonInteractive", "-Command", "$PSVersionTable"]
            else:
                cmd_args = [shell, "--version"]

            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=5)
            return process.returncode == 0
        except Exception:
            return False


# Global singleton
_shell_runner: ShellRunnerService | None = None


def get_shell_runner() -> ShellRunnerService:
    """Get the global shell runner service."""
    global _shell_runner
    if _shell_runner is None:
        _shell_runner = ShellRunnerService()
    return _shell_runner
