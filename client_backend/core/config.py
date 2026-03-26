"""
Client backend configuration settings.

All settings can be overridden via environment variables.
"""

import os
import platform
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, DotEnvSettingsSource, PydanticBaseSettingsSource

# Determine platform-appropriate default paths
_IS_WINDOWS = platform.system() == "Windows"
_DEFAULT_PROFILE_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", "~")) / "CodexDesktop"
    if _IS_WINDOWS
    else Path("~/.config/codex-desktop")
)


def _load_or_create_local_secret(secret_path: Path) -> str:
    """Load a persisted local secret, creating one on first run."""
    if secret_path.exists():
        secret = secret_path.read_text(encoding="utf-8").strip()
        if secret:
            return secret

    secret = secrets.token_urlsafe(48)
    secret_path.write_text(secret, encoding="utf-8")
    return secret


class ClientSettings(BaseSettings):
    """Configuration settings for the client backend."""

    model_config = {
        "env_prefix": "CLIENT_",
        "env_file_encoding": "utf-8",
        "enable_decoding": False,
    }

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        env_file = os.environ.get("CLIENT_ENV_FILE", ".env.client")
        return (
            init_settings,
            env_settings,
            DotEnvSettingsSource(
                settings_cls,
                env_file=env_file,
                env_file_encoding="utf-8",
            ),
            file_secret_settings,
        )

    # Server Connection
    server_api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for the canonical server backend API.",
    )
    server_api_timeout_seconds: int = Field(
        default=60,
        description="Timeout for server API requests in seconds.",
    )

    # Local Backend Binding
    backend_host: str = Field(
        default="127.0.0.1",
        description="Host to bind the client backend to. Default loopback only for security.",
    )
    backend_port: int = Field(
        default=8100,
        description="Port for the client backend API.",
    )

    # Profile and Storage
    profile_root: str = Field(
        default=str(_DEFAULT_PROFILE_ROOT.expanduser()),
        description="Root directory for local profile storage.",
    )
    device_name: str = Field(
        default=platform.node() or "UnknownDevice",
        description="Display name for this device.",
    )

    # MCP Configuration
    mcp_config_path: str = Field(
        default="",
        description="Path to the MCP JSON configuration file. Empty means use default in profile.",
    )
    mcp_startup_timeout_seconds: int = Field(
        default=30,
        description="Timeout for MCP server startup in seconds.",
    )

    # Skills Configuration
    skills_roots: list[str] = Field(
        default=[],
        description="List of directory paths to scan for local skills.",
    )

    # Workspace Configuration
    workspace_roots: list[str] = Field(
        default=[],
        description="Allowed workspace root directories for filesystem operations.",
    )

    # Shell Configuration
    allowed_shells: list[str] = Field(
        default=["bash", "sh", "cmd", "powershell"] if _IS_WINDOWS else ["bash", "sh", "zsh"],
        description="List of allowed shells for script execution.",
    )
    shell_timeout_seconds: int = Field(
        default=60,
        description="Default timeout for shell command execution.",
    )
    shell_max_output_bytes: int = Field(
        default=1048576,
        description="Maximum stdout/stderr size for shell commands (1MB default).",
    )

    # Runtime Configuration
    heartbeat_interval_seconds: int = Field(
        default=30,
        description="Interval for sending heartbeats to the server.",
    )
    reconnect_delay_seconds: int = Field(
        default=5,
        description="Delay before attempting to reconnect to the server.",
    )
    max_reconnect_attempts: int = Field(
        default=10,
        description="Maximum reconnection attempts before giving up.",
    )

    # Security
    local_session_secret: str = Field(
        default="",
        description="Secret key for local session tokens. Auto-generated if empty.",
    )
    local_session_expire_minutes: int = Field(
        default=1440,
        description="Local session expiration in minutes (24 hours default).",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    log_to_file: bool = Field(
        default=True,
        description="Enable logging to file in profile directory.",
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production).",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("workspace_roots", "skills_roots", mode="before")
    @classmethod
    def _parse_path_list(cls, v):
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    @field_validator("profile_root", mode="before")
    @classmethod
    def _normalize_profile_root(cls, v: str) -> str:
        if v is None or not str(v).strip():
            return str(_DEFAULT_PROFILE_ROOT.expanduser())
        return str(Path(v).expanduser())

    @field_validator("device_name", mode="before")
    @classmethod
    def _normalize_device_name(cls, v: str) -> str:
        if v is None or not str(v).strip():
            return platform.node() or "UnknownDevice"
        return str(v).strip()

    @field_validator("mcp_config_path", mode="before")
    @classmethod
    def _normalize_optional_path(cls, v: str) -> str:
        if not v:
            return ""
        return str(Path(v).expanduser())

    @field_validator("workspace_roots", "skills_roots", mode="after")
    @classmethod
    def _normalize_path_list(cls, values: list[str]) -> list[str]:
        return [str(Path(value).expanduser()) for value in values]

    @field_validator("allowed_shells", mode="before")
    @classmethod
    def _parse_shell_list(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("allowed_shells", mode="after")
    @classmethod
    def _normalize_shell_list(cls, values: list[str]) -> list[str]:
        return [value.lower() for value in values]

    @model_validator(mode="after")
    def _post_validate(self) -> "ClientSettings":
        # Ensure profile root exists
        profile_path = Path(self.profile_root)
        profile_path.mkdir(parents=True, exist_ok=True)

        # Generate and persist the local session secret if not provided.
        if not self.local_session_secret:
            self.local_session_secret = _load_or_create_local_secret(
                profile_path / ".local_session_secret"
            )

        return self

    def get_profile_path(self, *subpaths: str) -> Path:
        """Get a path within the profile directory."""
        return Path(self.profile_root).joinpath(*subpaths)

    def get_mcp_config_path(self) -> Path:
        """Get the effective MCP config file path."""
        if self.mcp_config_path:
            return Path(self.mcp_config_path)
        return self.get_profile_path("mcp_config.json")


@lru_cache
def get_client_settings() -> ClientSettings:
    """Get client settings with caching."""
    return ClientSettings()


# Global settings instance
client_settings = get_client_settings()
