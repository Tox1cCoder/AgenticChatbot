"""
Client backend configuration settings.

All settings can be overridden via environment variables.
"""

import ipaddress
import os
import platform
import secrets
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, field_validator
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

    # Local Runtime Execution
    tool_call_timeout_seconds: int = Field(
        default=60,
        description="Default timeout for client-side tool dispatch when the request omits one.",
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

    # Skill Archive Uploads
    skill_upload_max_bytes: int = Field(
        default=25 * 1024 * 1024,
        description="Maximum accepted size of an uploaded skill archive in bytes.",
    )
    skill_upload_max_expanded_bytes: int = Field(
        default=100 * 1024 * 1024,
        description="Maximum total expanded size of one skill archive in bytes.",
    )
    skill_upload_max_file_bytes: int = Field(
        default=50 * 1024 * 1024,
        description="Maximum expanded size of a single archive member in bytes.",
    )
    skill_upload_max_entries: int = Field(
        default=2_000,
        description="Maximum number of entries in one skill archive.",
    )
    skill_upload_max_compression_ratio: int = Field(
        default=200,
        description="Maximum expanded-to-compressed ratio before rejecting an archive.",
    )
    skill_upload_max_path_depth: int = Field(
        default=20,
        description="Maximum number of path components in an archive member name.",
    )
    skill_upload_max_path_chars: int = Field(
        default=240,
        description="Maximum length of a portable archive member path.",
    )
    skill_upload_ttl_seconds: int = Field(
        default=1_800,
        description="Lifetime of a staged, uninstalled skill upload in seconds.",
    )
    skill_operation_receipt_ttl_seconds: int = Field(
        default=3_600,
        description="Lifetime of a terminal installation receipt in seconds.",
    )
    skill_upload_max_outstanding: int = Field(
        default=5,
        description="Maximum concurrently staged uploads per user profile.",
    )
    skill_upload_quota_bytes: int = Field(
        default=250 * 1024 * 1024,
        description="Maximum total staged upload bytes per user profile.",
    )
    skill_upload_rate_limit_count: int = Field(
        default=10,
        description="Maximum upload attempts per user within the rate-limit window.",
    )
    skill_upload_rate_limit_window_seconds: int = Field(
        default=60,
        description="Length of the upload rate-limit window in seconds.",
    )
    skill_install_lock_timeout_seconds: float = Field(
        default=10,
        description="Seconds to wait for a skill mutation lock before failing.",
    )
    skill_catalog_freshness_seconds: float = Field(
        default=1.0,
        description="How long a projected skill catalog may be reused without rescanning.",
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
    allowed_origins: list[str] = Field(
        default=[
            # AI SDK frontend
            "http://127.0.0.1:3000",
            "http://localhost:3000",
            "http://[::1]:3000",
            # Streamlit frontend
            "http://127.0.0.1:8501",
            "http://localhost:8501",
        ],
        description="Explicit browser origins allowed to call the sidecar. Never '*'.",
    )
    allow_non_loopback_backend: bool = Field(
        default=False,
        description="Opt in to binding the sidecar off loopback. Exposes local tools.",
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

    @field_validator(
        "skill_upload_max_bytes",
        "skill_upload_max_expanded_bytes",
        "skill_upload_max_file_bytes",
        "skill_upload_max_entries",
        "skill_upload_max_compression_ratio",
        "skill_upload_max_path_depth",
        "skill_upload_max_path_chars",
        "skill_upload_ttl_seconds",
        "skill_operation_receipt_ttl_seconds",
        "skill_upload_max_outstanding",
        "skill_upload_quota_bytes",
        "skill_upload_rate_limit_count",
        "skill_upload_rate_limit_window_seconds",
        "skill_install_lock_timeout_seconds",
        "skill_catalog_freshness_seconds",
        mode="after",
    )
    @classmethod
    def _require_positive_limit(cls, value: int | float, info) -> int | float:
        # A zero or negative bound would silently disable the limit it names,
        # which is the difference between "small archives only" and "any
        # archive". Fail at construction instead.
        if value <= 0:
            raise ValueError(f"{info.field_name} must be greater than zero")
        return value

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, value) -> list[str]:
        if isinstance(value, str):
            candidates = [origin.strip() for origin in value.split(",")]
        else:
            candidates = [str(origin).strip() for origin in value or []]

        origins = [origin for origin in candidates if origin]
        if not origins:
            raise ValueError("allowed_origins must list at least one explicit origin")

        for origin in origins:
            if origin == "*":
                raise ValueError(
                    "allowed_origins must not contain '*'; the sidecar exposes local "
                    "tool execution and requires explicit browser origins"
                )
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    f"allowed_origins entry '{origin}' is not a scheme://host[:port] origin"
                )
        return origins

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


def _is_loopback_host(host: str) -> bool:
    """Report whether a bind host reaches only this machine."""
    normalized = (host or "").strip().strip("[]").lower()
    if normalized in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        # A hostname we cannot resolve to a loopback literal is treated as
        # remote-reachable; the operator opts in explicitly if it is not.
        return False


def initialize_client_environment(settings: ClientSettings | None = None) -> ClientSettings:
    """Apply runtime-only filesystem setup for the client environment."""

    resolved_settings = settings or get_client_settings()
    if not _is_loopback_host(resolved_settings.backend_host) and not (
        resolved_settings.allow_non_loopback_backend
    ):
        raise ValueError(
            f"backend_host '{resolved_settings.backend_host}' is not loopback. The sidecar "
            "executes local tools and skill commands, so binding it to a reachable "
            "interface requires setting allow_non_loopback_backend=true explicitly."
        )

    profile_path = Path(resolved_settings.profile_root)
    profile_path.mkdir(parents=True, exist_ok=True)

    if not resolved_settings.local_session_secret:
        resolved_settings.local_session_secret = _load_or_create_local_secret(
            profile_path / ".local_session_secret"
        )

    return resolved_settings


class _ClientSettingsProxy:
    """Lazy proxy so importing config does not freeze a settings snapshot immediately."""

    def __getattr__(self, name: str):
        return getattr(get_client_settings(), name)

    def __setattr__(self, name: str, value):
        setattr(get_client_settings(), name, value)

    def __repr__(self) -> str:
        return repr(get_client_settings())


# Global lazy settings proxy
client_settings = _ClientSettingsProxy()
