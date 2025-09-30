from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the workspace root
dotenv_path = Path(__file__).parent.parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)


class Settings(BaseSettings):
    # Database settings
    database_url: str = Field(
        default="postgresql://postgres:123123123@localhost:5432/chatbot",
        description="Database URL for PostgreSQL connection",
        env="DATABASE_URL",
    )

    # API settings
    api_host: str = Field(
        default="0.0.0.0",
        description="API host",
        env="API_HOST",
    )
    api_port: int = Field(
        default=8000,
        description="API port",
        env="API_PORT",
    )
    api_debug: bool = Field(
        default=False,
        description="Debug mode",
        env="API_DEBUG",
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment",
        env="ENVIRONMENT",
    )

    # Security
    secret_key: str = Field(
        default="secret-key",
        description="Secret key for security",
        env="SECRET_KEY",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
        env="JWT_ALGORITHM",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration in minutes",
        env="ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration in days",
        env="REFRESH_TOKEN_EXPIRE_DAYS",
    )

    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins",
        env="CORS_ORIGINS",
    )

    # LLM API Keys
    gemini_api_key: str = Field(
        default="",
        description="Gemini API Key",
        env="GEMINI_API_KEY",
    )

    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database URL",
        env="QDRANT_URL",
    )
    qdrant_collection_name: str = Field(
        default="documents",
        description="Qdrant collection name for document storage",
        env="QDRANT_COLLECTION_NAME",
    )

    # Redis Configuration
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL",
        env="CELERY_BROKER_URL",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL",
        env="CELERY_RESULT_BACKEND",
    )

    # File Storage Configuration
    temp_storage_path: str = Field(
        default="app/temp",
        description="Temporary file storage path",
        env="TEMP_STORAGE_PATH",
    )
    max_file_size_mb: int = Field(
        default=50,
        description="Maximum file upload size in MB",
        env="MAX_FILE_SIZE_MB",
    )

    # Application metadata
    app_name: str = Field(
        default="Sample Chatbot",
        description="Application name",
        env="APP_NAME",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
        env="APP_VERSION",
    )
    app_description: str = Field(
        default="A sample chatbot application with FastAPI and PostgreSQL",
        description="Application description",
        env="APP_DESCRIPTION",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Create a global settings instance
settings = get_settings()
