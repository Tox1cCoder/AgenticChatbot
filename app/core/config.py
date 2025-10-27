from functools import lru_cache
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings

from pathlib import Path
from dotenv import load_dotenv

# Load .env from the workspace root
dotenv_path = Path(__file__).parent.parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)


class Settings(BaseSettings):
    model_config = {
        "env_file": str(dotenv_path) if dotenv_path.exists() else ".env",
        "env_file_encoding": "utf-8",
    }
    # Database settings
    database_url: str = Field(
        default="postgresql://postgres:123123123@localhost:5432/chatbot",
        description="Database URL for PostgreSQL connection",
    )

    # API settings
    api_host: str = Field(
        default="0.0.0.0",
        description="API host",
    )
    api_port: int = Field(
        default=8000,
        description="API port",
    )
    api_debug: bool = Field(
        default=False,
        description="Debug mode",
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment",
    )

    # Security
    secret_key: str = Field(
        default="secret-key",
        description="Secret key for security",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    access_token_expire_minutes: int = Field(
        default=240,
        description="Access token expiration in minutes",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration in days",
    )

    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS allowed origins",
    )

    # LLM API Keys
    gemini_api_key: str = Field(
        default="",
        description="Gemini API Key",
    )
    tavily_api_key: str = Field(
        default="",
        description="Tavily API Key for web search",
    )

    # Image Generation Configuration
    enable_image_generation: bool = Field(
        default=True,
        description="Toggle for enabling or disabling the image generator agent",
    )
    image_generator_model: str = Field(
        default="gemini-2.0-flash-preview-image-generation",
        description="Gemini model identifier used for image generation",
    )
    image_generator_default_aspect_ratio: str = Field(
        default="1:1",
        description="Default aspect ratio for generated images (e.g., 1:1, 16:9)",
    )
    image_generator_max_images: int = Field(
        default=1,
        description="Maximum number of images to request per generation",
    )

    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database URL",
    )
    qdrant_collection_name: str = Field(
        default="documents_gemma",
        description="Qdrant collection name for document storage",
    )

    embedding_dimension: int = Field(
        default=768,
        description="Dimension of the embedding vectors",
    )

    # Conversation Memory Configuration
    memory_max_messages: int = Field(
        default=0,
        description="Maximum number of messages cached in memory per conversation (0 = no limit)",
    )
    memory_load_batch_size: int = Field(
        default=100,
        description="Number of messages to load per batch when hydrating memory from the database",
    )
    chat_history_max_messages: int = Field(
        default=0,
        description="Maximum prior messages to include when building chat prompts (0 = no limit)",
    )
    chat_history_max_tokens: int = Field(
        default=0,
        description="Approximate maximum tokens of chat history to include in prompts (0 = no limit)",
    )
    rag_history_max_messages: int = Field(
        default=0,
        description="Maximum prior messages to include when building RAG prompts (0 = no limit)",
    )
    rag_history_max_tokens: int = Field(
        default=0,
        description="Approximate maximum tokens of RAG history to include in prompts (0 = no limit)",
    )

    # Redis Configuration
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL",
    )

    # Celery Worker Configuration
    celery_task_time_limit: int = Field(
        default=300,
        description="Hard time limit for Celery tasks in seconds",
    )
    celery_task_soft_time_limit: int = Field(
        default=240,
        description="Soft time limit for Celery tasks in seconds",
    )
    celery_worker_concurrency: int = Field(
        default=2,
        description="Number of concurrent Celery workers",
    )
    celery_worker_prefetch_multiplier: int = Field(
        default=1,
        description="Task prefetch multiplier for Celery workers",
    )

    # File Storage Configuration
    temp_storage_path: str = Field(
        default="app/temp",
        description="Temporary file storage path",
    )
    max_file_size_mb: int = Field(
        default=50,
        description="Maximum file upload size in MB",
    )

    # Document Processing Configuration
    allowed_file_extensions: List[str] = Field(
        default=[".txt", ".pdf", ".docx"],
        description="List of allowed file extensions for document upload",
    )
    document_chunk_size: int = Field(
        default=1000,
        description="Chunk size for text splitting in characters",
    )
    document_chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks in characters",
    )
    document_processing_timeout: int = Field(
        default=300,
        description="Maximum processing time for documents in seconds",
    )

    # RAG Retrieval Configuration
    rag_top_k: int = Field(
        default=15,
        description="Number of chunks to retrieve from vector database",
    )
    rag_score_threshold: float = Field(
        default=0.2,
        description="Minimum similarity score for retrieval",
    )
    rag_max_context_tokens: int = Field(
        default=30000,
        description="Maximum tokens to include in RAG context",
    )

    # Re-ranking Configuration
    enable_reranking: bool = Field(
        default=True,
        description="Enable re-ranking of retrieved chunks",
    )
    reranker_model: str = Field(
        default="zeroentropy/zerank-1-small",
        description="Re-ranker model name",
    )
    rerank_top_k: int = Field(
        default=10,
        description="Number of chunks to keep after re-ranking",
    )

    # Vector Database Batch Processing
    qdrant_upsert_batch_size: int = Field(
        default=1000,
        description="Batch size for Qdrant upsert operations (points per batch)",
    )

    # Advanced Chunking Configuration
    chunk_by_sentences: bool = Field(
        default=True,
        description="Chunk by complete sentences instead of arbitrary splits",
    )
    preserve_cross_page_context: bool = Field(
        default=True,
        description="Preserve context across PDF pages",
    )

    # Prompt Configuration
    rag_chunks_in_prompt: int = Field(
        default=10,
        description="Maximum number of chunks to include in prompt (0 = all)",
    )
    max_chunk_chars_in_prompt: int = Field(
        default=2000,
        description="Maximum characters per chunk in prompt (0 = no limit)",
    )

    # Search Agent Configuration
    search_max_results: int = Field(
        default=5,
        description="Maximum number of search results to return",
    )
    search_history_max_messages: int = Field(
        default=0,
        description="Maximum prior messages to include when building search prompts (0 = no limit)",
    )
    search_history_max_tokens: int = Field(
        default=0,
        description="Approximate maximum tokens of search history to include in prompts (0 = no limit)",
    )

    # ReAct Agent Configuration
    react_agent_max_iterations: int = Field(
        default=10,
        description="Maximum reasoning/acting cycles before stopping",
    )
    react_agent_recursion_limit: int = Field(
        default=25,
        description="LangGraph recursion limit for agent execution",
    )
    enable_parallel_tool_calls: bool = Field(
        default=True,
        description="Allow models to call multiple tools in parallel",
    )
    tool_choice_mode: str = Field(
        default="auto",
        description="Tool calling mode: 'auto', 'any', 'none', or specific tool name",
    )

    # Health Check Configuration
    health_check_timeout: int = Field(
        default=5,
        description="Timeout for health check requests in seconds",
    )
    enable_health_checks: bool = Field(
        default=True,
        description="Enable or disable health check endpoints",
    )

    # LangGraph Checkpoint Configuration
    enable_langgraph_checkpoints: bool = Field(
        default=True,
        description="Enable or disable LangGraph checkpoint persistence",
    )
    checkpoint_schema: str = Field(
        default="public",
        description="PostgreSQL schema for checkpoint tables",
    )
    checkpoint_cleanup_days: int = Field(
        default=30,
        description="Days to retain old checkpoints before cleanup",
    )

    # Application metadata
    app_name: str = Field(
        default="Sample Chatbot",
        description="Application name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    app_description: str = Field(
        default="A sample chatbot application with FastAPI and PostgreSQL",
        description="Application description",
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
