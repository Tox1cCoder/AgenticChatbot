from functools import lru_cache
from typing import List
import logging
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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
        default=600,
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

    # Image Captioning Configuration
    image_caption_model: str = Field(
        default="gemini-flash-latest",
        description="Gemini model identifier for document image captioning",
    )
    image_caption_max_retry_attempts: int = Field(
        default=5,
        description="Maximum retry attempts when captioning document images",
    )
    image_caption_retry_delay_seconds: float = Field(
        default=5.0,
        description="Base delay (seconds) to wait before retrying caption requests when no retry hint is provided",
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
    document_chunk_size: int = Field(
        default=1000,
        description="Chunk size for text splitting in characters",
    )
    document_chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks in characters",
    )
    mineru_timeout: int = Field(
        default=300,
        description="Timeout for MinerU subprocess in seconds",
    )
    document_images_storage_path: str = Field(
        default="app/storage/document_images",
        description="Storage path for extracted document images",
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
    preserve_cross_page_context: bool = Field(
        default=True,
        description="Preserve context across PDF pages",
    )

    # Table Processing Configuration
    extract_tables_from_pdf: bool = Field(
        default=True, description="Enable table extraction from PDF documents"
    )
    table_format: str = Field(
        default="markdown",
        description="Format for extracted tables (markdown, grid, plain)",
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
        default=3,
        description="Maximum number of refinement iterations before stopping",
    )
    react_agent_quality_threshold: float = Field(
        default=0.7,
        description="Minimum quality score (0.0-1.0) to accept response without refinement",
    )
    react_agent_recursion_limit: int = Field(
        default=25,
        description="LangGraph recursion limit for agent execution. Should be set to 2 * react_agent_max_iterations + 1 per LangGraph best practices",
    )
    tool_choice_mode: str = Field(
        default="auto",
        description="Tool calling mode: 'auto', 'any', 'none', or specific tool name",
    )

    # Tool Execution Configuration
    tool_execution_timeout: int = Field(
        default=30,
        description="Timeout for individual tool calls in seconds",
    )
    tool_execution_max_retries: int = Field(
        default=2,
        description="Maximum retry attempts for failed tool executions",
    )
    tool_validation_enabled: bool = Field(
        default=True,
        description="Enable/disable Pydantic validation for tool arguments and results",
    )

    # Hallucination Prevention Configuration
    confidence_threshold_abstain: float = Field(
        default=0.3,
        description="Minimum confidence score below which agent should abstain from answering",
    )
    enable_structured_output_validation: bool = Field(
        default=False,
        description="Toggle for using Pydantic structured output schemas for response validation",
    )
    confidence_weight_tool_success: float = Field(
        default=0.4,
        description="Weight for tool success rate in confidence calculation",
    )
    confidence_weight_completeness: float = Field(
        default=0.3,
        description="Weight for response completeness in confidence calculation",
    )
    confidence_weight_retrieval: float = Field(
        default=0.3,
        description="Weight for retrieval quality (RAG) in confidence calculation",
    )
    enable_citation_verification: bool = Field(
        default=True,
        description="Enable citation verification for RAG agent responses",
    )
    min_citation_coverage: float = Field(
        default=0.5,
        description="Minimum fraction of retrieved docs that should be referenced in RAG response",
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

    # Human-in-the-Loop Configuration
    enable_human_in_the_loop: bool = Field(
        default=True,
        description="Toggle to enable/disable human-in-the-loop globally",
    )
    hitl_default_allow_edit: bool = Field(
        default=True,
        description="Whether to allow editing tool arguments by default",
    )
    hitl_default_allow_respond: bool = Field(
        default=True,
        description="Whether to allow responding with feedback by default",
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
