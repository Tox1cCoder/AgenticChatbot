import logging
import os
import secrets
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

# Load .env from the workspace root
dotenv_path = Path(__file__).parent.parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

_langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
_langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")

if _langsmith_tracing and _langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = _langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "sample-chatbot")


def _inject_redis_password(url: str, password: str) -> str:
    """Attach a password to a redis/rediss URL when credentials are missing."""
    raw_url = (url or "").strip()
    raw_password = (password or "").strip()
    if not raw_url or not raw_password:
        return raw_url

    parsed = urlparse(raw_url)
    if parsed.scheme.lower() not in {"redis", "rediss"}:
        return raw_url
    if parsed.password:
        return raw_url
    if not parsed.hostname:
        return raw_url

    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    hostport = hostname
    if parsed.port is not None:
        hostport = f"{hostport}:{parsed.port}"

    username = parsed.username or ""
    credentials = f"{username}:{raw_password}" if username else f":{raw_password}"
    return urlunparse(parsed._replace(netloc=f"{credentials}@{hostport}"))


def _normalize_redis_loopback_host(url: str) -> str:
    """Use 127.0.0.1 instead of localhost for Redis on Windows async clients."""
    raw_url = (url or "").strip()
    if not raw_url:
        return raw_url

    parsed = urlparse(raw_url)
    if parsed.scheme.lower() not in {"redis", "rediss"}:
        return raw_url
    if sys.platform != "win32":
        return raw_url
    if (parsed.hostname or "").lower() != "localhost":
        return raw_url

    hostport = "127.0.0.1"
    if parsed.port is not None:
        hostport = f"{hostport}:{parsed.port}"

    credentials = ""
    if parsed.username or parsed.password:
        username = parsed.username or ""
        password = parsed.password or ""
        credentials = f"{username}:{password}@" if username else f":{password}@"

    return urlunparse(parsed._replace(netloc=f"{credentials}{hostport}"))


class Settings(BaseSettings):
    model_config = {
        "env_file": str(dotenv_path) if dotenv_path.exists() else ".env",
        "env_file_encoding": "utf-8",
        # Tolerate stale env vars from retired settings (e.g. AGENTIC_RAG_ENABLED)
        # rather than failing hard on startup after the Phase 9 cleanup.
        "extra": "ignore",
    }
    # Database settings
    database_url: str = Field(
        default="postgresql://localhost:5432/chatbot",
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
        default="",
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
    cors_origins: list[str] = Field(
        default=[],
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
    smithery_api_key: str = Field(
        default="",
        description="Smithery API Key for MCP server access",
    )

    # Multi-Provider Configuration
    model_encryption_key: str = Field(
        default="",
        description="Fernet encryption key for storing provider API keys (32 url-safe base64-encoded bytes)",
    )
    openai_request_timeout_seconds: int = Field(
        default=60,
        description="Timeout for OpenAI API requests in seconds",
    )
    provider_retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for provider API calls before fallback",
    )
    provider_retry_delay_seconds: float = Field(
        default=1.0,
        description="Base delay in seconds between retry attempts (uses exponential backoff)",
    )

    # LangSmith Configuration
    langsmith_api_key: str = Field(
        default="",
        description="LangSmith API Key for tracing and observability",
    )
    langsmith_project: str = Field(
        default="sample-chatbot",
        description="LangSmith project name for organizing traces",
    )
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing (requires valid API key)",
    )

    # Agent Model Configuration
    rag_agent_model: str = Field(default="gemini-3.1-pro-preview")
    chat_agent_model: str = Field(default="gemini-3-flash-preview")
    search_agent_model: str = Field(default="gemini-3-flash-preview")

    # RAG Embedding / Reranker / Chunking
    rag_embedding_provider: str = Field(
        default="gemini",
        description="Active RAG embedding provider. 'gemini' uses the Gemini "
        "Embeddings API (gemini-embedding-2). 'sentence_transformers' is a "
        "local-only fallback for offline development.",
    )
    rag_embedding_model: str = Field(
        default="gemini-embedding-2",
        description="Embedding model identifier. With provider 'gemini' this "
        "is a Gemini API model name; with 'sentence_transformers' it is a "
        "HuggingFace model id.",
    )
    rag_embedding_dimension: int = Field(
        default=3072,
        description="Output dimensionality requested from the embedding "
        "provider. Must match the Qdrant collection vector size.",
    )
    rag_embedding_query_task: str = Field(
        default="search result",
        description="Gemini Embeddings 2 retrieval task hint for queries. "
        "Use 'search result' for keyword-style retrieval or "
        "'question answering' for QA-style retrieval.",
    )
    rag_multimodal_image_embeddings_enabled: bool = Field(
        default=False,
        description="When True, raw document images are embedded as separate "
        "multimodal Qdrant points. Disabled by default — caption-augmented "
        "text chunks are the primary image-retrieval path.",
    )
    rag_reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder used to rerank top-k retrieved chunks.",
    )
    rag_chunk_target_tokens: int = Field(
        default=400,
        description="Preferred token count per generated chunk.",
    )
    rag_chunk_overlap_tokens: int = Field(
        default=40,
        description="Token overlap applied when splitting long text.",
    )
    rag_chunk_max_tokens: int = Field(
        default=800,
        description="Hard ceiling on per-chunk token count.",
    )
    rag_index_batch_size: int = Field(
        default=16,
        description="Batch size used when embedding chunks during indexing.",
    )

    # Media Resolution Configuration (for vision models)
    media_resolution: str = Field(
        default="high",
        description="Media resolution for vision models: low, medium, high (Gemini 3 supports per-part resolution)",
    )
    enable_gemini_code_execution: bool = Field(
        default=True,
        description="Enable Gemini code execution tool for agentic vision workflows across agents",
    )

    # Image Generation Configuration
    enable_image_generation: bool = Field(
        default=True,
        description="Toggle for enabling or disabling the image generator agent",
    )
    image_generator_model: str = Field(
        default="gemini-3-pro-image-preview",
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
        default="gemini-3-flash-preview",
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
        default="documents_gemini_embedding_2_3072",
        description="Qdrant collection name for document storage. The default "
        "is namespaced by embedding provider/model/dimension so swapping "
        "providers requires a deliberate collection cutover.",
    )

    # Conversation Memory Configuration
    memory_max_messages: int = Field(
        default=0,
        description="Maximum number of messages cached in memory per conversation (0 = no limit)",
    )
    memory_load_batch_size: int = Field(
        default=100,
        description="Number of messages to load per batch when hydrating memory from the database (max 100)",
    )
    chat_history_max_messages: int = Field(
        default=24,
        description="Maximum prior messages to include when building chat prompts (0 = no limit)",
    )
    chat_history_max_tokens: int = Field(
        default=9000,
        description="Approximate maximum tokens of chat history to include in prompts (0 = no limit)",
    )
    rag_history_max_messages: int = Field(
        default=12,
        description="Maximum prior messages to include when building RAG prompts (0 = no limit)",
    )
    rag_history_max_tokens: int = Field(
        default=3000,
        description="Approximate maximum tokens of RAG history to include in prompts (0 = no limit)",
    )

    # Summarization Middleware Configuration
    enable_summarization: bool = Field(
        default=True,
        description="Enable automatic conversation summarization for long conversations",
    )
    summarization_trigger_tokens: int = Field(
        default=18000,
        description="Trigger summarization when estimated tokens exceed this threshold",
    )
    summarization_trigger_messages: int = Field(
        default=60,
        description="Trigger summarization when message count exceeds this threshold",
    )
    summarization_trigger_fraction: float = Field(
        default=0.55,
        description="Trigger summarization when context usage exceeds this fraction of model's context window (0.0-1.0)",
    )
    summarization_model_context_size: int = Field(
        default=128000,
        description="Model context window size in tokens (cross-provider practical baseline)",
    )
    summarization_keep_messages: int = Field(
        default=8,
        description="Number of recent messages to keep after summarization",
    )
    summarization_model: str = Field(
        default="gemini-3-flash-preview",
        description="Model to use for generating conversation summaries",
    )
    summarization_max_summary_tokens: int = Field(
        default=1500,
        description="Hard cap on rolling summary size in estimated tokens. "
        "Summaries exceeding this limit are truncated to stay within budget. "
        "Set to 0 for unlimited (no truncation).",
    )
    summarization_timeout_seconds: int = Field(
        default=30,
        description="Maximum seconds to wait for a summarization model call before timing out. "
        "On timeout the original state is returned unchanged (fail-closed).",
    )
    suppress_internal_stream_chunks: bool = Field(
        default=True,
        description="When True, stream chunks tagged as 'internal' (e.g. summarization node output) "
        "are silently dropped before being forwarded to clients. "
        "Disable only for debugging.",
    )

    # Redis Configuration
    redis_url: str = Field(
        default="",
        description="Redis connection URL used for widget runtime state, HITL timeout tracking, and other shared-state features. "
        "Required for live widget flows when the widgets MCP server runs out-of-process. "
        "Falls back to celery_broker_url if blank.",
    )
    redis_password: str = Field(
        default="",
        description="Optional Redis password convenience variable for local Docker setups. "
        "When set, it is automatically injected into redis:// and rediss:// URLs that omit credentials.",
    )
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
    mineru_timeout: int = Field(
        default=300,
        description="Timeout for MinerU subprocess in seconds",
    )
    mineru_api_url: str = Field(
        default="",
        description=(
            "Optional MinerU FastAPI base URL. When blank, each mineru CLI call starts "
            "a temporary local mineru-api service (higher startup overhead)."
        ),
    )
    mineru_backend: str = Field(
        default="pipeline",
        description=(
            "MinerU processing backend. Supported values: 'pipeline', "
            "'hybrid-auto-engine', 'hybrid-http-client', 'vlm-auto-engine', "
            "'vlm-http-client'."
        ),
    )
    mineru_method: str = Field(
        default="auto",
        description=(
            "MinerU parsing method for pipeline/hybrid backends: 'auto', 'txt', or 'ocr'."
        ),
    )
    mineru_lang: str = Field(
        default="",
        description=(
            "Optional MinerU OCR language for pipeline/hybrid backends (for example: 'en', 'ch')."
        ),
    )
    extract_formulas_from_pdf: bool = Field(
        default=True,
        description="Enable formula extraction from PDF documents",
    )
    mineru_extra_args: list[str] = Field(
        default_factory=list,
        description="Extra CLI arguments forwarded verbatim to the mineru command (e.g. ['--device', 'cpu'])",
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
    # Re-ranking Configuration
    enable_reranking: bool = Field(
        default=True,
        description="Enable re-ranking of retrieved chunks",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Re-ranker model name (kept for backward compat; canonical "
        "setting is rag_reranker_model).",
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

    # Table Processing Configuration
    extract_tables_from_pdf: bool = Field(
        default=True, description="Enable table extraction from PDF documents"
    )
    table_format: str = Field(
        default="markdown",
        description="Format for extracted tables (markdown, grid, plain)",
    )

    # Search Agent Configuration
    search_history_max_messages: int = Field(
        default=16,
        description="Maximum prior messages to include when building search prompts (0 = no limit)",
    )
    search_history_max_tokens: int = Field(
        default=5000,
        description="Approximate maximum tokens of search history to include in prompts (0 = no limit)",
    )

    # Planning Agent History Configuration
    planning_history_max_messages: int = Field(
        default=16,
        description="Maximum prior messages to include when building planning prompts (0 = no limit)",
    )
    planning_history_max_tokens: int = Field(
        default=5000,
        description="Approximate maximum tokens of planning history to include in prompts (0 = no limit)",
    )

    # ReAct Agent Configuration
    react_agent_max_iterations: int = Field(
        default=50,
        description="Maximum number of refinement iterations before stopping",
    )
    react_agent_quality_threshold: float = Field(
        default=0.7,
        description="Minimum quality score (0.0-1.0) to accept response without refinement",
    )
    react_agent_recursion_limit: int = Field(
        default=101,
        description="LangGraph recursion limit for agent execution. Should be set to 2 * react_agent_max_iterations + 1 per LangGraph best practices",
    )
    tool_choice_mode: str = Field(
        default="auto",
        description="Tool calling mode: 'auto', 'any', 'none', or specific tool name",
    )

    # Planning Agent Configuration
    max_auto_plan_tasks: int = Field(
        default=20,
        description="Absolute maximum number of tasks per execution session (safety limit)",
    )
    execution_call_budget: int = Field(
        default=20,
        description="Maximum LLM calls per execution session before pausing for user",
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

    # Tool Result Token Management
    tool_result_max_chars: int = Field(
        default=16000,
        description="Maximum characters to include in ToolMessage content sent to model (0 = no limit). Full output is preserved in artifacts for UI.",
    )
    tool_result_truncation_suffix: str = Field(
        default="\n\n[Output truncated - full result available in tool artifacts]",
        description="Suffix to append when tool result is truncated",
    )

    # Per-Agent Tool Allowlists
    # Empty list means bind all available tools; non-empty list restricts to specified tools/servers
    chat_agent_allowed_tools: list[str] = Field(
        default=[],
        description="Tool names or server names that chat agent can use. Empty = all tools.",
    )
    search_agent_allowed_tools: list[str] = Field(
        default=[],
        description="Tool names or server names that search agent can use. Empty = all tools.",
    )
    rag_agent_allowed_tools: list[str] = Field(
        default=[],
        description="Tool names or server names that RAG agent can use. Empty = all tools.",
    )
    planning_agent_allowed_tools: list[str] = Field(
        default=[],
        description="Tool names or server names that planning agent can use. Empty = all tools.",
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
        default="1.0",
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
    hitl_tools_require_approval: list[str] = Field(
        default=[],
        description="List of tool names that require human approval. Empty list means NO tools require approval when HITL is enabled.",
    )
    hitl_approval_timeout_minutes: int = Field(
        default=30,
        description="Timeout in minutes for pending approval requests. After timeout, the workflow can be auto-rejected or cleaned up.",
    )

    # Gemini Thinking Configuration
    enable_thinking: bool = Field(
        default=True,
        description="Enable Gemini thinking mode for models that support it",
    )
    include_thoughts_in_response: bool = Field(
        default=True,
        description="Include thought summaries in streaming responses when thinking mode is enabled",
    )
    thinking_level: str = Field(
        default="high",
        description="Thinking level for Gemini 3 models (minimal, low, medium, high).",
    )
    thinking_budget: int = Field(
        default=-1,
        description="Thinking budget for Gemini 2.5 models (-1 for dynamic, 0 to disable, or specific token count like 1024).",
    )

    # Agentic RAG Configuration
    agentic_max_iterations: int = Field(
        default=10,
        description="Maximum tool calls in agentic RAG mode before forcing final answer",
    )
    agentic_preview_chars: int = Field(
        default=1500,
        description="Characters to include in document preview during scan phase (~1 page)",
    )

    # Auto-Continue Configuration
    auto_continue_enabled: bool = Field(
        default=True,
        description="Enable automatic continuation when agent hits iteration limits",
    )
    auto_continue_max_rounds: int = Field(
        default=5,
        description="Maximum number of continuation rounds per user message (safety cap)",
    )
    auto_continue_soft_limit_ratio: float = Field(
        default=0.8,
        description="Fraction of the loop budget to consume per round before rolling to the next round (0.1-1.0)",
    )
    auto_continue_emit_events: bool = Field(
        default=False,
        description="Emit continuation_start events during streaming (debug/UX)",
    )
    auto_continue_max_total_iterations: int = Field(
        default=200,
        description="Absolute max iterations across all continuation rounds",
    )
    auto_continue_timeout_seconds: int = Field(
        default=300,
        description="Maximum wall-clock time for all continuation rounds (seconds)",
    )

    # Planning Agent Explicit Settings (promoted from getattr defaults)
    planning_max_iterations: int = Field(
        default=20,
        description="Maximum planning tool calls before pausing for user",
    )
    planning_consecutive_errors_limit: int = Field(
        default=3,
        description="Maximum consecutive planning tool errors before stopping",
    )

    # MCP Tool Search Configuration (Deferred Loading)
    mcp_tool_search_enabled: bool = Field(
        default=True,
        description="Enable deferred MCP tool loading via tool_search. When enabled, only tool_search + pinned tools are bound by default.",
    )
    mcp_tool_search_default_top_k: int = Field(
        default=5,
        description="Default number of tools to return from tool_search queries.",
    )
    mcp_tool_search_max_top_k: int = Field(
        default=100,
        description="Maximum allowed top_k value for tool_search (clamped to this).",
    )
    mcp_tool_search_autoload_top_k: int = Field(
        default=3,
        description="Number of top-ranked tools to automatically load/bind after tool_search. Reduced to 3 to limit aggressive autoloading.",
    )
    mcp_tool_search_inventory_default_top_k: int = Field(
        default=20,
        description="Default number of tools to return per server in per-server inventory mode (tool_search with server_name but no query).",
    )
    mcp_tool_search_inventory_max_top_k: int = Field(
        default=50,
        description="Maximum allowed top_k for inventory mode (per-server tool listing).",
    )
    mcp_tool_search_pinned_tools: list[str] = Field(
        default=[],
        description="Tool names (or server::tool_name) that are always bound, not deferred. Recommended 3-5 high-frequency tools.",
    )
    mcp_tool_search_max_pinned_tools: int = Field(
        default=5,
        description="Safety cap on pinned tools to prevent schema bloat.",
    )
    mcp_tool_search_max_loaded_tools_per_conversation: int = Field(
        default=8,
        description="Maximum deferred tools that can be loaded per conversation.",
    )
    mcp_tool_search_loaded_tools_ttl_minutes: int = Field(
        default=30,
        description="TTL in minutes for loaded deferred tools (evicted after expiry).",
    )
    mcp_tool_search_min_relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum relevance score for a tool to appear in search results. Tools below this threshold are excluded entirely.",
    )
    mcp_tool_search_autoload_min_relevance_score: float = Field(
        default=2.0,
        ge=0.0,
        description="Minimum relevance score for a tool to be autoloaded. Stricter than min_relevance_score to prevent arbitrary autoloading.",
    )
    mcp_tool_search_log_queries: bool = Field(
        default=False,
        description="Log tool_search queries (disable in production to avoid logging sensitive queries).",
    )

    # Client Runtime Bridge Configuration
    enable_client_runtime_bridge: bool = Field(
        default=True,
        description="Enable the client-runtime bridge for per-device client backends. "
        "When enabled, the server can dispatch tool calls to connected client devices.",
    )
    client_runtime_ws_timeout_seconds: int = Field(
        default=60,
        description="Timeout in seconds for client runtime WebSocket operations (tool dispatch, heartbeat).",
    )
    client_runtime_catalog_cache_ttl_seconds: int = Field(
        default=300,
        description="TTL in seconds for caching client device tool/skill catalogs. "
        "Catalogs are refreshed when a device reconnects or explicitly syncs.",
    )
    client_runtime_require_connected_device_for_local_tools: bool = Field(
        default=True,
        description="When True, tool calls targeting client-local tools fail if no device is connected. "
        "When False, such calls return a recoverable error allowing the model to adapt.",
    )
    client_runtime_heartbeat_interval_seconds: int = Field(
        default=30,
        description="Expected heartbeat interval from connected client devices. "
        "Devices not sending heartbeats within 2x this interval are marked offline.",
    )
    client_runtime_max_tool_result_size_bytes: int = Field(
        default=1048576,
        description="Maximum size in bytes for tool results returned from client devices (1MB default). "
        "Results exceeding this are truncated with a warning.",
    )

    # ── Validators ──────────────────────────────────────────────────────

    @field_validator(
        "chat_history_max_messages",
        "chat_history_max_tokens",
        "rag_history_max_messages",
        "rag_history_max_tokens",
        "search_history_max_messages",
        "search_history_max_tokens",
        "planning_history_max_messages",
        "planning_history_max_tokens",
        "summarization_trigger_tokens",
        "summarization_trigger_messages",
        "summarization_keep_messages",
        "summarization_model_context_size",
        "memory_max_messages",
        "memory_load_batch_size",
        "tool_result_max_chars",
        "summarization_max_summary_tokens",
        "summarization_timeout_seconds",
        mode="before",
    )
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        v = int(v)
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @field_validator(
        "summarization_trigger_fraction",
        "auto_continue_soft_limit_ratio",
        mode="before",
    )
    @classmethod
    def _validate_fraction_fields(cls, v: float) -> float:
        v = float(v)
        if not (0.0 < v <= 1.0):
            raise ValueError("Value must be in the range (0.0, 1.0]")
        return v

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "Settings":
        if self.redis_password:
            self.redis_url = _inject_redis_password(self.redis_url, self.redis_password)
            self.celery_broker_url = _inject_redis_password(
                self.celery_broker_url, self.redis_password
            )
            self.celery_result_backend = _inject_redis_password(
                self.celery_result_backend, self.redis_password
            )
        self.redis_url = _normalize_redis_loopback_host(self.redis_url)
        self.celery_broker_url = _normalize_redis_loopback_host(self.celery_broker_url)
        self.celery_result_backend = _normalize_redis_loopback_host(self.celery_result_backend)
        if not self.secret_key or self.secret_key == "secret-key":
            if self.environment == "development":
                self.secret_key = secrets.token_urlsafe(48)
                logging.getLogger(__name__).warning(
                    "SECRET_KEY is not set; generated ephemeral development key. "
                    "Set SECRET_KEY in .env for stable local auth sessions."
                )
            else:
                raise ValueError("secret_key must be set to a strong value outside development")
        if self.environment != "development" and self.api_debug:
            raise ValueError("api_debug must be disabled outside development")
        if self.summarization_keep_messages >= self.summarization_trigger_messages:
            raise ValueError(
                f"summarization_keep_messages ({self.summarization_keep_messages}) "
                f"must be less than summarization_trigger_messages ({self.summarization_trigger_messages})"
            )
        if self.summarization_model_context_size < self.summarization_trigger_tokens:
            raise ValueError(
                f"summarization_model_context_size ({self.summarization_model_context_size}) "
                f"must be >= summarization_trigger_tokens ({self.summarization_trigger_tokens})"
            )
        return self


def _log_startup_warnings(s: "Settings") -> None:
    """Log warnings for settings that may indicate misconfiguration."""
    _logger = logging.getLogger(__name__)
    zero_budget_fields = []
    for attr in (
        "chat_history_max_messages",
        "chat_history_max_tokens",
        "rag_history_max_messages",
        "rag_history_max_tokens",
        "search_history_max_messages",
        "search_history_max_tokens",
        "planning_history_max_messages",
        "planning_history_max_tokens",
    ):
        if getattr(s, attr, 0) == 0:
            zero_budget_fields.append(attr)

    if zero_budget_fields and s.environment != "development":
        _logger.warning(
            "History budget(s) set to 0 (unlimited) in '%s' environment — "
            "this may cause unbounded prompt growth: %s",
            s.environment,
            ", ".join(zero_budget_fields),
        )


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings with caching.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Create a global settings instance
settings = get_settings()

# Emit startup warnings for potentially misconfigured budgets
_log_startup_warnings(settings)
