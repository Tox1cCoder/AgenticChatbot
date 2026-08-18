import logging
import os
import secrets
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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


_DEV_SECRET_KEY_PATH = Path(__file__).resolve().parents[2] / ".dev_secret_key"


def _load_or_create_dev_secret_key(key_path: Path | None = None) -> str:
    """Return a stable development signing key, persisted across restarts.

    A fresh random key on every process start would invalidate all previously
    issued access/refresh tokens, forcing everyone to re-authenticate on each
    reload (the root cause of recurring 401s in development). Persisting the key
    to a gitignored local file keeps sessions stable between restarts. Production
    never reaches this path — it hard-requires an explicit ``SECRET_KEY``.
    """
    path = key_path or _DEV_SECRET_KEY_PATH
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    generated = secrets.token_urlsafe(48)
    try:
        path.write_text(generated, encoding="utf-8")
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not persist development SECRET_KEY to %s; auth sessions will "
            "reset on every restart. Set SECRET_KEY explicitly to avoid this.",
            path,
        )
    return generated


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


class ToolExecutionPolicyMatch(BaseModel):
    """Deployment-configured selector for one tool execution policy override.

    Accepts exactly one of four shapes, in increasing specificity: origin
    only; origin + exposed tool name; origin + server name + source tool
    name; or origin + qualified tool id. Mixed shapes and partial
    server/source pairs are rejected so a rule's specificity — and therefore
    its precedence against other rules — never depends on which optional
    fields happen to be set.
    """

    model_config = ConfigDict(extra="forbid")

    tool_origin: Literal["internal", "server_mcp", "client_mcp", "client_skill"]
    qualified_tool_id: str | None = None
    server_name: str | None = None
    source_tool_name: str | None = None
    exposed_tool_name: str | None = None

    @model_validator(mode="after")
    def _validate_single_match_shape(self) -> "ToolExecutionPolicyMatch":
        has_server = self.server_name is not None
        has_source = self.source_tool_name is not None
        if has_server != has_source:
            raise ValueError(
                "tool execution policy match must set server_name and "
                "source_tool_name together, not one without the other"
            )

        shape_count = sum(
            [
                self.qualified_tool_id is not None,
                self.exposed_tool_name is not None,
                has_server and has_source,
            ]
        )
        if shape_count > 1:
            raise ValueError(
                "tool execution policy match must use exactly one of: "
                "origin only; origin + exposed_tool_name; "
                "origin + server_name + source_tool_name; or "
                "origin + qualified_tool_id"
            )
        return self


class ToolExecutionPolicyOverride(BaseModel):
    """One deployment-configured tool execution policy override.

    Keyed by a diagnostic dict key (not identity) in
    `Settings.tool_execution_policies`. Resolution, caps, and the
    `disable_outer_timeout` code-owned allowlist check are applied by the
    policy resolver, not here.
    """

    model_config = ConfigDict(extra="forbid")

    match: ToolExecutionPolicyMatch
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    hard_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    total_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_attempts: int | None = Field(default=None, ge=1, le=5)
    retry_safe: bool | None = None
    idempotent: bool | None = None
    trust_mcp_metadata: bool = False
    disable_outer_timeout: bool = False
    timeout_hint: str | None = Field(default=None, max_length=240)

    @model_validator(mode="after")
    def _validate_trust_requires_exact_qualified_match(self) -> "ToolExecutionPolicyOverride":
        if self.trust_mcp_metadata and self.match.qualified_tool_id is None:
            raise ValueError(
                "trust_mcp_metadata is only valid on an exact origin + qualified_tool_id match rule"
            )
        return self


class InternalToolExecutionPolicy(BaseModel):
    """Strict application-owned execution metadata for internal tools."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    hard_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    total_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_attempts: int | None = Field(default=None, ge=1, le=5)
    retry_safe: bool | None = None
    idempotent: bool | None = None
    disable_outer_timeout: bool | None = None
    timeout_hint: str | None = Field(default=None, max_length=240)


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
    db_pool_size: int = Field(
        default=20,
        description=(
            "Async engine connection pool size. Sized for concurrent streaming "
            "requests; SQLAlchemy's default of 5 is a concurrency ceiling."
        ),
    )
    db_max_overflow: int = Field(
        default=10,
        description="Additional async connections allowed above db_pool_size under burst.",
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
    brave_search_api_key: str = Field(
        default="",
        description="Brave Search API Key for image search (X-Subscription-Token)",
    )

    # Brave Image Search budget/safety limits (operational, not behavior hardcoding)
    brave_image_search_default_count: int = Field(
        default=12,
        description=(
            "Default number of image results requested from Brave Image Search. "
            "Confidence tiering, the recency window and deduplication all cut this "
            "pool before two survivors are chosen, so a thin pool collapses to none. "
            "Results are private to the server and never reach the model, so a larger "
            "pool costs one response body and no tokens."
        ),
    )
    brave_image_search_max_count: int = Field(
        default=20,
        description="Hard cap on image results returned per Brave Image Search call.",
    )
    brave_image_search_timeout_seconds: float = Field(
        default=2.5,
        description="Brave image search request timeout.",
    )
    brave_image_search_default_safesearch: str = Field(
        default="strict",
        description="Default Brave safesearch level. Brave supports 'off' and 'strict'.",
    )

    # Tavily retrieval budget/safety limits (operational, not behavior hardcoding)
    tavily_search_default_max_results: int = Field(
        default=5,
        description="Default Tavily search result count.",
    )
    tavily_search_max_results: int = Field(
        default=10,
        description="Hard cap on Tavily search results returned per call.",
    )
    tavily_search_default_depth: str = Field(
        default="basic",
        description="Default Tavily search depth: basic, fast, ultra-fast, or advanced.",
    )
    tavily_search_auto_parameters: bool = Field(
        default=False,
        description="Allow Tavily to auto-select search parameters. May increase credit use.",
    )
    tavily_extract_max_urls: int = Field(
        default=5,
        description="Hard cap on URLs accepted by Tavily extract per call.",
    )
    tavily_extract_default_depth: str = Field(
        default="basic",
        description="Default Tavily extract depth: basic or advanced.",
    )
    tavily_extract_default_format: str = Field(
        default="markdown",
        description="Default Tavily extract format: markdown or text.",
    )
    tavily_extract_timeout_seconds: float = Field(
        default=20.0,
        description="Timeout sent to Tavily Extract, in seconds.",
    )
    tavily_map_max_depth: int = Field(default=2, description="Maximum Tavily map depth.")
    tavily_map_max_breadth: int = Field(default=20, description="Maximum Tavily map breadth.")
    tavily_map_limit: int = Field(default=50, description="Maximum Tavily map URL count.")
    tavily_map_timeout_seconds: float = Field(default=30.0, description="Tavily Map timeout.")
    tavily_crawl_max_depth: int = Field(default=1, description="Maximum Tavily crawl depth.")
    tavily_crawl_max_breadth: int = Field(default=10, description="Maximum Tavily crawl breadth.")
    tavily_crawl_limit: int = Field(default=20, description="Maximum Tavily crawl page count.")
    tavily_crawl_timeout_seconds: float = Field(default=45.0, description="Tavily Crawl timeout.")

    # Multi-Provider Configuration
    model_encryption_key: str = Field(
        default="",
        description=(
            "Fernet encryption key for storing provider API keys (32 url-safe base64-encoded bytes)"
        ),
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
    router_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model identifier used for request routing",
    )
    image_generator_tool_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model identifier used for image-agent tool calling",
    )
    canvas_agent_model: str = Field(
        default="gemini-3.1-pro-preview",
        description="Default model identifier used by the canvas agent",
    )
    suggestion_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model identifier used to generate follow-up suggestions",
    )
    title_generator_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model identifier used to generate conversation titles",
    )

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
    rag_rerank_candidate_pool: int = Field(
        default=40,
        ge=1,
        description="Maximum authorized fused candidates sent to the reranker.",
    )
    rag_evidence_candidate_limit: int = Field(
        default=10,
        ge=1,
        description="Maximum reranked candidates forwarded to evidence assembly.",
    )
    rag_reranker_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Total reranker queue and inference timeout in seconds.",
    )
    rag_reranker_max_concurrency: int = Field(
        default=2,
        ge=1,
        description="Maximum concurrent reranker model calls per service instance.",
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
    rag_semantic_chunking_enabled: bool = Field(
        default=False,
        description=(
            "Enable experimental embedding-based semantic chunk boundaries. "
            "Keep disabled until the shadow evaluation beats structural chunking."
        ),
    )
    rag_semantic_breakpoint_percentile: float = Field(
        default=90.0,
        ge=0.0,
        le=100.0,
        description="Adjacent-block embedding distance percentile used as a boundary.",
    )
    # Gemini Embeddings API accepts up to 100 contents per embed_content call
    # (documented limit for gemini-embedding-2). Default 32 is conservative.
    rag_embedding_batch_size: int = Field(
        default=32,
        description=(
            "Max number of chunks per Gemini embed_content request. "
            "Clamped to the API maximum of 100."
        ),
    )
    rag_embedding_max_concurrency: int = Field(
        default=4,
        description="Max concurrent Gemini embed_content requests during indexing.",
    )

    # Media Resolution Configuration (for vision models)
    media_resolution: str = Field(
        default="high",
        description=(
            "Media resolution for vision models: low, medium, high "
            "(Gemini 3 supports per-part resolution)"
        ),
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
        default="gemini-3-pro-image",
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
    enable_image_streaming: bool = Field(
        default=True,
        description=(
            "Stream generated images to clients as transient previews the moment "
            "each one is ready, instead of only with the terminal complete event."
        ),
    )
    image_stream_preview_max_b64_chars: int = Field(
        default=4_000_000,
        description=(
            "Maximum base64 length for a streamed image preview (~3 MB binary). "
            "Larger images skip the preview and arrive only at completion."
        ),
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
        description=(
            "Base delay (seconds) to wait before retrying caption requests when no retry "
            "hint is provided"
        ),
    )
    image_caption_max_concurrency: int = Field(
        default=4,
        description="Maximum number of simultaneous Gemini caption API calls per document.",
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
        description=(
            "Number of messages to load per batch when hydrating memory from the database (max 100)"
        ),
    )
    # Durable conversation memory (Memory Refactor 2026-04-29)
    memory_cache_ttl_seconds: int = Field(
        default=60,
        description=(
            "TTL (seconds) for the in-process prompt-history cache used by "
            "ConversationHistoryProvider"
        ),
    )
    memory_cache_max_conversations: int = Field(
        default=256,
        description=(
            "Maximum number of conversations cached by ConversationHistoryProvider "
            "before LRU eviction"
        ),
    )

    # Production conversation compaction. These settings are intentionally
    # provider-specific and form the single namespace used by the pipeline.
    conversation_summary_enabled: bool = Field(
        default=True,
        description="Enable durable background conversation compaction",
    )
    conversation_summary_provider: str = Field(
        default="gemini",
        description="Provider used only for conversation compaction",
    )
    conversation_summary_model: str = Field(
        default="gemini-2.5-flash",
        description="Stable model used only for conversation compaction",
    )
    conversation_summary_trigger_messages: int = Field(
        default=60,
        description="Eligible unsummarized messages required for background compaction",
    )
    conversation_summary_trigger_tokens: int = Field(
        default=18_000,
        description="Eligible unsummarized tokens required for background compaction",
    )
    conversation_summary_soft_context_ratio: float = Field(
        default=0.70,
        description="Input-budget ratio that requests durable background compaction",
    )
    conversation_summary_hard_context_ratio: float = Field(
        default=0.85,
        description="Input-budget ratio that triggers bounded request-path compaction",
    )
    conversation_summary_keep_recent_turns: int = Field(
        default=4,
        description="Newest complete turns retained verbatim during compaction",
    )
    conversation_summary_max_tokens: int = Field(
        default=1_500,
        description="Hard token cap for validated structured conversation memory",
    )
    conversation_summary_timeout_seconds: int = Field(
        default=30,
        description="Timeout for background and emergency compaction provider calls",
    )
    conversation_summary_max_attempts: int = Field(
        default=5,
        description="Maximum transient-failure attempts before a summary job becomes dead",
    )
    conversation_summary_lease_seconds: int = Field(
        default=120,
        description="Duration of a summary worker claim lease",
    )
    conversation_summary_retry_base_seconds: int = Field(
        default=5,
        description="Base delay for exponential summary-job retry backoff",
    )
    conversation_summary_retry_max_seconds: int = Field(
        default=900,
        description="Maximum delay for summary-job retry backoff",
    )
    conversation_summary_reconcile_seconds: int = Field(
        default=60,
        description="Celery Beat interval and dispatch debounce for summary reconciliation",
    )
    conversation_summary_safety_margin_tokens: int = Field(
        default=1_024,
        description="Tokens held back from the provider input context as a safety margin",
    )
    conversation_summary_default_reserved_output_tokens: int = Field(
        default=4_096,
        description="Default output-token reservation used by request preflight",
    )

    # Per-user model-usage analytics
    model_usage_tracking_enabled: bool = Field(
        default=True,
        description="Record per-user model-call usage events and minute rollups",
    )
    model_usage_ui_enabled: bool = Field(
        default=True,
        description="Expose the model-usage analytics UI/read endpoints",
    )
    model_usage_raw_retention_days: int = Field(
        default=90,
        description="Days of raw model-usage events retained before cleanup",
    )
    model_usage_rollup_retention_days: int = Field(
        default=730,
        description="Days of model-usage minute rollups retained before cleanup",
    )
    model_usage_reconcile_minutes: int = Field(
        default=2880,
        description="Trailing window (minutes) rebuilt by the reconcile task",
    )
    model_usage_reconcile_chunk_minutes: int = Field(
        default=60,
        description="Maximum UTC-minute span reconciled in one database transaction",
    )
    model_usage_cleanup_batch_size: int = Field(
        default=5000,
        description="Batch size for model-usage retention deletes",
    )
    model_usage_retry_max_attempts: int = Field(
        default=5,
        description="Maximum failed-write retry attempts for a ledger event",
    )
    model_usage_retry_base_seconds: int = Field(
        default=10,
        description="Base delay for exponential failed-write retry backoff",
    )
    model_usage_user_hash_secret: str = Field(
        default="",
        description="Secret keying the per-user hash for model-usage identity",
    )
    model_usage_health_lookback_minutes: int = Field(
        default=60,
        description="Complete-minute lookback used by model-usage operational health",
    )
    model_usage_health_unattributed_degraded_ratio: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Unattributed-attempt ratio above which usage health is degraded",
    )
    model_usage_health_rollup_lag_degraded_minutes: int = Field(
        default=2,
        ge=0,
        description="Durable event-to-rollup lag above which usage health is degraded",
    )
    model_usage_health_rollup_lag_unhealthy_minutes: int = Field(
        default=5,
        ge=0,
        description="Durable event-to-rollup lag above which usage health is unhealthy",
    )
    model_usage_health_failure_window_seconds: int = Field(
        default=300,
        le=3_600,
        description="Recent deployment-shared persistence-failure health window",
    )
    model_usage_failure_store_ttl_seconds: int = Field(
        default=900,
        ge=60,
        le=86_400,
        description="TTL for content-free Redis model-usage failure minute buckets",
    )
    model_usage_failure_store_timeout_seconds: float = Field(
        default=0.25,
        gt=0,
        description="Connect/read timeout for best-effort usage failure counters",
    )

    chat_history_max_messages: int = Field(
        default=24,
        description="Maximum prior messages to include when building chat prompts (0 = no limit)",
    )
    chat_history_max_tokens: int = Field(
        default=9000,
        description=(
            "Approximate maximum tokens of chat history to include in prompts (0 = no limit)"
        ),
    )
    rag_history_max_messages: int = Field(
        default=12,
        description="Maximum prior messages to include when building RAG prompts (0 = no limit)",
    )
    rag_history_max_tokens: int = Field(
        default=3000,
        description=(
            "Approximate maximum tokens of RAG history to include in prompts (0 = no limit)"
        ),
    )

    suppress_internal_stream_chunks: bool = Field(
        default=True,
        description="When True, stream chunks tagged as 'internal' "
        "(e.g. summarization node output) "
        "are silently dropped before being forwarded to clients. "
        "Disable only for debugging.",
    )

    # Redis Configuration
    redis_url: str = Field(
        default="",
        description="Redis connection URL used for widget runtime state, HITL timeout "
        "tracking, and other shared-state features. "
        "Required for live widget flows when the widgets MCP server runs out-of-process. "
        "Falls back to celery_broker_url if blank.",
    )
    redis_password: str = Field(
        default="",
        description="Optional Redis password convenience variable for local Docker setups. "
        "When set, it is automatically injected into redis:// and rediss:// URLs that "
        "omit credentials.",
    )
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL",
    )

    # Celery worker startup configuration.
    # Windows local development: pool=auto resolves to 'threads' for real
    # concurrency. Linux production: pool=auto resolves to 'prefork'.
    celery_worker_pool: str = Field(
        default="auto",
        description=(
            "Celery worker pool. 'auto' resolves to 'threads' on Windows and "
            "'prefork' on Linux. Set to 'solo' for single-task debug runs."
        ),
    )
    celery_worker_concurrency: int = Field(
        default=2,
        description="Number of worker processes/threads handling tasks concurrently.",
    )
    celery_worker_prefetch_multiplier: int = Field(
        default=1,
        description="Celery worker prefetch multiplier (1 = no over-fetch).",
    )
    celery_worker_max_tasks_per_child: int = Field(
        default=10,
        description="Recycle worker process after this many tasks to bound memory growth.",
    )
    celery_worker_time_limit: int = Field(
        default=300,
        description="Hard task time limit in seconds.",
    )
    celery_worker_soft_time_limit: int = Field(
        default=240,
        description="Soft task time limit in seconds (raises SoftTimeLimitExceeded).",
    )
    celery_worker_cancel_long_running_tasks_on_connection_loss: bool = Field(
        default=True,
        description=(
            "Cancel late-acknowledged running tasks when the broker connection is lost. "
            "This avoids duplicate concurrent execution after Redis reconnect/redelivery."
        ),
    )
    celery_broker_health_check_interval: int = Field(
        default=30,
        description="Redis broker socket health-check interval in seconds.",
    )
    celery_broker_visibility_timeout: int = Field(
        default=3600,
        description=(
            "Redis broker visibility timeout in seconds for late-acknowledged tasks. "
            "Must exceed expected task runtime."
        ),
    )
    celery_broker_socket_keepalive: bool = Field(
        default=True,
        description="Enable TCP keepalive on Redis broker sockets.",
    )
    celery_broker_retry_on_timeout: bool = Field(
        default=True,
        description="Retry Redis broker operations that fail with socket timeouts.",
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
            "Base URL of a running mineru-api service (e.g. 'http://localhost:8765'). "
            "When set, workers skip the per-document subprocess cold start. "
            "Use with MINERU_BACKEND=hybrid-http-client or vlm-http-client. "
            "Start the service with scripts/start_mineru_service.ps1."
        ),
    )
    mineru_backend: str = Field(
        default="pipeline",
        description=(
            "MinerU parse backend. Options: 'pipeline', 'hybrid-engine', "
            "'hybrid-http-client', 'vlm-engine', 'vlm-http-client'. "
            "Legacy names 'hybrid-auto-engine'/'vlm-auto-engine' are still accepted. "
            "Use 'hybrid-http-client' or 'vlm-http-client' when MINERU_API_URL is set. "
            "Hybrid backends default to '--effort medium', which skips image/chart "
            "analysis; add ['--effort', 'high'] to MINERU_EXTRA_ARGS to re-enable it."
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
            "Optional MinerU OCR language for pipeline/hybrid backends (for example: 'ch', "
            "'korean', 'th'). Since MinerU 3.4 'en', 'japan', 'chinese_cht', and 'latin' "
            "are remapped to the unified 'ch' model."
        ),
    )
    extract_formulas_from_pdf: bool = Field(
        default=True,
        description="Enable formula extraction from PDF documents",
    )
    mineru_extra_args: list[str] = Field(
        default_factory=list,
        description=(
            "Extra CLI arguments forwarded verbatim to the mineru command "
            "(e.g. ['--device', 'cpu'])"
        ),
    )
    document_images_storage_path: str = Field(
        default="app/storage/document_images",
        description="Storage path for extracted document images",
    )
    chat_images_storage_path: str = Field(
        default="app/storage/chat_images",
        description="Content-addressed storage root for externalized chat image bytes.",
    )
    chat_image_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum decoded byte size accepted when externalizing a chat image.",
    )
    chat_image_history_rehydrate_limit: int = Field(
        default=4,
        description=(
            "Max stored image references resolved back to bytes for the model "
            "per request. Bounds storage reads and injected base64. 0 = unlimited."
        ),
    )
    subagent_event_queue_maxsize: int = Field(
        default=512,
        description=(
            "Soft cap on queued subagent stream events. When saturated, transient "
            "frames (image previews, message deltas) are dropped to bound memory "
            "under a slow client; lifecycle events are never dropped."
        ),
    )
    parse_artifacts_storage_path: str = Field(
        default="app/storage/parse_artifacts",
        description="Directory where parse artifact JSON files are stored.",
    )
    celery_index_time_limit: int = Field(
        default=600,
        description="Hard time limit in seconds for index_document_task.",
    )
    celery_parse_concurrency: int = Field(
        default=2,
        description="Number of concurrent workers for the parse queue.",
    )
    celery_index_concurrency: int = Field(
        default=8,
        description="Number of concurrent workers for the index queue.",
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
    rag_hybrid_retrieval_enabled: bool = Field(
        default=False,
        description="Enable rank-fused dense and PostgreSQL lexical retrieval.",
    )
    rag_dense_candidate_limit: int = Field(
        default=40,
        ge=1,
        description="Dense candidates considered before authorization and fusion.",
    )
    rag_lexical_candidate_limit: int = Field(
        default=40,
        ge=1,
        description="Lexical candidates considered before authorization and fusion.",
    )
    rag_rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal-rank-fusion smoothing constant.",
    )
    rag_exact_cache_enabled: bool = Field(
        default=False,
        description="Enable exact-match Redis caches for RAG document/query embeddings "
        "and retrieval results. Disabled by default; ignored (cache stays disabled) "
        "when redis_url is blank.",
    )
    rag_query_embedding_cache_ttl_seconds: int = Field(
        default=300,
        ge=1,
        description="TTL for cached query embeddings, keyed by exact normalized query.",
    )
    rag_retrieval_cache_ttl_seconds: int = Field(
        default=60,
        ge=1,
        description="TTL for cached retrieval fusion results, keyed by the active "
        "index generation fingerprint.",
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
        description=(
            "Approximate maximum tokens of search history to include in prompts (0 = no limit)"
        ),
    )

    # Planning Agent History Configuration
    planning_history_max_messages: int = Field(
        default=16,
        description=(
            "Maximum prior messages to include when building planning prompts (0 = no limit)"
        ),
    )
    planning_history_max_tokens: int = Field(
        default=5000,
        description=(
            "Approximate maximum tokens of planning history to include in prompts (0 = no limit)"
        ),
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
    max_handoff_delegation_depth: int = Field(
        default=5,
        description="Maximum inter-agent handoffs permitted in one user turn",
    )
    react_agent_recursion_limit: int = Field(
        default=105,
        description=(
            "LangGraph recursion limit for agent execution. Must leave headroom for "
            "route/agent/tool cycles plus the final no-tools synthesis turn "
            "(minimum used at runtime: 2 * react_agent_max_iterations + 5)."
        ),
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
    tool_execution_consecutive_errors_limit: int = Field(
        default=3,
        description=(
            "Maximum repeated same tool/error/argument outputs before forcing "
            "final no-tools synthesis."
        ),
    )
    tool_validation_enabled: bool = Field(
        default=True,
        description="Enable/disable Pydantic validation for tool arguments and results",
    )

    # Tool Execution Policy Configuration (origin-aware timeout/retry policy)
    tool_execution_policies: dict[str, ToolExecutionPolicyOverride] = Field(
        default_factory=dict,
        description=(
            "Deployment-configured tool execution policy overrides, keyed by "
            "diagnostic rule name (the key is not identity — see match)."
        ),
    )
    tool_execution_max_interactive_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Global wall-clock cap for any single interactive tool call; "
            "participates as the final accumulated timeout cap."
        ),
    )
    tool_execution_cancellation_grace_seconds: float = Field(
        default=2.0,
        ge=0,
        allow_inf_nan=False,
        description=(
            "Grace period reserved between a tool's soft timeout and its hard "
            "timeout for cooperative cancellation to complete."
        ),
    )
    tool_execution_client_execution_grace_seconds: float = Field(
        default=2.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Seconds subtracted from the server soft timeout to derive the "
            "client-side execution deadline for client-runtime tools."
        ),
    )
    tool_execution_client_response_grace_seconds: float = Field(
        default=1.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Seconds subtracted from the server soft timeout to derive the "
            "server-side bridge response deadline for client-runtime tools."
        ),
    )

    # Tool Result Token Management
    tool_result_max_chars: int = Field(
        default=16000,
        description=(
            "Maximum characters to include in ToolMessage content sent to model "
            "(0 = no limit). Full output is preserved in artifacts for UI."
        ),
    )
    tool_result_truncation_suffix: str = Field(
        default="\n\n[Output truncated - full result available in tool artifacts]",
        description="Suffix to append when tool result is truncated",
    )

    tool_result_offload_enabled: bool = Field(
        default=True,
        description=(
            "Persist full large tool outputs outside model-visible ToolMessages and "
            "return a preview plus blob_id."
        ),
    )
    tool_result_offload_threshold_chars: int = Field(
        default=16000,
        description="Character count above which full tool output is offloaded.",
    )
    tool_result_offload_preview_chars: int = Field(
        default=4000,
        description="Preview characters kept inline after a tool result is offloaded.",
    )
    tool_result_offload_answer_share: float = Field(
        default=0.25,
        ge=0.0,
        le=0.9,
        description=(
            "Share of the preview budget reserved for a synthesized answer before "
            "results are allocated."
        ),
    )
    tool_result_offload_min_result_chars: int = Field(
        default=200,
        ge=1,
        description=(
            "Per-result content floor in a preview. Below this, later results are "
            "dropped whole instead of shrinking every result into uselessness."
        ),
    )
    tool_result_read_max_chars: int = Field(
        default=8000,
        ge=1,
        description="Maximum characters returned by one read_tool_result call.",
    )
    research_max_search_calls_per_turn: int = Field(
        default=2,
        ge=1,
        description=(
            "Distinct Tavily network requests allowed per user turn. Further calls "
            "return the accumulated research result instead of searching again."
        ),
    )
    research_max_image_searches_per_turn: int = Field(
        default=3,
        ge=1,
        description=(
            "Distinct visual subjects an answer may search for per turn. One "
            "subject per request: an answer wanting a thing's identity art and a "
            "shot of it in use needs two, while a how-to usually needs one. A "
            "repeat of a subject already searched is refused rather than counted."
        ),
    )
    research_near_duplicate_threshold: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description=(
            "Token-set overlap, as a share of the smaller query, above which a "
            "research query reuses the existing result."
        ),
    )
    research_budget_enabled: bool = Field(
        default=True,
        description="Kill switch for turn-local research dedup and call caps.",
    )
    remote_image_enrichment_enabled: bool = Field(
        default=True,
        description="Enable Brave-backed remote image enrichment for rich responses.",
    )
    rich_image_gallery_max_items: int = Field(
        default=6,
        ge=2,
        le=8,
        description=(
            "Maximum images in one provider-native discovery gallery grid. "
            "Only reachable through "
            "image_intent='gallery'; figure mode stays bound by "
            "rich_auto_place_max_images."
        ),
    )
    tool_result_blob_storage_dir: str = Field(
        default="data/tool_result_blobs",
        description=(
            "Legacy read-only directory for blobs created before content moved "
            "to Postgres. New blobs are stored in the tool_result_blobs table."
        ),
    )

    context_overflow_retry_enabled: bool = Field(
        default=True,
        description=(
            "Retry one model call with compacted tool messages when a provider rejects "
            "the prompt for context length."
        ),
    )
    context_overflow_retry_tool_preview_chars: int = Field(
        default=4000,
        description="Characters retained per ToolMessage during context-overflow retry.",
    )

    enable_user_memory_tools: bool = Field(
        default=False,
        description="Enable explicit agent-editable user memory tools.",
    )
    user_memory_max_prompt_items: int = Field(
        default=20,
        description="Maximum user memory items exposed to agents when memory tools are enabled.",
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
    rag_grounded_answer_gate_enabled: bool = Field(
        default=False,
        description=(
            "Enforce the grounded-answer gate on RAG final responses: constrained citation "
            "prompting, one regeneration, and explicit abstention. Disabled keeps the current "
            "final response and only records validation shadow metrics."
        ),
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
    runtime_time_context_timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone used for server-generated runtime time context in model prompts. "
            "UTC is always included as the canonical timestamp."
        ),
    )

    # Human-in-the-Loop Configuration
    enable_human_in_the_loop: bool = Field(
        default=True,
        description="Toggle to enable/disable human-in-the-loop globally",
    )
    hitl_tools_require_approval: list[str] = Field(
        default=[],
        description=(
            "List of tool names that require human approval. Empty list means NO tools "
            "require approval when HITL is enabled."
        ),
    )
    hitl_approval_timeout_minutes: int = Field(
        default=30,
        description=(
            "Timeout in minutes for pending approval requests. After timeout, the "
            "workflow can be auto-rejected or cleaned up."
        ),
    )

    # Gemini Thinking Configuration
    enable_thinking: bool = Field(
        default=True,
        description="Enable Gemini thinking mode for models that support it",
    )
    include_thoughts_in_response: bool = Field(
        default=True,
        description=(
            "Include thought summaries in streaming responses when thinking mode is enabled"
        ),
    )
    thinking_level: str = Field(
        default="high",
        description="Thinking level for Gemini 3 models (minimal, low, medium, high).",
    )
    thinking_budget: int = Field(
        default=-1,
        description=(
            "Thinking budget for Gemini 2.5 models (-1 for dynamic, 0 to disable, or "
            "specific token count like 1024)."
        ),
    )
    chat_agent_thinking_level: str = Field(
        default="low",
        description=(
            "Gemini 3 thinking_level for chat_agent specifically. chat_agent only "
            "routes/hands off or answers simple conversation, so it runs below the "
            "global thinking_level to cut latency; search/rag agents keep the global "
            "level for synthesis quality. A per-request reasoning_effort still overrides this."
        ),
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
    agentic_rag_max_images: int = Field(
        default=6,
        ge=1,
        description=(
            "Maximum document images attached to one agentic RAG turn after "
            "retrieval, applied by RAGImageSelector."
        ),
    )
    rag_vision_max_bytes: int = Field(
        default=8 * 1024 * 1024,
        gt=0,
        description=(
            "Maximum combined encoded byte size for images selected for one "
            "agentic RAG turn (Task 11 bounded multimodal retrieval)."
        ),
    )
    rag_vision_max_pixels: int = Field(
        default=40_000_000,
        gt=0,
        description=(
            "Maximum combined decoded pixel count for images selected for one "
            "agentic RAG turn. Oversized individual images are resized down "
            "to fit rather than dropped."
        ),
    )
    rag_vision_max_tokens: int = Field(
        default=4096,
        gt=0,
        description=(
            "Maximum combined estimated vision-token cost for images selected "
            "for one agentic RAG turn. A conservative, provider-agnostic "
            "budgeting heuristic, not an exact provider accounting."
        ),
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
        description=(
            "Fraction of the loop budget to consume per round before rolling to the "
            "next round (0.1-1.0)"
        ),
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
        default=0,
        description=(
            "Maximum planning tool calls before pausing for user. "
            "Set to 0 to disable the planning iteration budget."
        ),
    )
    planning_consecutive_errors_limit: int = Field(
        default=3,
        description="Maximum consecutive planning tool errors before stopping",
    )
    planning_rubric_enabled: bool = Field(
        default=True,
        description=(
            "Enable native Planning rubric grading. When enabled, generated or "
            "modified todo plans are evaluated against a planning-quality rubric "
            "and revised before persistence or final response when possible."
        ),
    )
    planning_rubric_max_iterations: int = Field(
        default=3,
        description=(
            "Maximum Planning rubric grading passes per attempt. Minimum 1. "
            "Set planning_rubric_enabled=False to disable grading."
        ),
    )

    # Planning-mode subagent dispatcher configuration
    planning_subagents_enabled: bool = Field(
        default=True,
        description=(
            "Enable the Planning-mode subagent dispatch tool. When True, the Planning "
            "Agent can receive a `dispatch_subagents` internal tool while Planning "
            "mode is active so it can fan out independent worker tasks to other graph "
            "agents. Prompt policy controls when the tool should be used."
        ),
    )

    # MCP Tool Search Configuration (Deferred Loading)
    mcp_tool_search_enabled: bool = Field(
        default=True,
        description=(
            "Enable deferred MCP tool loading via tool_search. When enabled, only "
            "tool_search + pinned tools are bound by default."
        ),
    )
    mcp_tool_search_default_top_k: int = Field(
        default=3,
        description="Default public candidates returned for tool_search discovery queries.",
    )
    mcp_tool_search_max_top_k: int = Field(
        default=100,
        description=(
            "Maximum explicit top_k accepted for tool_search discovery queries. "
            "Keep high for compatibility; default output stays compact."
        ),
    )
    mcp_tool_search_description_max_chars: int = Field(
        default=120,
        description="Maximum characters in model-facing tool_search descriptions.",
    )
    mcp_tool_search_match_reasons_max: int = Field(
        default=2,
        description="Maximum match reasons exposed per search result.",
    )
    mcp_tool_search_debug_scores: bool = Field(
        default=False,
        description="Include explicit tool_search score diagnostics in discovery output.",
    )
    mcp_tool_search_autoload_top_k: int = Field(
        default=1,
        description=(
            "Maximum high-confidence recommended tools to automatically load after tool_search."
        ),
    )
    mcp_tool_search_inventory_default_top_k: int = Field(
        default=20,
        description=(
            "Default number of tools to return per server in per-server inventory mode "
            "(tool_search with server_name but no query)."
        ),
    )
    mcp_tool_search_inventory_max_top_k: int = Field(
        default=50,
        description="Maximum allowed top_k for inventory mode (per-server tool listing).",
    )
    mcp_tool_search_pinned_tools: list[str] = Field(
        default=[],
        description=(
            "Tool names (or server::tool_name) that are always bound, not deferred. "
            "Recommended 3-5 high-frequency tools."
        ),
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
        description=(
            "Minimum relevance score for a tool to appear in search results. Tools "
            "below this threshold are excluded entirely."
        ),
    )
    mcp_tool_search_autoload_min_relevance_score: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Minimum relevance score for a tool to be autoloaded. Stricter than "
            "min_relevance_score to prevent arbitrary autoloading."
        ),
    )
    mcp_tool_search_log_queries: bool = Field(
        default=False,
        description=(
            "Log tool_search queries (disable in production to avoid logging sensitive queries)."
        ),
    )

    # Client Runtime Bridge Configuration
    enable_client_runtime_bridge: bool = Field(
        default=True,
        description="Enable the client-runtime bridge for per-device client backends. "
        "When enabled, the server can dispatch tool calls to connected client devices.",
    )
    client_runtime_ws_timeout_seconds: int = Field(
        default=60,
        description=(
            "Timeout in seconds for client runtime WebSocket operations (tool dispatch, heartbeat)."
        ),
    )
    client_runtime_catalog_cache_ttl_seconds: int = Field(
        default=300,
        description="TTL in seconds for caching client device tool/skill catalogs. "
        "Catalogs are refreshed when a device reconnects or explicitly syncs.",
    )
    client_runtime_require_connected_device_for_local_tools: bool = Field(
        default=True,
        description="When True, tool calls targeting client-local tools fail if no "
        "device is connected. "
        "When False, such calls return a recoverable error allowing the model to adapt.",
    )
    client_runtime_heartbeat_interval_seconds: int = Field(
        default=30,
        description="Expected heartbeat interval from connected client devices. "
        "Devices not sending heartbeats within 2x this interval are marked offline.",
    )
    client_runtime_max_tool_result_size_bytes: int = Field(
        default=1048576,
        description="Maximum size in bytes for tool results returned from client "
        "devices (1MB default). "
        "Results exceeding this are truncated with a warning.",
    )

    # Inline Rich Response Configuration
    inline_rich_response_enabled: bool = Field(
        default=True,
        description=(
            "Kill switch for the inline rich-response feature. When False, the "
            "backend never emits marker-bearing v1 content or rich-item stream "
            "events, regardless of the per-request capability."
        ),
    )
    rich_item_inventory_max_items: int = Field(
        default=12,
        description=(
            "Maximum rich-item candidates exposed to the model in the bounded inventory block."
        ),
    )
    rich_item_inventory_max_chars: int = Field(
        default=2400,
        description=(
            "Maximum characters of the bounded rich-item inventory block injected into prompts."
        ),
    )
    rich_item_summary_max_chars: int = Field(
        default=180,
        description="Maximum characters used per rich-item summary line in the inventory block.",
    )
    rich_item_selected_image_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        description=(
            "Maximum decoded byte size accepted for a selected inline base64 image payload. "
            "Oversized inline data is rejected at finalization rather than persisted or streamed."
        ),
    )
    rich_image_candidate_max_count: int = Field(
        default=8,
        description="Maximum normalized web-image candidates retained per tool result.",
    )
    rich_image_min_width_px: int = Field(
        default=320,
        description="Reject provider images with a known width below this value.",
    )
    rich_image_min_height_px: int = Field(
        default=180,
        description="Reject provider images with a known height below this value.",
    )
    rich_image_min_aspect_ratio: float = Field(
        default=0.2,
        gt=0,
        description=(
            "Reject provider images narrower than this width/height ratio when both "
            "dimensions are known. Deliberately loose: a tight photo-shaped band "
            "rejects tall infographics, screenshots, and flowcharts."
        ),
    )
    rich_image_max_aspect_ratio: float = Field(
        default=5.0,
        gt=0,
        description=(
            "Reject provider images wider than this width/height ratio when both "
            "dimensions are known. Catches hero strips the minimum-dimension "
            "gates miss; still admits panoramas and wide charts."
        ),
    )
    rich_image_group_max_items: int = Field(
        default=3,
        ge=2,
        le=3,
        description=(
            "Maximum cells in a legacy raw-tool image_group. Provider-native "
            "galleries use rich_image_gallery_max_items instead. The renderer "
            "wraps groups after three columns without dropping approved cells."
        ),
    )
    web_image_fetch_connect_timeout_seconds: float = Field(
        default=2.0,
        gt=0,
        description="Connect timeout for render-time selected web-image retrieval.",
    )
    web_image_fetch_read_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Read timeout for render-time selected web-image retrieval.",
    )
    web_image_fetch_max_redirects: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Maximum independently validated redirects for selected web images.",
    )
    web_image_fetch_max_bytes: int = Field(
        default=5 * 1024 * 1024,
        gt=0,
        description="Maximum response bytes accepted from a selected web-image upstream.",
    )
    web_image_fetch_max_pixels: int = Field(
        default=25_000_000,
        gt=0,
        description="Maximum decoded pixel count accepted for a selected web image.",
    )
    rich_auto_place_enabled: bool = Field(
        default=True,
        description=(
            "Deterministically insert inline markers for relevant unreferenced "
            "rich items (images, widgets) into the final answer at persistence time."
        ),
    )
    rich_auto_place_max_images: int = Field(
        default=2,
        description=(
            "Maximum image items per answer. Governs both the model-facing "
            "inventory (a group counts as one) and the anchoring path."
        ),
    )
    rich_auto_place_min_score: float = Field(
        default=0.25,
        description=(
            "Minimum keyword-overlap score (fraction of an item's descriptive tokens "
            "found in a paragraph) required to auto-place the item after that paragraph. "
            "Governs widget auto-placement only; images use "
            "RICH_IMAGE_ANCHOR_MIN_SCORE on the query-anchoring path. The score is "
            "quantized by token count, so at the default a widget title of four tokens "
            "or fewer needs exactly one matching token — the same as any match at all."
        ),
    )
    rich_image_anchor_min_score: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of image-query tokens a paragraph must contain to receive that "
            "image's marker directly. This selects the placement, not whether the image "
            "appears: below it the image anchors at the first substantial paragraph "
            "instead, and only an image matching no paragraph at all is dropped. The "
            "score is quantized by query length, so at the default a one- or two-token "
            "query cannot land below the threshold, and a three- to five-token query "
            "needs two matching tokens."
        ),
    )

    # ── Validators ──────────────────────────────────────────────────────

    @field_validator(
        "react_agent_max_iterations",
        "max_handoff_delegation_depth",
        "react_agent_recursion_limit",
        "planning_rubric_max_iterations",
        "agentic_max_iterations",
        "auto_continue_max_rounds",
        "auto_continue_max_total_iterations",
        "auto_continue_timeout_seconds",
        "planning_consecutive_errors_limit",
        "tool_execution_consecutive_errors_limit",
        "celery_worker_concurrency",
        "celery_worker_prefetch_multiplier",
        "celery_worker_max_tasks_per_child",
        "celery_worker_time_limit",
        "celery_worker_soft_time_limit",
        "celery_broker_health_check_interval",
        "celery_broker_visibility_timeout",
        "brave_image_search_default_count",
        "brave_image_search_max_count",
        "conversation_summary_timeout_seconds",
        "conversation_summary_max_attempts",
        "conversation_summary_lease_seconds",
        "conversation_summary_retry_base_seconds",
        "conversation_summary_retry_max_seconds",
        "conversation_summary_reconcile_seconds",
        "conversation_summary_max_tokens",
        "model_usage_raw_retention_days",
        "model_usage_rollup_retention_days",
        "model_usage_reconcile_minutes",
        "model_usage_reconcile_chunk_minutes",
        "model_usage_cleanup_batch_size",
        "model_usage_retry_max_attempts",
        "model_usage_retry_base_seconds",
        "model_usage_health_lookback_minutes",
        "model_usage_health_failure_window_seconds",
        "model_usage_failure_store_ttl_seconds",
        "chat_image_max_bytes",
        "subagent_event_queue_maxsize",
        "rich_image_candidate_max_count",
        "rich_image_min_width_px",
        "rich_image_min_height_px",
        mode="before",
    )
    @classmethod
    def _positive_int(cls, v: int) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("brave_image_search_timeout_seconds", mode="before")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        v = float(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("brave_image_search_default_safesearch", mode="before")
    @classmethod
    def _validate_brave_safesearch(cls, v: str) -> str:
        allowed = {"off", "strict"}
        value = str(v).strip().lower()
        if value not in allowed:
            raise ValueError(
                f"brave_image_search_default_safesearch must be one of {sorted(allowed)}"
            )
        return value

    @field_validator(
        "tavily_search_default_max_results",
        "tavily_search_max_results",
        "tavily_extract_max_urls",
        "tavily_map_max_depth",
        "tavily_map_max_breadth",
        "tavily_map_limit",
        "tavily_crawl_max_depth",
        "tavily_crawl_max_breadth",
        "tavily_crawl_limit",
        mode="before",
    )
    @classmethod
    def validate_positive_tavily_int(cls, v):
        if v in (None, ""):
            return v
        parsed = int(v)
        if parsed < 1:
            raise ValueError("Tavily numeric settings must be positive")
        return parsed

    @field_validator("tavily_search_default_depth", mode="before")
    @classmethod
    def validate_tavily_search_depth(cls, v):
        value = str(v or "basic").strip().lower()
        if value not in {"basic", "fast", "ultra-fast", "advanced"}:
            raise ValueError(
                "TAVILY_SEARCH_DEFAULT_DEPTH must be basic, fast, ultra-fast, or advanced"
            )
        return value

    @field_validator("tavily_extract_default_depth", mode="before")
    @classmethod
    def validate_tavily_extract_depth(cls, v):
        value = str(v or "basic").strip().lower()
        if value not in {"basic", "advanced"}:
            raise ValueError("TAVILY_EXTRACT_DEFAULT_DEPTH must be basic or advanced")
        return value

    @field_validator("tavily_extract_default_format", mode="before")
    @classmethod
    def validate_tavily_extract_format(cls, v):
        value = str(v or "markdown").strip().lower()
        if value not in {"markdown", "text"}:
            raise ValueError("TAVILY_EXTRACT_DEFAULT_FORMAT must be markdown or text")
        return value

    @field_validator(
        "chat_history_max_messages",
        "chat_history_max_tokens",
        "rag_history_max_messages",
        "rag_history_max_tokens",
        "search_history_max_messages",
        "search_history_max_tokens",
        "planning_history_max_messages",
        "planning_history_max_tokens",
        "memory_max_messages",
        "memory_load_batch_size",
        "tool_result_max_chars",
        "mcp_tool_search_default_top_k",
        "mcp_tool_search_max_top_k",
        "mcp_tool_search_description_max_chars",
        "mcp_tool_search_match_reasons_max",
        "mcp_tool_search_autoload_top_k",
        "mcp_tool_search_inventory_default_top_k",
        "mcp_tool_search_inventory_max_top_k",
        "mcp_tool_search_max_pinned_tools",
        "mcp_tool_search_max_loaded_tools_per_conversation",
        "planning_max_iterations",
        "conversation_summary_trigger_messages",
        "conversation_summary_trigger_tokens",
        "conversation_summary_keep_recent_turns",
        "conversation_summary_safety_margin_tokens",
        "conversation_summary_default_reserved_output_tokens",
        "chat_image_history_rehydrate_limit",
        mode="before",
    )
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        v = int(v)
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @field_validator(
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
        if self.model_usage_reconcile_minutes >= self.model_usage_raw_retention_days * 1_440:
            raise ValueError(
                "model usage reconcile window must be strictly shorter than raw-event retention"
            )
        if self.model_usage_rollup_retention_days < self.model_usage_raw_retention_days:
            raise ValueError(
                "model usage rollup retention cannot be shorter than raw-event retention"
            )
        if (
            self.model_usage_failure_store_ttl_seconds
            < self.model_usage_health_failure_window_seconds + 60
        ):
            raise ValueError(
                "model usage failure store TTL must cover the health window plus one minute bucket"
            )
        if (
            self.model_usage_health_rollup_lag_unhealthy_minutes
            < self.model_usage_health_rollup_lag_degraded_minutes
        ):
            raise ValueError(
                "model usage unhealthy rollup lag must be greater than or equal "
                "to degraded rollup lag"
            )
        if (
            self.tool_execution_max_interactive_timeout_seconds
            <= self.tool_execution_cancellation_grace_seconds
        ):
            raise ValueError("tool execution maximum must exceed cancellation grace")
        if (
            self.tool_execution_client_execution_grace_seconds
            <= self.tool_execution_client_response_grace_seconds
        ):
            raise ValueError("client execution grace must exceed client response grace")

        selectors: dict[str, str] = {}
        for key, override in self.tool_execution_policies.items():
            selector = override.match.model_dump_json(exclude_none=True)
            if selector in selectors:
                raise ValueError(
                    f"duplicate tool execution policy selectors: {selectors[selector]!r}, {key!r}"
                )
            selectors[selector] = key
            if override.disable_outer_timeout:
                raise ValueError("deployment policy cannot disable the outer timeout")

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
                self.secret_key = _load_or_create_dev_secret_key()
                logging.getLogger(__name__).warning(
                    "SECRET_KEY is not set; using a persisted development key "
                    "(.dev_secret_key). Set SECRET_KEY explicitly for shared or "
                    "production environments so sessions stay valid across machines."
                )
            else:
                raise ValueError("secret_key must be set to a strong value outside development")
        if self.environment != "development" and self.api_debug:
            raise ValueError("api_debug must be disabled outside development")
        if (
            self.conversation_summary_enabled
            and self.conversation_summary_trigger_messages == 0
            and self.conversation_summary_trigger_tokens == 0
        ):
            raise ValueError(
                "conversation summaries require at least one background threshold when enabled"
            )
        if not (
            0.0
            < self.conversation_summary_soft_context_ratio
            < self.conversation_summary_hard_context_ratio
            < 1.0
        ):
            raise ValueError(
                "conversation summary soft context ratio must be greater than zero, "
                "below the hard context ratio, and the hard ratio must be below one"
            )
        if (
            self.conversation_summary_trigger_messages > 0
            and self.conversation_summary_keep_recent_turns
            >= self.conversation_summary_trigger_messages
        ):
            raise ValueError(
                "conversation summary keep recent turns must be below the enabled message threshold"
            )
        if (
            self.conversation_summary_retry_max_seconds
            < self.conversation_summary_retry_base_seconds
        ):
            raise ValueError(
                "conversation summary retry max must be greater than or equal to retry base"
            )
        if self.environment.strip().lower() == "production":
            if not self.conversation_summary_provider.strip():
                raise ValueError("conversation summary provider must be explicit in production")
            if not self.conversation_summary_model.strip():
                raise ValueError("conversation summary model must be explicit in production")
            if "preview" in self.conversation_summary_model.strip().lower():
                raise ValueError("conversation summary model must be stable in production")
            if self.langsmith_tracing and not self.model_usage_user_hash_secret.strip():
                raise ValueError(
                    "model_usage_user_hash_secret must be set in production when "
                    "LangSmith tracing is enabled"
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
