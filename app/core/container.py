"""
Dependency Injection Container.
"""

from dependency_injector import containers, providers
from qdrant_client import QdrantClient

from app.ai.agents.planning_agent import PlanningAgent
from app.ai.checkpoint import CheckpointManager
from app.ai.conversation_summarizer import ConversationSummarizer
from app.ai.graph import create_workflow
from app.ai.history import ConversationHistoryProvider
from app.ai.mcp_integration import MCPManager
from app.ai.mcp_registry import MCPRegistry
from app.ai.planning_runtime_adapter import PlanningRuntimeAdapter
from app.core.config import settings
from app.core.dependency_injection import AppAutoInjector, AppContainerInjector
from app.database.database import Database
from app.interfaces import (
    IAuthService,
    IConversationService,
    IDocumentService,
    IFeedbackService,
    IMessageService,
    IUserService,
)
from app.interfaces.planning_runtime_interface import IPlanningRuntimeService
from app.interfaces.task_plan_service_interface import ITaskPlanService
from app.repositories.agent_model_config import AgentModelConfigRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.conversation_memory_summary import (
    ConversationMemorySummaryRepository,
)
from app.repositories.custom_agent import CustomAgentRepository
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.document_image import DocumentImageRepository
from app.repositories.document_parse_artifact import DocumentParseArtifactRepository
from app.repositories.feedback import FeedbackRepository
from app.repositories.hitl_interrupt import HITLInterruptRepository
from app.repositories.message import MessageRepository
from app.repositories.model_provider import ModelProviderRepository
from app.repositories.task_plan import TaskPlanRepository
from app.repositories.tool_approval import ToolApprovalRepository
from app.repositories.tool_result_blob import ToolResultBlobRepository
from app.repositories.user import UserRepository
from app.repositories.user_memory import UserMemoryRepository
from app.services.ai_service import AIService
from app.services.auth_service import AuthService
from app.services.conversation_service import ConversationService
from app.services.custom_agent_service import CustomAgentService
from app.services.document_chunk_builder import DocumentChunkBuilder
from app.services.document_index_service import DocumentIndexService
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import DocumentService
from app.services.feedback_service import FeedbackService
from app.services.generation_registry import get_generation_registry
from app.services.jwt_service import JwtService
from app.services.mcp_service import MCPService
from app.services.message_service import MessageService
from app.services.model_config_service import ModelConfigService
from app.services.provider_service import ProviderService
from app.services.rag_embedding_service import (
    GeminiRAGEmbeddingService,
    SentenceTransformerRAGEmbeddingService,
)
from app.services.task_plan_service import TaskPlanService
from app.services.tool_result_blob_service import ToolResultBlobService
from app.services.user_service import UserService
from app.utils.validation.conversation_validation import ConversationValidationUtils
from app.utils.validation.document_validation import DocumentValidationUtils
from app.utils.validation.feedback_validation import FeedbackValidationUtils
from app.utils.validation.message_validation import MessageValidationUtils
from app.utils.validation.task_plan_validation import TaskPlanValidationUtils
from app.utils.validation.user_validation import UserValidationUtils
from app.workers.celery_app import celery_app


class Container(containers.DeclarativeContainer):
    """Application dependency injection container."""

    wiring_config = containers.WiringConfiguration(
        modules=[
            "app.api.auth",
            "app.api.users",
            "app.api.conversations",
            "app.api.custom_agents",
            "app.api.messages",
            "app.api.feedback",
            "app.api.documents",
            "app.api.mcp",
            "app.api.task_plans",
            "app.api.ai_sdk",
            "app.api.providers",
            "app.api.model_config",
        ]
    )

    # Configuration
    config = providers.Configuration()

    # Database
    db = providers.Singleton(
        Database,
        db_url=settings.database_url,
    )

    # Qdrant client
    qdrant_client = providers.Singleton(
        QdrantClient,
        url=settings.qdrant_url,
    )

    # Active RAG embedding adapter. Selected at container-build time based on
    # ``rag_embedding_provider``. The Gemini path requires GEMINI_API_KEY and
    # uses ``gemini-embedding-2`` at the configured ``rag_embedding_dimension``.
    # The sentence_transformers fallback is for offline development only.
    def _build_rag_embedding_service():
        provider = (settings.rag_embedding_provider or "gemini").lower()
        if provider == "gemini":
            api_key = settings.gemini_api_key
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is required when rag_embedding_provider='gemini'"
                )
            if api_key.startswith("GEMINI_API_KEY="):
                api_key = api_key.split("=", 1)[-1].strip()
            return GeminiRAGEmbeddingService(
                api_key=api_key,
                model_name=settings.rag_embedding_model,
                dimension=settings.rag_embedding_dimension,
                query_task=settings.rag_embedding_query_task,
                embedding_batch_size=settings.rag_embedding_batch_size,
                embedding_max_concurrency=settings.rag_embedding_max_concurrency,
            )
        if provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            # Load offline-first: a cached model is revalidated against
            # huggingface.co on every construction unless local_files_only is
            # set, so a slow/unreachable hub times out even when cached.
            try:
                model = SentenceTransformer(
                    settings.rag_embedding_model,
                    device=device,
                    local_files_only=True,
                )
            except OSError:
                model = SentenceTransformer(settings.rag_embedding_model, device=device)
            return SentenceTransformerRAGEmbeddingService(
                model=model,
                model_name=settings.rag_embedding_model,
                dimension=settings.rag_embedding_dimension,
            )
        raise ValueError(f"Unknown rag_embedding_provider: {settings.rag_embedding_provider}")

    rag_embedding_service = providers.Singleton(_build_rag_embedding_service)

    # JWT Service
    jwt_service = providers.Factory(
        JwtService,
    )

    # Checkpoint manager
    checkpoint_manager = providers.Singleton(
        CheckpointManager,
        db_url=settings.database_url,
        settings=providers.Object(settings),
    )

    # MCP Manager - uses MCPRegistry to share instance with agents
    # This returns the sync accessor; async initialization happens via get_manager_async()
    mcp_manager = providers.Singleton(
        lambda: MCPRegistry.get_manager_sync() or MCPManager(),
    )

    # Repositories - use session factory from database
    user_repository = providers.Factory(
        UserRepository,
        session_factory=db.provided.session,
    )

    conversation_repository = providers.Factory(
        ConversationRepository,
        session_factory=db.provided.session,
    )

    custom_agent_repository = providers.Factory(
        CustomAgentRepository,
        session_factory=db.provided.session,
    )

    message_repository = providers.Factory(
        MessageRepository,
        session_factory=db.provided.session,
    )

    feedback_repository = providers.Factory(
        FeedbackRepository,
        session_factory=db.provided.session,
    )

    document_repository = providers.Factory(
        DocumentRepository,
        session_factory=db.provided.session,
    )

    document_image_repository = providers.Factory(
        DocumentImageRepository,
        session_factory=db.provided.session,
    )

    document_chunk_repository = providers.Factory(
        DocumentChunkRepository,
        session_factory=db.provided.session,
    )

    document_parse_artifact_repository = providers.Factory(
        DocumentParseArtifactRepository,
        session_factory=db.provided.session,
    )

    task_plan_repository = providers.Factory(
        TaskPlanRepository,
        session_factory=db.provided.session,
    )

    model_provider_repository = providers.Factory(
        ModelProviderRepository,
        session_factory=db.provided.session,
    )

    agent_model_config_repository = providers.Factory(
        AgentModelConfigRepository,
        session_factory=db.provided.session,
    )

    tool_approval_repository = providers.Factory(
        ToolApprovalRepository,
        session_factory=db.provided.session,
    )

    tool_result_blob_repository = providers.Factory(
        ToolResultBlobRepository,
        session_factory=db.provided.session,
    )

    tool_result_blob_service = providers.Singleton(
        ToolResultBlobService,
        repository=tool_result_blob_repository,
        storage_root=providers.Object(settings.tool_result_blob_storage_dir),
        threshold_chars=providers.Object(settings.tool_result_offload_threshold_chars),
        preview_chars=providers.Object(settings.tool_result_offload_preview_chars),
    )

    user_memory_repository = providers.Factory(
        UserMemoryRepository,
        session_factory=db.provided.session,
    )

    hitl_interrupt_repository = providers.Factory(
        HITLInterruptRepository,
        session_factory=db.provided.session,
    )

    conversation_memory_summary_repository = providers.Factory(
        ConversationMemorySummaryRepository,
        session_factory=db.provided.session,
    )

    history_provider = providers.Singleton(
        ConversationHistoryProvider,
        message_repository=message_repository,
        summary_repository=conversation_memory_summary_repository,
        settings=providers.Object(settings),
    )

    # Validation utils
    user_validation_utils = providers.Factory(
        UserValidationUtils,
        session_factory=db.provided.session,
    )
    conversation_validation_utils = providers.Factory(
        ConversationValidationUtils,
        session_factory=db.provided.session,
    )
    message_validation_utils = providers.Factory(
        MessageValidationUtils,
        session_factory=db.provided.session,
    )
    feedback_validation_utils = providers.Factory(
        FeedbackValidationUtils,
        session_factory=db.provided.session,
    )
    document_validation_utils = providers.Factory(
        DocumentValidationUtils,
        session_factory=db.provided.session,
    )
    task_plan_validation_utils = providers.Factory(
        TaskPlanValidationUtils,
        session_factory=db.provided.session,
    )

    provider_service = providers.Factory(
        ProviderService,
        provider_repository=model_provider_repository,
    )

    model_config_service = providers.Factory(
        ModelConfigService,
        repository=agent_model_config_repository,
        provider_service=provider_service,
    )

    # Durable summarizer resolves Gemini credentials directly from provider
    # settings so it never passes another provider's key to the Gemini summary
    # model.
    conversation_summarizer = providers.Singleton(
        ConversationSummarizer,
        settings=providers.Object(settings),
        provider_service=provider_service,
    )

    # Planning agent
    planning_agent = providers.Factory(
        PlanningAgent,
        runtime_model_resolver=model_config_service,
    )

    # Business services
    user_service: providers.Provider[IUserService] = providers.Factory(
        UserService,
        user_repository=user_repository,
        user_validation_utils=user_validation_utils,
    )

    def _get_checkpointer():
        if not settings.enable_langgraph_checkpoints:
            return None

        checkpoint_mgr = container.checkpoint_manager()
        return checkpoint_mgr.get_checkpointer()

    # AI service with conditional checkpoint injection
    def _create_ai_service():
        """Factory function to create AIService with conditional checkpointer."""
        checkpointer = Container._get_checkpointer()
        workflow_runtime = create_workflow(
            qdrant_client=container.qdrant_client(),
            embedding_service=container.rag_embedding_service(),
            checkpointer=checkpointer,
            document_repository=container.document_repository(),
            runtime_model_resolver=container.model_config_service(),
            history_provider=container.history_provider(),
        )

        return AIService(
            workflow_runtime=workflow_runtime,
            conversation_repository=container.conversation_repository(),
            checkpointer=checkpointer,
        )

    ai_service = providers.ThreadSafeSingleton(_create_ai_service)

    conversation_service: providers.Provider[IConversationService] = providers.Factory(
        ConversationService,
        conversation_repository=conversation_repository,
        user_validation_utils=user_validation_utils,
        conversation_validation_utils=conversation_validation_utils,
        ai_service=ai_service,
        checkpoint_manager=checkpoint_manager,
    )

    # Custom agents (per-user, per-conversation). generation_registry is the
    # module-level singleton so locks see the same in-flight/paused entries the
    # streaming and resume paths register.
    custom_agent_service = providers.Factory(
        CustomAgentService,
        repository=custom_agent_repository,
        conversation_validation_utils=conversation_validation_utils,
        model_config_service=model_config_service,
        generation_registry=providers.Callable(get_generation_registry),
    )

    planning_runtime_service: providers.Provider[IPlanningRuntimeService] = providers.Factory(
        PlanningRuntimeAdapter,
        planning_agent=planning_agent,
    )

    task_plan_service: providers.Provider[ITaskPlanService] = providers.Factory(
        TaskPlanService,
        task_plan_repository=task_plan_repository,
        conversation_validation_utils=conversation_validation_utils,
        task_plan_validation_utils=task_plan_validation_utils,
        planning_runtime=planning_runtime_service,
        conversation_repository=conversation_repository,
    )

    message_service: providers.Provider[IMessageService] = providers.Factory(
        MessageService,
        message_repository=message_repository,
        conversation_validation_utils=conversation_validation_utils,
        message_validation_utils=message_validation_utils,
        ai_service=ai_service,
        tool_approval_repository=tool_approval_repository,
        hitl_interrupt_repository=hitl_interrupt_repository,
        task_plan_service=task_plan_service,
        summary_repository=conversation_memory_summary_repository,
        conversation_summarizer=conversation_summarizer,
        custom_agent_service=custom_agent_service,
    )

    feedback_service: providers.Provider[IFeedbackService] = providers.Factory(
        FeedbackService,
        feedback_repository=feedback_repository,
        message_repository=message_repository,
        user_repository=user_repository,
        user_validation_utils=user_validation_utils,
        message_validation_utils=message_validation_utils,
        feedback_validation_utils=feedback_validation_utils,
    )

    auth_service: providers.Provider[IAuthService] = providers.Factory(
        AuthService,
        user_service=user_service,
        jwt_service=jwt_service,
    )

    document_index_service = providers.Factory(
        DocumentIndexService,
        chunk_repository=document_chunk_repository,
        qdrant_client=qdrant_client,
        embedding_service=rag_embedding_service,
        collection_name=settings.qdrant_collection_name,
        embedding_model_name=settings.rag_embedding_model,
        embedding_dimension=settings.rag_embedding_dimension,
        embedding_provider=settings.rag_embedding_provider,
        qdrant_upsert_batch_size=settings.qdrant_upsert_batch_size,
    )

    document_chunk_builder = providers.Factory(
        DocumentChunkBuilder,
        target_tokens=settings.rag_chunk_target_tokens,
        overlap_tokens=settings.rag_chunk_overlap_tokens,
        max_tokens=settings.rag_chunk_max_tokens,
    )

    document_processing_service = providers.Factory(
        DocumentProcessingService,
        settings=providers.Object(settings),
        celery_app=providers.Object(celery_app),
        document_image_repository=document_image_repository,
        document_index_service=document_index_service,
        document_chunk_builder=document_chunk_builder,
        document_parse_artifact_repository=document_parse_artifact_repository,
    )

    document_service: providers.Provider[IDocumentService] = providers.Factory(
        DocumentService,
        document_repository=document_repository,
        document_processing_service=document_processing_service,
        document_validation_utils=document_validation_utils,
        document_index_service=document_index_service,
    )

    mcp_service = providers.Factory(
        MCPService,
        mcp_manager=mcp_manager,
    )


# Initialize auto-injection wiring map before container instantiation
def setup_auto_injection(container_ref: Container | type[Container] | None = None):
    """Setup auto-injection wiring maps."""

    target = container_ref or Container
    AppAutoInjector.setup_wiring_map(target)
    AppContainerInjector.setup_wiring_map(target)


setup_auto_injection(Container)


# Create global container instance
container = Container()


def get_container() -> Container:
    """Get the global container instance."""
    return container
