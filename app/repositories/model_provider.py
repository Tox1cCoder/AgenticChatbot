"""
Repository for ModelProvider database operations.

Handles CRUD operations for storing and retrieving user-specific AI provider
API keys and configurations.
"""

from uuid import UUID

from sqlalchemy import and_

from app.models.model_provider import ModelProvider


class ModelProviderRepository:
    """
    Repository for ModelProvider operations.

    Provides methods for managing user-specific AI provider configurations
    including encrypted API keys and provider-specific metadata.
    """

    def __init__(self, session_factory: callable):
        """
        Args:
            session_factory: Callable that yields a SQLAlchemy session context manager
        """
        self.session_factory = session_factory

    def create(self, model_provider: ModelProvider) -> ModelProvider:
        """
        Args:
            model_provider: ModelProvider entity to create

        Returns:
            ModelProvider: Created provider with ID
        """
        with self.session_factory() as session:
            session.add(model_provider)
            session.commit()
            session.refresh(model_provider)
            return model_provider

    def get_by_id(self, provider_id: UUID) -> ModelProvider | None:
        """
        Args:
            provider_id: Provider UUID

        Returns:
            Optional[ModelProvider]: Provider if found, None otherwise
        """
        with self.session_factory() as session:
            return (
                session.query(ModelProvider)
                .filter(
                    and_(
                        ModelProvider.id == provider_id,
                        ModelProvider.deleted_at.is_(None),
                    )
                )
                .first()
            )

    def get_by_user_and_type(self, user_id: UUID, provider_type: str) -> ModelProvider | None:
        """
        Args:
            user_id: User UUID
            provider_type: Provider type ('gemini', 'openai', 'anthropic')

        Returns:
            Optional[ModelProvider]: Provider if found, None otherwise
        """
        with self.session_factory() as session:
            return (
                session.query(ModelProvider)
                .filter(
                    and_(
                        ModelProvider.user_id == user_id,
                        ModelProvider.provider_type == provider_type,
                        ModelProvider.deleted_at.is_(None),
                    )
                )
                .first()
            )

    def get_all_by_user(self, user_id: UUID) -> list[ModelProvider]:
        """
        Args:
            user_id: User UUID

        Returns:
            List[ModelProvider]: List of all providers for the user
        """
        with self.session_factory() as session:
            return (
                session.query(ModelProvider)
                .filter(
                    and_(
                        ModelProvider.user_id == user_id,
                        ModelProvider.deleted_at.is_(None),
                    )
                )
                .order_by(ModelProvider.created_at.desc())
                .all()
            )

    def get_default_provider(self, user_id: UUID) -> ModelProvider | None:
        """
        Args:
            user_id: User UUID

        Returns:
            Optional[ModelProvider]: Default provider if set, None otherwise
        """
        with self.session_factory() as session:
            return (
                session.query(ModelProvider)
                .filter(
                    and_(
                        ModelProvider.user_id == user_id,
                        ModelProvider.is_default,
                        ModelProvider.deleted_at.is_(None),
                    )
                )
                .first()
            )

    def _unset_default_providers(self, session, user_id: UUID) -> None:
        session.query(ModelProvider).filter(
            and_(
                ModelProvider.user_id == user_id,
                ModelProvider.is_default,
                ModelProvider.deleted_at.is_(None),
            )
        ).update({"is_default": False})

    def upsert(
        self,
        user_id: UUID,
        provider_type: str,
        api_key_encrypted: str,
        is_default: bool = False,
        provider_metadata: dict | None = None,
    ) -> ModelProvider:
        """
        Args:
            user_id: User UUID
            provider_type: Provider type ('gemini', 'openai', 'anthropic')
            api_key_encrypted: Encrypted API key
            is_default: Whether this should be the default provider
            provider_metadata: Optional provider-specific metadata

        Returns:
            ModelProvider: Created or updated provider
        """
        with self.session_factory() as session:
            if is_default:
                self._unset_default_providers(session, user_id)

            existing = (
                session.query(ModelProvider)
                .filter(
                    and_(
                        ModelProvider.user_id == user_id,
                        ModelProvider.provider_type == provider_type,
                        ModelProvider.deleted_at.is_(None),
                    )
                )
                .first()
            )

            if existing:
                existing.api_key_encrypted = api_key_encrypted
                existing.is_default = is_default
                if provider_metadata is not None:
                    existing.provider_metadata = provider_metadata
                session.commit()
                session.refresh(existing)
                return existing

            new_provider = ModelProvider(
                user_id=user_id,
                provider_type=provider_type,
                api_key_encrypted=api_key_encrypted,
                is_default=is_default,
                provider_metadata=provider_metadata or {},
            )
            session.add(new_provider)
            session.commit()
            session.refresh(new_provider)
            return new_provider

    def update(
        self,
        provider_id: UUID,
        api_key_encrypted: str | None = None,
        is_default: bool | None = None,
        provider_metadata: dict | None = None,
    ) -> ModelProvider | None:
        """
        Args:
            provider_id: Provider UUID
            api_key_encrypted: Optional new encrypted API key
            is_default: Optional new default status
            provider_metadata: Optional new metadata

        Returns:
            Optional[ModelProvider]: Updated provider if found, None otherwise
        """
        with self.session_factory() as session:
            provider = (
                session.query(ModelProvider)
                .filter(
                    and_(
                        ModelProvider.id == provider_id,
                        ModelProvider.deleted_at.is_(None),
                    )
                )
                .first()
            )
            if not provider:
                return None

            if is_default is True:
                self._unset_default_providers(session, provider.user_id)

            if api_key_encrypted is not None:
                provider.api_key_encrypted = api_key_encrypted
            if is_default is not None:
                provider.is_default = is_default
            if provider_metadata is not None:
                provider.provider_metadata = provider_metadata

            session.commit()
            session.refresh(provider)
            return provider

    def soft_delete(self, provider_id: UUID) -> bool:
        """
        Args:
            provider_id: Provider UUID

        Returns:
            bool: True if deleted, False if not found
        """
        with self.session_factory() as session:
            provider = (
                session.query(ModelProvider)
                .filter(
                    and_(
                        ModelProvider.id == provider_id,
                        ModelProvider.deleted_at.is_(None),
                    )
                )
                .first()
            )
            if not provider:
                return False

            from datetime import datetime, timezone

            provider.deleted_at = datetime.now(timezone.utc)
            session.commit()
            return True

    def delete_by_user_and_type(self, user_id: UUID, provider_type: str) -> bool:
        """
        Args:
            user_id: User UUID
            provider_type: Provider type

        Returns:
            bool: True if deleted, False if not found
        """
        with self.session_factory() as session:
            provider = (
                session.query(ModelProvider)
                .filter(
                    and_(
                        ModelProvider.user_id == user_id,
                        ModelProvider.provider_type == provider_type,
                        ModelProvider.deleted_at.is_(None),
                    )
                )
                .first()
            )
            if not provider:
                return False

            from datetime import datetime, timezone

            provider.deleted_at = datetime.now(timezone.utc)
            session.commit()
            return True
