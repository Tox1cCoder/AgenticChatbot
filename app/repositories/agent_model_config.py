"""
Repository for AgentModelConfig database operations.

Handles CRUD operations for storing and retrieving user-specific persistent
per-agent provider/model selections.
"""

from uuid import UUID

from sqlalchemy import and_

from app.models.agent_model_config import AgentModelConfig


class AgentModelConfigRepository:
    """
    Repository for AgentModelConfig operations.
    """

    def __init__(self, session_factory: callable):
        self.session_factory = session_factory

    def get_by_user_and_agent_key(self, user_id: UUID, agent_key: str) -> AgentModelConfig | None:
        with self.session_factory() as session:
            return (
                session.query(AgentModelConfig)
                .filter(
                    and_(
                        AgentModelConfig.user_id == user_id,
                        AgentModelConfig.agent_key == agent_key,
                    )
                )
                .first()
            )

    def get_all_by_user(self, user_id: UUID) -> list[AgentModelConfig]:
        with self.session_factory() as session:
            return (
                session.query(AgentModelConfig)
                .filter(AgentModelConfig.user_id == user_id)
                .order_by(AgentModelConfig.updated_at.desc())
                .all()
            )

    def upsert(
        self,
        *,
        user_id: UUID,
        agent_key: str,
        provider_type: str,
        model: str,
        allow_custom_model: bool = False,
        temperature: float | None = None,
    ) -> AgentModelConfig:
        with self.session_factory() as session:
            existing = (
                session.query(AgentModelConfig)
                .filter(
                    and_(
                        AgentModelConfig.user_id == user_id,
                        AgentModelConfig.agent_key == agent_key,
                    )
                )
                .first()
            )

            if existing:
                existing.provider_type = provider_type
                existing.model = model
                existing.allow_custom_model = allow_custom_model
                existing.temperature = temperature
                session.commit()
                session.refresh(existing)
                return existing

            entity = AgentModelConfig(
                user_id=user_id,
                agent_key=agent_key,
                provider_type=provider_type,
                model=model,
                allow_custom_model=allow_custom_model,
                temperature=temperature,
            )
            session.add(entity)
            session.commit()
            session.refresh(entity)
            return entity

    def delete_all_by_user(self, user_id: UUID) -> int:
        with self.session_factory() as session:
            deleted = (
                session.query(AgentModelConfig)
                .filter(AgentModelConfig.user_id == user_id)
                .delete(synchronize_session=False)
            )
            session.commit()
            return int(deleted or 0)

    def delete_by_user_and_agent_key(self, user_id: UUID, agent_key: str) -> bool:
        with self.session_factory() as session:
            deleted = (
                session.query(AgentModelConfig)
                .filter(
                    and_(
                        AgentModelConfig.user_id == user_id,
                        AgentModelConfig.agent_key == agent_key,
                    )
                )
                .delete(synchronize_session=False)
            )
            session.commit()
            return bool(deleted)

    def delete_by_user_and_provider_type(self, user_id: UUID, provider_type: str) -> list[str]:
        with self.session_factory() as session:
            rows = (
                session.query(AgentModelConfig)
                .filter(
                    and_(
                        AgentModelConfig.user_id == user_id,
                        AgentModelConfig.provider_type == provider_type,
                    )
                )
                .all()
            )

            affected_agent_keys = [
                str(row.agent_key) for row in rows if getattr(row, "agent_key", None)
            ]
            for row in rows:
                session.delete(row)

            session.commit()
            return affected_agent_keys
