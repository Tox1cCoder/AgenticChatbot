"""
Repository for AgentModelConfig database operations.

Handles CRUD operations for storing and retrieving user-specific persistent
per-agent provider/model selections.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_

from app.models.agent_model_config import AgentModelConfig


class AgentModelConfigRepository:
    """
    Repository for AgentModelConfig operations.
    """

    def __init__(self, session_factory: callable):
        self.session_factory = session_factory

    def get_by_user_and_agent_key(
        self, user_id: UUID, agent_key: str
    ) -> Optional[AgentModelConfig]:
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

    def get_all_by_user(self, user_id: UUID) -> List[AgentModelConfig]:
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
        temperature: Optional[float] = None,
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
                existing.temperature = temperature
                session.commit()
                session.refresh(existing)
                return existing

            entity = AgentModelConfig(
                user_id=user_id,
                agent_key=agent_key,
                provider_type=provider_type,
                model=model,
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

