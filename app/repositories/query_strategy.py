"""
Query strategy pattern interfaces for repository read operations.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType")


class QueryStrategy(ABC, Generic[ModelType]):
    """Abstract strategy for query (read) operations"""

    @abstractmethod
    def get_by_id(self, db: Session, id: int | UUID) -> ModelType | None:
        """Get a record by ID"""
        pass

    @abstractmethod
    def exists(self, db: Session, id: int | UUID) -> bool:
        """Check if a record exists by ID"""
        pass


class DefaultQueryStrategy(QueryStrategy[ModelType]):
    """Default implementation of query operations strategy"""

    def __init__(self, model: type[ModelType]):
        self.model = model

    def _exclude_soft_deleted(self, statement):
        """Add the ``deleted_at IS NULL`` filter only for models that support
        soft delete. Models without a ``deleted_at`` column pass through
        unfiltered, so the generic strategy is safe for hard-delete models."""
        if hasattr(self.model, "deleted_at"):
            statement = statement.where(self.model.deleted_at.is_(None))
        return statement

    def get_by_id(self, db: Session, id: int | UUID) -> ModelType | None:
        """Get a record by ID (excluding soft deleted when supported)"""
        statement = self._exclude_soft_deleted(select(self.model).where(self.model.id == id))
        return db.execute(statement).scalar_one_or_none()

    def exists(self, db: Session, id: int | UUID) -> bool:
        """Check if a record exists by ID (excluding soft deleted when supported)"""
        statement = self._exclude_soft_deleted(select(self.model.id).where(self.model.id == id))
        return db.execute(statement).scalar() is not None
