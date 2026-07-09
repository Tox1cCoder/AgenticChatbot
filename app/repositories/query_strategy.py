"""
Query strategy pattern interfaces for repository read operations.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType")


class QueryStrategy(ABC, Generic[ModelType]):
    """Abstract strategy for query (read) operations"""

    @abstractmethod
    def get_by_id(self, db: Session, id: int | UUID) -> ModelType | None:
        """Get a record by ID"""
        pass

    @abstractmethod
    def get_all(self, db: Session, page: int = 1, limit: int = 10) -> list[ModelType]:
        """Get all records with page-based pagination"""
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

    def get_all(self, db: Session, page: int = 1, limit: int = 10) -> list[ModelType]:
        """Get all records with page-based pagination (excluding soft deleted when supported)"""
        offset = (page - 1) * limit
        statement = self._exclude_soft_deleted(select(self.model)).offset(offset).limit(limit)
        return list(db.execute(statement).scalars().all())

    def get_all_with_ordering(
        self,
        db: Session,
        page: int = 1,
        limit: int = 10,
        order_by: str | None = None,
        order_direction: str = "desc",
    ) -> list[ModelType]:
        """Get all records with page-based pagination and dynamic ordering
        (excluding soft deleted when supported)"""
        offset = (page - 1) * limit
        statement = self._exclude_soft_deleted(select(self.model))

        # Apply ordering
        if order_by and hasattr(self.model, order_by):
            order_column = getattr(self.model, order_by)
            statement = statement.order_by(
                asc(order_column) if order_direction.lower() == "asc" else desc(order_column)
            )

        statement = statement.offset(offset).limit(limit)
        return list(db.execute(statement).scalars().all())

    def count_all(self, db: Session) -> int:
        """Count all records via SQL COUNT (excluding soft deleted when supported)"""
        statement = self._exclude_soft_deleted(select(func.count(self.model.id)))
        return int(db.execute(statement).scalar() or 0)

    def exists(self, db: Session, id: int | UUID) -> bool:
        """Check if a record exists by ID (excluding soft deleted when supported)"""
        statement = self._exclude_soft_deleted(select(self.model.id).where(self.model.id == id))
        return db.execute(statement).scalar() is not None
