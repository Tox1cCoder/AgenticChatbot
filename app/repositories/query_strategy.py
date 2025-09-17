"""
Query strategy pattern interfaces for repository read operations.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Union
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

ModelType = TypeVar("ModelType")


class QueryStrategy(ABC, Generic[ModelType]):
    """Abstract strategy for query (read) operations"""

    @abstractmethod
    def get_by_id(self, db: Session, id: Union[int, UUID]) -> Optional[ModelType]:
        """Get a record by ID"""
        pass

    @abstractmethod
    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Get all records with pagination"""
        pass

    @abstractmethod
    def exists(self, db: Session, id: Union[int, UUID]) -> bool:
        """Check if a record exists by ID"""
        pass


class DefaultQueryStrategy(QueryStrategy[ModelType]):
    """Default implementation of query operations strategy"""

    def __init__(self, model: type[ModelType]):
        self.model = model

    def get_by_id(self, db: Session, id: Union[int, UUID]) -> Optional[ModelType]:
        """Get a record by ID (excluding soft deleted)"""
        stmt = select(self.model).where(
            self.model.id == id, self.model.deleted_at.is_(None)
        )
        return db.execute(stmt).scalar_one_or_none()

    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Get all records with pagination (excluding soft deleted)"""
        stmt = (
            select(self.model)
            .where(self.model.deleted_at.is_(None))
            .offset(skip)
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())

    def exists(self, db: Session, id: Union[int, UUID]) -> bool:
        """Check if a record exists by ID (excluding soft deleted)"""
        stmt = select(self.model.id).where(
            self.model.id == id, self.model.deleted_at.is_(None)
        )
        return db.execute(stmt).scalar() is not None
