"""
Query strategy pattern interfaces for repository read operations.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Union
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select, asc, desc

ModelType = TypeVar("ModelType")


class QueryStrategy(ABC, Generic[ModelType]):
    """Abstract strategy for query (read) operations"""

    @abstractmethod
    def get_by_id(self, db: Session, id: Union[int, UUID]) -> Optional[ModelType]:
        """Get a record by ID"""
        pass

    @abstractmethod
    def get_all(self, db: Session, page: int = 1, limit: int = 10) -> List[ModelType]:
        """Get all records with page-based pagination"""
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

    def get_all(self, db: Session, page: int = 1, limit: int = 10) -> List[ModelType]:
        """Get all records with page-based pagination (excluding soft deleted)"""
        offset = (page - 1) * limit
        stmt = (
            select(self.model)
            .where(self.model.deleted_at.is_(None))
            .offset(offset)
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())

    def get_all_with_ordering(
        self,
        db: Session,
        page: int = 1,
        limit: int = 10,
        order_by: Optional[str] = None,
        order_direction: str = "desc",
    ) -> List[ModelType]:
        """Get all records with page-based pagination and dynamic ordering (excluding soft deleted)"""
        offset = (page - 1) * limit
        stmt = select(self.model).where(self.model.deleted_at.is_(None))

        # Apply ordering if specified
        if order_by:
            order_column = getattr(self.model, order_by, None)
            if order_column is not None:
                if order_direction.lower() == "asc":
                    stmt = stmt.order_by(asc(order_column))
                else:
                    stmt = stmt.order_by(desc(order_column))

        stmt = stmt.offset(offset).limit(limit)
        return list(db.execute(stmt).scalars().all())

    def count_all(self, db: Session) -> int:
        """Count all records (excluding soft deleted)"""
        stmt = select(self.model).where(self.model.deleted_at.is_(None))
        return len(list(db.execute(stmt).scalars().all()))

    def exists(self, db: Session, id: Union[int, UUID]) -> bool:
        """Check if a record exists by ID (excluding soft deleted)"""
        stmt = select(self.model.id).where(
            self.model.id == id, self.model.deleted_at.is_(None)
        )
        return db.execute(stmt).scalar() is not None
