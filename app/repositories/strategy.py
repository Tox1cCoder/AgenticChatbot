"""
Strategy pattern interfaces for repository operations.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Any, Dict, Union
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete

from app.models.base import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class CRUDStrategy(ABC, Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Abstract strategy for CRUD operations"""

    @abstractmethod
    def create(self, db: Session, input_schema: CreateSchemaType) -> ModelType:
        """Create a new record"""
        pass

    @abstractmethod
    def get_by_id(self, db: Session, id: Union[int, UUID]) -> Optional[ModelType]:
        """Get a record by ID"""
        pass

    @abstractmethod
    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Get all records with pagination"""
        pass

    @abstractmethod
    def update(
        self, db: Session, db_obj: ModelType, input_schema: UpdateSchemaType
    ) -> ModelType:
        """Update an existing record"""
        pass

    @abstractmethod
    def delete(self, db: Session, id: Union[int, UUID]) -> bool:
        """Delete a record by ID"""
        pass

    @abstractmethod
    def exists(self, db: Session, id: Union[int, UUID]) -> bool:
        """Check if a record exists by ID"""
        pass


class DefaultCRUDStrategy(CRUDStrategy[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Default implementation of CRUD operations strategy"""

    def __init__(self, model: type[ModelType]):
        self.model = model

    def create(self, db: Session, input_schema: CreateSchemaType) -> ModelType:
        """Create a new record"""
        if isinstance(input_schema, dict):
            input_schema_data = input_schema
        else:
            input_schema_data = (
                input_schema.model_dump()
                if hasattr(input_schema, "model_dump")
                else input_schema.__dict__
            )
        db_obj = self.model(**input_schema_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_by_id(self, db: Session, id: Union[int, UUID]) -> Optional[ModelType]:
        """Get a record by ID"""
        stmt = select(self.model).where(self.model.id == id)
        return db.execute(stmt).scalar_one_or_none()

    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Get all records with pagination"""
        stmt = select(self.model).offset(skip).limit(limit)
        return list(db.execute(stmt).scalars().all())

    def update(
        self, db: Session, db_obj: ModelType, input_schema: UpdateSchemaType
    ) -> ModelType:
        """Update an existing record"""
        obj_data = (
            input_schema.model_dump(exclude_unset=True)
            if hasattr(input_schema, "model_dump")
            else input_schema.__dict__
        )
        for field, value in obj_data.items():
            setattr(db_obj, field, value)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, id: Union[int, UUID]) -> bool:
        """Delete a record by ID"""
        db_obj = self.get_by_id(db, id)
        if db_obj:
            db.delete(db_obj)
            db.commit()
            return True
        return False

    def exists(self, db: Session, id: Union[int, UUID]) -> bool:
        """Check if a record exists by ID"""
        stmt = select(self.model.id).where(self.model.id == id)
        return db.execute(stmt).scalar() is not None


class Repository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Repository using strategy pattern.
    """

    def __init__(
        self,
        db: Session,
        crud_strategy: CRUDStrategy[ModelType, CreateSchemaType, UpdateSchemaType],
    ):
        self.db = db
        self._crud_strategy = crud_strategy

    def create(self, input_schema: CreateSchemaType) -> ModelType:
        """Create a new record"""
        return self._crud_strategy.create(self.db, input_schema)

    def get_by_id(self, id: Union[int, UUID]) -> Optional[ModelType]:
        """Get a record by ID"""
        return self._crud_strategy.get_by_id(self.db, id)

    def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Get all records with pagination"""
        return self._crud_strategy.get_all(self.db, skip, limit)

    def update(self, db_obj: ModelType, input_schema: UpdateSchemaType) -> ModelType:
        """Update an existing record"""
        return self._crud_strategy.update(self.db, db_obj, input_schema)

    def delete(self, id: Union[int, UUID]) -> bool:
        """Delete a record by ID"""
        return self._crud_strategy.delete(self.db, id)

    def exists(self, id: Union[int, UUID]) -> bool:
        """Check if a record exists by ID"""
        return self._crud_strategy.exists(self.db, id)

    def set_strategy(
        self, strategy: CRUDStrategy[ModelType, CreateSchemaType, UpdateSchemaType]
    ):
        """Change the CRUD strategy at runtime"""
        self._crud_strategy = strategy
