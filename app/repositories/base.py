from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Any, Dict, Union
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete

from app.models.base import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType], ABC):
    """Base repository class with common CRUD operations"""

    def __init__(self, model: type[ModelType], db: Session):
        self.model = model
        self.db = db

    def create(self, input_schema: CreateSchemaType) -> ModelType:
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
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj

    def get_by_id(self, id: Union[int, UUID]) -> Optional[ModelType]:
        """Get a record by ID"""
        stmt = select(self.model).where(self.model.id == id)
        return self.db.execute(stmt).scalar_one_or_none()

    def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Get all records with pagination"""
        stmt = select(self.model).offset(skip).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def update(self, db_obj: ModelType, input_schema: UpdateSchemaType) -> ModelType:
        """Update an existing record"""
        obj_data = (
            input_schema.model_dump(exclude_unset=True)
            if hasattr(input_schema, "model_dump")
            else input_schema.__dict__
        )
        for field, value in obj_data.items():
            setattr(db_obj, field, value)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj

    def delete(self, id: Union[int, UUID]) -> bool:
        """Delete a record by ID"""
        db_obj = self.get_by_id(id)
        if db_obj:
            self.db.delete(db_obj)
            self.db.commit()
            return True
        return False

    def exists(self, id: Union[int, UUID]) -> bool:
        """Check if a record exists by ID"""
        stmt = select(self.model.id).where(self.model.id == id)
        return self.db.execute(stmt).scalar() is not None
