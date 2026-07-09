"""
Command strategy pattern interfaces for repository write operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class CommandStrategy(ABC, Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Abstract strategy for command (write) operations"""

    @abstractmethod
    def create(self, db: Session, input_schema: CreateSchemaType) -> ModelType:
        """Create a new record"""
        pass

    @abstractmethod
    def update(self, db: Session, db_obj: ModelType, input_schema: UpdateSchemaType) -> ModelType:
        """Update an existing record"""
        pass

    @abstractmethod
    def delete(self, db: Session, id: int | UUID) -> bool:
        """Delete a record by ID"""
        pass


class DefaultCommandStrategy(CommandStrategy[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Default implementation of command operations strategy"""

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
        db.expunge(db_obj)
        return db_obj

    def update(self, db: Session, db_obj: ModelType, input_schema: UpdateSchemaType) -> ModelType:
        """Update an existing record"""
        obj_data = (
            input_schema.model_dump(exclude_unset=True)
            if hasattr(input_schema, "model_dump")
            else input_schema.__dict__
        )
        for field, value in obj_data.items():
            if field in ["id", "created_at"]:
                continue
            if hasattr(db_obj, field):
                current_value = getattr(db_obj, field)
                if isinstance(current_value, UUID) and field == "id":
                    continue

                setattr(db_obj, field, value)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, id: int | UUID) -> bool:
        """Delete a record by ID.

        Soft-deletes (sets ``deleted_at``) for models that support it; hard-
        deletes models without a ``deleted_at`` column so the generic strategy
        is safe for both.
        """
        soft_delete = hasattr(self.model, "deleted_at")
        statement = select(self.model).where(self.model.id == id)
        if soft_delete:
            statement = statement.where(self.model.deleted_at.is_(None))
        db_obj = db.execute(statement).scalar_one_or_none()

        if not db_obj:
            return False

        if soft_delete:
            db_obj.deleted_at = datetime.now(timezone.utc)
        else:
            db.delete(db_obj)
        db.commit()
        return True
