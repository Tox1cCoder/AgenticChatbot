"""
Strategy pattern interfaces for repository operations.
DEPRECATED: Use command_strategy.py and query_strategy.py instead
"""

# Re-export for backwards compatibility
from .command_strategy import CommandStrategy, DefaultCommandStrategy
from .query_strategy import QueryStrategy, DefaultQueryStrategy
from typing import Generic, TypeVar, Optional, List, Union
from uuid import UUID
from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


# Legacy CRUD interface for backwards compatibility
class CRUDStrategy(
    CommandStrategy[ModelType, CreateSchemaType, UpdateSchemaType],
    QueryStrategy[ModelType],
):
    """Legacy CRUD strategy - use CommandStrategy and QueryStrategy instead"""

    pass


class DefaultCRUDStrategy(
    DefaultCommandStrategy[ModelType, CreateSchemaType, UpdateSchemaType],
    DefaultQueryStrategy[ModelType],
):
    """Legacy default CRUD strategy - use DefaultCommandStrategy and DefaultQueryStrategy instead"""

    def __init__(self, model: type[ModelType]):
        DefaultCommandStrategy.__init__(self, model)
        DefaultQueryStrategy.__init__(self, model)


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
