# Import all models to register them with SQLAlchemy metadata

# Import shared Base for Alembic
from app.models.base import Base

# Make Base available for Alembic
__all__ = ["Base"]
