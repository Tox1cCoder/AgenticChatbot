# Importing ``app.models.base`` runs ``app/models/__init__.py`` first, and that
# package import is what registers every model table on ``Base.metadata``.
# Alembic's env.py relies on it: a model missing from ``app/models/__init__.py``
# is invisible to autogenerate.
from app.models.base import Base

__all__ = ["Base"]
