"""
Shared declarative base for all models.
Enables relationship resolution without inheritance.
"""

from sqlalchemy.orm import declarative_base

# Single shared declarative base for all models
Base = declarative_base()
