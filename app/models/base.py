"""
Shared declarative base for all models.
Enables relationship resolution without inheritance.
"""

from sqlalchemy.orm import declarative_base

Base = declarative_base()
