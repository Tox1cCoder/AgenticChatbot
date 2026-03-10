"""merge_hitl_with_main

Revision ID: 1ce64a959f7d
Revises: be1969e4b7f2, h1i2j3k4l5m6
Create Date: 2026-03-09 14:45:41.251799

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1ce64a959f7d'
down_revision: Union[str, None] = ('be1969e4b7f2', 'h1i2j3k4l5m6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
