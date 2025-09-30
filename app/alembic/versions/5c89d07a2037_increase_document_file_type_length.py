"""increase_document_file_type_length

Revision ID: 5c89d07a2037
Revises: 1d24e8e1ec28
Create Date: 2025-09-30 09:22:38.664601

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5c89d07a2037"
down_revision: Union[str, None] = "1d24e8e1ec28"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Increase the file_type column length from 50 to 100 characters
    op.alter_column(
        "documents",
        "file_type",
        existing_type=sa.String(length=50),
        type_=sa.String(length=100),
        existing_nullable=False,
    )


def downgrade() -> None:
    # Revert the file_type column length back to 50 characters
    op.alter_column(
        "documents",
        "file_type",
        existing_type=sa.String(length=100),
        type_=sa.String(length=50),
        existing_nullable=False,
    )
