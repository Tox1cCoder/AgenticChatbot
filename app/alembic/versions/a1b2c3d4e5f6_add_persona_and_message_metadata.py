"""add_persona_and_message_metadata

Revision ID: a1b2c3d4e5f6
Revises: f5c3900b4134
Create Date: 2025-10-17 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f5c3900b4134"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add persona_prompt column to conversations table
    op.add_column(
        "conversations",
        sa.Column("persona_prompt", sa.Text(), nullable=True),
    )

    # Add message_metadata column to messages table
    op.add_column(
        "messages",
        sa.Column(
            "message_metadata",
            JSONB,
            nullable=True,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    # Drop message_metadata column from messages table
    op.drop_column("messages", "message_metadata")

    # Drop persona_prompt column from conversations table
    op.drop_column("conversations", "persona_prompt")
