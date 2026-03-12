"""rename_feedback_table_to_feedbacks

Revision ID: ef1b2c3d4e5f
Revises: 4dc4ad0b1936
Create Date: 2025-09-15 10:35:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ef1b2c3d4e5f"
down_revision: Union[str, None] = "4dc4ad0b1936"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename feedback table to feedbacks for consistency"""
    op.rename_table("feedback", "feedbacks")


def downgrade() -> None:
    """Rename feedbacks table back to feedback"""
    op.rename_table("feedbacks", "feedback")
