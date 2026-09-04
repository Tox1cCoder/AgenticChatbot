"""Add cached bytes for verified remote rich images.

Visual verification already downloads and decodes every image it approves.
Storing those bytes on the reference row makes an approved image one fetch
instead of two, and closes the window in which an image passes every check, is
placed in the answer, and then fails at render time.

All three columns are nullable: a reference registered without bytes still
renders by fetching upstream, exactly as before.

Revision ID: b2c3d4e5f6a7
Revises: f9a0b1c2d3e4
Create Date: 2026-08-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "f9a0b1c2d3e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("web_image_references", sa.Column("content", sa.LargeBinary(), nullable=True))
    op.add_column("web_image_references", sa.Column("cached_width", sa.Integer(), nullable=True))
    op.add_column("web_image_references", sa.Column("cached_height", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("web_image_references", "cached_height")
    op.drop_column("web_image_references", "cached_width")
    op.drop_column("web_image_references", "content")
