"""add_message_lexical_search_index

Revision ID: de19068933b7
Revises: 9778bb07ea35
Create Date: 2026-09-22 00:00:00.000000

Adds the GIN index that lets an agent search a project's past conversations.

``'simple'`` rather than ``'english'``, matching
``idx_document_chunks_content_simple_fts``: the simple configuration
lowercases and tokenizes without stemming, so it behaves the same for
Vietnamese as for English. An English stemmer would mangle both.

The index must match the search expression exactly or PostgreSQL will not
use it - see ``postgres_message_lexical_expressions``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "de19068933b7"
down_revision: str | Sequence[str] | None = "9778bb07ea35"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Index message text for lexical search."""

    op.create_index(
        "idx_messages_content_simple_fts",
        "messages",
        [sa.text("to_tsvector('simple', content)")],
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Drop the search index. Search degrades to a sequential scan."""

    op.drop_index("idx_messages_content_simple_fts", table_name="messages")
