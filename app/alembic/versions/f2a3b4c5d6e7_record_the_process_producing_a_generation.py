"""record which process is producing a generation

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-09-10 00:00:00.000000

``uq_generations_active_per_conversation`` admits one active row per
conversation. That is what stops a conversation running two turns at once, and
its cost is that a row left active by a worker that died blocks that
conversation permanently:

    duplicate key value violates unique constraint
    "uq_generations_active_per_conversation"

The next question cannot even be inserted, the API reports
``conversation_turn_conflict``, and no retry clears it because no worker exists
to finish the turn. In development this is routine rather than rare: uvicorn's
reloader replaces the serving child on every code change, so any turn
streaming at that moment is abandoned.

Reclaiming such a row requires knowing whose it was, and nothing already
recorded answers that. ``build_sha`` is identical across every worker of one
build, and the worker count deliberately lives in the launch command rather
than in a setting, so "there is only one worker, therefore everything active
is abandoned" is not a fact the application may assume -- under
``--workers N`` acting on it would terminalize a peer's streaming turn.

``producer_token`` is ``hostname:pid:started_at`` for the worker that inserted
the row. The start time is what makes a recycled pid distinguishable from the
process that originally held it; without it a new process inheriting the pid
would read as the dead producer still running, and the conversation would stay
blocked for good.

Nullable, with no backfill. A value invented for an existing row would be a
claim about a process this migration cannot inspect, and the reaper treats an
absent producer as unknown -- left alone -- rather than as dead. Any row
already stranded by this bug therefore stays stranded, and has to be
terminalized by hand; the column stops it happening again.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f2a3b4c5d6e7"
down_revision: str | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "generations",
        sa.Column("producer_token", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("generations", "producer_token")
