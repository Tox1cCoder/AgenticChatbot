"""Lexical search and bounded reads over a project's past conversations.

Scope mirrors :mod:`app.repositories.user_memory` exactly, so there is one
rule to remember rather than two: inside a project you reach that project's
conversations, outside one you reach only conversations that belong to no
project. Ownership is enforced in every statement - a conversation id the
caller does not own is simply not found.

Search is lexical (PostgreSQL full text), matching the approach already used
for document chunks. ``'simple'`` does no stemming, so Vietnamese and English
behave the same way.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, String, func, literal_column, select
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.message import Message
from app.repositories.session_transport import RepositorySessionMixin

#: Terms are OR-ed rather than AND-ed. ``websearch_to_tsquery`` defaults to
#: requiring every term, which makes a natural-language question ("what is my
#: favourite colour") match nothing. OR plus ``ts_rank_cd`` ordering keeps the
#: best conversation first while still finding partial matches.
_TSQUERY_JOINER = " OR "

_LANGUAGE = literal_column("'simple'")


@dataclass(frozen=True)
class ConversationHit:
    """One matching past conversation."""

    conversation_id: UUID
    title: str | None
    created_at: datetime
    snippet: str
    rank: float


@dataclass(frozen=True)
class TranscriptLine:
    """One message from a past conversation."""

    sender: int
    created_at: datetime
    content: str


def build_tsquery(query: str):
    """An OR-ed tsquery over the caller's terms, or None when empty.

    ``websearch_to_tsquery`` is used rather than ``to_tsquery`` because it
    never raises on arbitrary user text - the model's query reaches it
    unsanitized.
    """
    terms = [term for term in str(query or "").split() if term]
    if not terms:
        return None
    return func.websearch_to_tsquery(_LANGUAGE, _TSQUERY_JOINER.join(terms))


class ConversationSearchRepository(RepositorySessionMixin):
    """Reads across a user's own conversations within one project scope."""

    def __init__(
        self,
        session_factory: Callable[[], Any],
        async_session_factory: Callable[[], Any] | None = None,
    ):
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )

    # ------------------------------------------------------------------
    # Scoping
    # ------------------------------------------------------------------

    @staticmethod
    def _scoped_conversations(user_id: str, project_id: str | None) -> Select:
        """Live conversations the caller may reach from this project scope."""
        scope = (
            Conversation.project_id == project_id
            if project_id is not None
            else Conversation.project_id.is_(None)
        )
        return select(Conversation.id).where(
            Conversation.owner_id == user_id,
            Conversation.deleted_at.is_(None),
            scope,
        )

    # ------------------------------------------------------------------
    # Query bodies - written once, run over either transport
    # ------------------------------------------------------------------

    @classmethod
    def _search_work(
        cls,
        user_id: str,
        project_id: str | None,
        query: str,
        limit: int,
        exclude_conversation_id: str | None,
        snippet_chars: int,
    ) -> Callable[[Session], list[ConversationHit]]:
        tsquery = build_tsquery(query)

        def work(session: Session) -> list[ConversationHit]:
            if tsquery is None:
                return []

            reachable = cls._scoped_conversations(user_id, project_id)
            if exclude_conversation_id:
                reachable = reachable.where(Conversation.id != exclude_conversation_id)

            vector = func.to_tsvector(_LANGUAGE, Message.content)
            rank = func.ts_rank_cd(vector, tsquery)

            # Rank every matching message, then keep each conversation's best
            # one: a conversation is the unit the caller asked about, not a
            # message.
            ranked = (
                select(
                    Message.conversation_id.label("conversation_id"),
                    Message.content.label("content"),
                    rank.label("rank"),
                    func.row_number()
                    .over(
                        partition_by=Message.conversation_id,
                        order_by=(rank.desc(), Message.sequence.asc()),
                    )
                    .label("row_number"),
                )
                .where(
                    Message.deleted_at.is_(None),
                    Message.conversation_id.in_(reachable),
                    vector.op("@@")(tsquery),
                )
                .subquery()
            )

            statement = (
                select(
                    ranked.c.conversation_id,
                    Conversation.title,
                    Conversation.created_at,
                    func.left(ranked.c.content, snippet_chars).label("snippet"),
                    ranked.c.rank,
                )
                .join(Conversation, Conversation.id == ranked.c.conversation_id)
                .where(ranked.c.row_number == 1)
                .order_by(ranked.c.rank.desc(), Conversation.created_at.desc())
                .limit(limit)
            )

            return [
                ConversationHit(
                    conversation_id=row.conversation_id,
                    title=row.title,
                    created_at=row.created_at,
                    snippet=(row.snippet or "").strip(),
                    rank=float(row.rank or 0.0),
                )
                for row in session.execute(statement)
            ]

        return work

    @classmethod
    def _read_work(
        cls,
        user_id: str,
        project_id: str | None,
        conversation_id: str,
        max_messages: int,
        max_chars_per_message: int,
    ) -> Callable[[Session], tuple[str | None, list[TranscriptLine], int] | None]:
        def work(session: Session) -> tuple[str | None, list[TranscriptLine], int] | None:
            reachable = cls._scoped_conversations(user_id, project_id)
            # A prefix is what the model is shown, so a prefix is what it can
            # pass back. Ownership still comes from ``reachable``.
            conversation = session.execute(
                select(Conversation.id, Conversation.title)
                .where(
                    Conversation.id.in_(reachable),
                    Conversation.id.cast(String).like(f"{conversation_id}%"),
                )
                .limit(2)
            ).all()
            if len(conversation) != 1:
                return None

            resolved_id, title = conversation[0]
            total = session.execute(
                select(func.count(Message.id)).where(
                    Message.conversation_id == resolved_id,
                    Message.deleted_at.is_(None),
                )
            ).scalar_one()

            # Newest first for the cap, then reversed: truncating a long
            # conversation should drop the beginning, not the conclusion.
            rows = session.execute(
                select(Message.sender, Message.created_at, Message.content)
                .where(
                    Message.conversation_id == resolved_id,
                    Message.deleted_at.is_(None),
                )
                .order_by(Message.sequence.desc())
                .limit(max_messages)
            ).all()

            lines = [
                TranscriptLine(
                    sender=int(row.sender),
                    created_at=row.created_at,
                    content=_truncate(row.content, max_chars_per_message),
                )
                for row in reversed(rows)
            ]
            return title, lines, int(total)

        return work

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def search(
        self,
        user_id: str,
        query: str,
        project_id: str | None = None,
        limit: int = 5,
        exclude_conversation_id: str | None = None,
        snippet_chars: int = 240,
    ) -> list[ConversationHit]:
        """Past conversations matching ``query``, best first."""
        return self._run(
            self._search_work(
                user_id, project_id, query, limit, exclude_conversation_id, snippet_chars
            )
        )

    async def asearch(
        self,
        user_id: str,
        query: str,
        project_id: str | None = None,
        limit: int = 5,
        exclude_conversation_id: str | None = None,
        snippet_chars: int = 240,
    ) -> list[ConversationHit]:
        """Async twin of :meth:`search`."""
        return await self._arun(
            self._search_work(
                user_id, project_id, query, limit, exclude_conversation_id, snippet_chars
            )
        )

    def read(
        self,
        user_id: str,
        conversation_id: str,
        project_id: str | None = None,
        max_messages: int = 30,
        max_chars_per_message: int = 600,
    ) -> tuple[str | None, list[TranscriptLine], int] | None:
        """A bounded transcript, or None when not reachable in this scope."""
        return self._run(
            self._read_work(
                user_id, project_id, conversation_id, max_messages, max_chars_per_message
            )
        )

    async def aread(
        self,
        user_id: str,
        conversation_id: str,
        project_id: str | None = None,
        max_messages: int = 30,
        max_chars_per_message: int = 600,
    ) -> tuple[str | None, list[TranscriptLine], int] | None:
        """Async twin of :meth:`read`."""
        return await self._arun(
            self._read_work(
                user_id, project_id, conversation_id, max_messages, max_chars_per_message
            )
        )

    def resolve_project_id(self, user_id: str, conversation_id: str | None) -> str | None:
        """The project a conversation belongs to, ownership-checked."""
        if not conversation_id:
            return None

        def work(session: Session) -> str | None:
            project_id = session.execute(
                select(Conversation.project_id).where(
                    Conversation.id == conversation_id,
                    Conversation.owner_id == user_id,
                )
            ).scalars().first()
            return str(project_id) if project_id else None

        return self._run(work)


def _truncate(content: str, limit: int) -> str:
    text_value = str(content or "").strip()
    if limit <= 0 or len(text_value) <= limit:
        return text_value
    return text_value[:limit].rstrip() + " […]"


__all__ = [
    "ConversationHit",
    "ConversationSearchRepository",
    "TranscriptLine",
    "build_tsquery",
]
