"""Server-wide keys the application generated for itself.

One row per key name. ``app.core.server_secrets`` is the only reader and writer:
it inserts a row only when no key is configured in the environment, and it never
updates one. Anyone holding a database backup holds these keys.
"""

from sqlalchemy import Column, DateTime, String, Text, func

from app.models.base import Base


class ServerSecret(Base):
    __tablename__ = "server_secrets"

    name = Column(String(64), primary_key=True)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    def __repr__(self) -> str:
        # Never the value: the repr reaches logs and tracebacks.
        return f"<ServerSecret(name={self.name!r})>"
