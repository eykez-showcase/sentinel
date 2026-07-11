from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel.core.database import Base


class AISummary(Base):
    __tablename__ = "ai_summaries"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # One summary per event (unique constraint enforced at the DB level).
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("events.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Store raw request/response for auditability and prompt tuning.
    raw_request: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_used: Mapped[str] = mapped_column(String(64), nullable=False)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    event: Mapped["Event"] = relationship(  # noqa: F821
        back_populates="ai_summary", lazy="noload"
    )
