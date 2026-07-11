from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel.core.database import Base


class Track(Base):
    """
    A persistent identity for an object tracked across multiple frames.

    tracker_id is ByteTrack's integer (can be reused after a track goes
    inactive). The UUID id is Sentinel's stable primary key.
    """

    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    camera_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )
    tracker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    camera: Mapped["Camera"] = relationship(  # noqa: F821
        back_populates="tracks", lazy="noload"
    )
    events: Mapped[list["Event"]] = relationship(  # noqa: F821
        back_populates="track", lazy="noload"
    )
