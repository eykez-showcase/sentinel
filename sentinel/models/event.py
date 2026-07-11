from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel.core.database import Base

EVENT_TYPES = (
    "zone_entry",
    "zone_exit",
    "loitering",
    "perimeter_breach",
    "unknown_vehicle",
    "package_detected",
    "camera_offline",
    "camera_online",
)

SEVERITIES = ("low", "medium", "high")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    camera_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )
    track_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tracks.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(
        Enum(*EVENT_TYPES, name="event_type"), nullable=False
    )
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    class_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    severity: Mapped[str] = mapped_column(
        Enum(*SEVERITIES, name="severity"), nullable=False, default="low"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Arbitrary JSON for extra context (e.g. centroid coords, speed estimate).
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    camera: Mapped["Camera"] = relationship(  # noqa: F821
        back_populates="events", lazy="noload"
    )
    track: Mapped["Track | None"] = relationship(  # noqa: F821
        back_populates="events", lazy="noload"
    )
    ai_summary: Mapped["AISummary | None"] = relationship(  # noqa: F821
        back_populates="event", lazy="noload", uselist=False
    )
