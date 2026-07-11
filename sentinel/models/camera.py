from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    stream_url: Mapped[str] = mapped_column(String(512), nullable=False)
    # Key from property.yaml (e.g. "front_door"). Optional — cameras can be
    # created via API without a corresponding YAML entry.
    config_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    frame_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    detections: Mapped[list["Detection"]] = relationship(  # noqa: F821
        back_populates="camera", lazy="noload"
    )
    tracks: Mapped[list["Track"]] = relationship(  # noqa: F821
        back_populates="camera", lazy="noload"
    )
    events: Mapped[list["Event"]] = relationship(  # noqa: F821
        back_populates="camera", lazy="noload"
    )
