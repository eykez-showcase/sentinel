from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from sentinel.schemas.ai_summary import AISummaryRead

EventType = Literal[
    "zone_entry",
    "zone_exit",
    "loitering",
    "perimeter_breach",
    "unknown_vehicle",
    "package_detected",
    "camera_offline",
    "camera_online",
    "face_recognized",
    "unknown_face",
]

Severity = Literal["low", "medium", "high"]


class EventBase(BaseModel):
    camera_id: str
    track_id: str | None = None
    event_type: EventType
    zone_id: str | None = None
    class_name: str | None = None
    severity: Severity = "low"
    timestamp: datetime
    metadata_json: dict[str, Any] | None = None


class EventCreate(EventBase):
    pass


class EventRead(EventBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    ai_summary: AISummaryRead | None = None


class EventFilter(BaseModel):
    camera_id: str | None = None
    event_type: EventType | None = None
    zone_id: str | None = None
    severity: Severity | None = None
    from_ts: datetime | None = None
    to_ts: datetime | None = None
    limit: int = 50
    offset: int = 0
