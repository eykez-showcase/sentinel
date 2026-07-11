from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TrackBase(BaseModel):
    camera_id: str
    tracker_id: int
    class_name: str
    zone_id: str | None = None


class TrackCreate(TrackBase):
    first_seen: datetime
    last_seen: datetime


class TrackUpdate(BaseModel):
    last_seen: datetime | None = None
    zone_id: str | None = None
    is_active: bool | None = None


class TrackRead(TrackBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    first_seen: datetime
    last_seen: datetime
    is_active: bool
    created_at: datetime
    updated_at: datetime
