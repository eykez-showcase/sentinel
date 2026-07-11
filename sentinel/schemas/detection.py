from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DetectionBase(BaseModel):
    camera_id: str
    frame_number: int
    timestamp: datetime
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    tracker_id: int | None = None
    zone_id: str | None = None


class DetectionCreate(DetectionBase):
    pass


class DetectionRead(DetectionBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


class DetectionBatch(BaseModel):
    items: list[DetectionCreate]
