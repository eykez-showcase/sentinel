from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CameraBase(BaseModel):
    name: str
    stream_url: str
    config_id: str | None = None
    is_active: bool = True


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: str | None = None
    stream_url: str | None = None
    config_id: str | None = None
    is_active: bool | None = None
    frame_width: int | None = None
    frame_height: int | None = None


class CameraRead(CameraBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    frame_width: int | None
    frame_height: int | None
    created_at: datetime
    updated_at: datetime


class CameraList(BaseModel):
    items: list[CameraRead]
    total: int
