from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PersonCreate(BaseModel):
    name: str
    role: str = "family"
    notes: str | None = None


class PersonUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    notes: str | None = None


class PersonRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    role: str
    notes: str | None
    created_at: datetime
    photo_count: int = 0


class FaceEmbeddingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    person_id: str
    photo_path: str | None
    created_at: datetime


class RecognitionResult(BaseModel):
    person_id: str | None
    person_name: str | None
    role: str | None
    confidence: float
