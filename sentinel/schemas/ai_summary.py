from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class AISummaryBase(BaseModel):
    event_id: str
    summary_text: str
    model_used: str
    tokens_used: int | None = None


class AISummaryCreate(AISummaryBase):
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None


class AISummaryRead(AISummaryBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
