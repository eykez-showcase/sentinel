from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from sentinel.api.deps import get_db, get_event_or_404
from sentinel.models.ai_summary import AISummary
from sentinel.models.event import Event
from sentinel.schemas.ai_summary import AISummaryRead
from sentinel.schemas.event import EventRead

router = APIRouter()


@router.get("/", response_model=list[EventRead])
async def list_events(
    camera_id: str | None = Query(None),
    event_type: str | None = Query(None),
    zone_id: str | None = Query(None),
    severity: str | None = Query(None),
    from_ts: datetime | None = Query(None),
    to_ts: datetime | None = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
) -> list[EventRead]:
    stmt = (
        select(Event)
        .options(selectinload(Event.ai_summary))
        .order_by(Event.timestamp.desc())
    )

    if camera_id:
        stmt = stmt.where(Event.camera_id == camera_id)
    if event_type:
        stmt = stmt.where(Event.event_type == event_type)
    if zone_id:
        stmt = stmt.where(Event.zone_id == zone_id)
    if severity:
        stmt = stmt.where(Event.severity == severity)
    if from_ts:
        stmt = stmt.where(Event.timestamp >= from_ts)
    if to_ts:
        stmt = stmt.where(Event.timestamp <= to_ts)

    stmt = stmt.limit(limit).offset(offset)
    result = await db.execute(stmt)
    events = result.scalars().all()
    return [EventRead.model_validate(e) for e in events]


@router.get("/stats", response_model=dict)
async def event_stats(
    from_ts: datetime | None = Query(None),
    to_ts: datetime | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Aggregate event counts grouped by type and zone."""
    stmt = select(
        Event.event_type,
        Event.zone_id,
        func.count(Event.id).label("count"),
    )
    if from_ts:
        stmt = stmt.where(Event.timestamp >= from_ts)
    if to_ts:
        stmt = stmt.where(Event.timestamp <= to_ts)
    stmt = stmt.group_by(Event.event_type, Event.zone_id)

    result = await db.execute(stmt)
    rows = result.all()
    return {
        "stats": [
            {"event_type": r.event_type, "zone_id": r.zone_id, "count": r.count}
            for r in rows
        ]
    }


@router.get("/{event_id}", response_model=EventRead)
async def get_event(
    event_id: str,
    db: AsyncSession = Depends(get_db),
) -> EventRead:
    stmt = (
        select(Event)
        .where(Event.id == event_id)
        .options(selectinload(Event.ai_summary))
    )
    result = await db.execute(stmt)
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return EventRead.model_validate(event)


@router.post("/{event_id}/summarize", response_model=AISummaryRead)
async def summarize_event(
    event_id: str,
    db: AsyncSession = Depends(get_db),
    event: Event = Depends(get_event_or_404),
) -> AISummaryRead:
    """
    Trigger Claude AI summarization for an event and its surrounding context.
    Returns an existing summary if one already exists.
    """
    from sentinel.ai.summarizer import make_summarizer
    from sentinel.core.events import bus
    from sentinel.zones.loader import get_property_config

    # Return existing summary if present.
    existing = await db.execute(
        select(AISummary).where(AISummary.event_id == event_id)
    )
    existing_summary = existing.scalar_one_or_none()
    if existing_summary:
        return AISummaryRead.model_validate(existing_summary)

    # Fetch surrounding events (same camera, ±10 minutes).
    window_start = event.timestamp.replace(
        minute=max(0, event.timestamp.minute - 10)
    )
    stmt = (
        select(Event)
        .where(
            Event.camera_id == event.camera_id,
            Event.timestamp >= window_start,
            Event.timestamp <= event.timestamp,
        )
        .order_by(Event.timestamp.asc())
        .limit(20)
    )
    result = await db.execute(stmt)
    context_events = result.scalars().all()
    event_reads = [EventRead.model_validate(e) for e in context_events]

    try:
        prop_config = get_property_config()
        property_name = prop_config.name
    except RuntimeError:
        property_name = "Property"

    summarizer = make_summarizer()
    summary_text, raw_req, raw_resp = await summarizer.summarize(
        event_reads, property_name
    )

    tokens = None
    if "usage" in raw_resp:
        tokens = raw_resp["usage"].get("input_tokens", 0) + raw_resp["usage"].get("output_tokens", 0)

    summary = AISummary(
        event_id=event_id,
        summary_text=summary_text,
        raw_request=raw_req,
        raw_response=raw_resp,
        model_used=raw_resp.get("model", "unknown"),
        tokens_used=tokens,
    )
    db.add(summary)
    await db.commit()
    await db.refresh(summary)

    await bus.publish(
        "ai_summary.created",
        AISummaryRead.model_validate(summary).model_dump(mode="json"),
    )

    return AISummaryRead.model_validate(summary)
