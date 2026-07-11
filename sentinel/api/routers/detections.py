from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.api.deps import get_db
from sentinel.models.detection import Detection
from sentinel.schemas.detection import DetectionBatch, DetectionRead

router = APIRouter()


@router.get("/", response_model=list[DetectionRead])
async def list_detections(
    camera_id: str | None = Query(None),
    class_name: str | None = Query(None),
    zone_id: str | None = Query(None),
    from_ts: datetime | None = Query(None),
    to_ts: datetime | None = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
) -> list[DetectionRead]:
    stmt = select(Detection).order_by(Detection.timestamp.desc())

    if camera_id:
        stmt = stmt.where(Detection.camera_id == camera_id)
    if class_name:
        stmt = stmt.where(Detection.class_name == class_name)
    if zone_id:
        stmt = stmt.where(Detection.zone_id == zone_id)
    if from_ts:
        stmt = stmt.where(Detection.timestamp >= from_ts)
    if to_ts:
        stmt = stmt.where(Detection.timestamp <= to_ts)

    stmt = stmt.limit(limit).offset(offset)
    result = await db.execute(stmt)
    detections = result.scalars().all()
    return [DetectionRead.model_validate(d) for d in detections]


@router.get("/{detection_id}", response_model=DetectionRead)
async def get_detection(
    detection_id: str,
    db: AsyncSession = Depends(get_db),
) -> DetectionRead:
    from fastapi import HTTPException, status

    result = await db.execute(select(Detection).where(Detection.id == detection_id))
    det = result.scalar_one_or_none()
    if det is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection not found")
    return DetectionRead.model_validate(det)


@router.post("/batch", response_model=dict)
async def ingest_detections(
    body: DetectionBatch,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Bulk-insert detections from an external CV process.
    In single-process mode the CV pipeline writes directly to the DB;
    this endpoint supports multi-process deployments.
    """
    records = [Detection(**item.model_dump()) for item in body.items]
    db.add_all(records)
    await db.commit()
    return {"inserted": len(records)}
