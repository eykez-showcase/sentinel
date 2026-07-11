from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.core.database import get_db
from sentinel.models.camera import Camera
from sentinel.models.event import Event


async def get_camera_or_404(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
) -> Camera:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return camera


async def get_event_or_404(
    event_id: str,
    db: AsyncSession = Depends(get_db),
) -> Event:
    result = await db.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return event
