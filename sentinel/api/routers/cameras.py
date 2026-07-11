from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.api.deps import get_camera_or_404, get_db
from sentinel.models.camera import Camera
from sentinel.schemas.camera import CameraCreate, CameraList, CameraRead, CameraUpdate

router = APIRouter()


@router.get("/", response_model=CameraList)
async def list_cameras(
    active_only: bool = False,
    db: AsyncSession = Depends(get_db),
) -> CameraList:
    stmt = select(Camera)
    if active_only:
        stmt = stmt.where(Camera.is_active == True)  # noqa: E712
    result = await db.execute(stmt)
    cameras = result.scalars().all()
    return CameraList(items=[CameraRead.model_validate(c) for c in cameras], total=len(cameras))


@router.post("/", response_model=CameraRead, status_code=status.HTTP_201_CREATED)
async def create_camera(
    body: CameraCreate,
    db: AsyncSession = Depends(get_db),
) -> CameraRead:
    camera = Camera(**body.model_dump())
    db.add(camera)
    await db.commit()
    await db.refresh(camera)
    return CameraRead.model_validate(camera)


@router.get("/{camera_id}", response_model=CameraRead)
async def get_camera(
    camera: Camera = Depends(get_camera_or_404),
) -> CameraRead:
    return CameraRead.model_validate(camera)


@router.patch("/{camera_id}", response_model=CameraRead)
async def update_camera(
    body: CameraUpdate,
    camera: Camera = Depends(get_camera_or_404),
    db: AsyncSession = Depends(get_db),
) -> CameraRead:
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(camera, field, value)
    await db.commit()
    await db.refresh(camera)
    return CameraRead.model_validate(camera)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera: Camera = Depends(get_camera_or_404),
    db: AsyncSession = Depends(get_db),
) -> None:
    # Soft-delete: deactivate rather than remove rows.
    camera.is_active = False
    await db.commit()


@router.post("/{camera_id}/start", response_model=dict)
async def start_camera_pipeline(
    camera_id: str,
    request: Request,
    camera: Camera = Depends(get_camera_or_404),
) -> dict:
    """Start the CV pipeline for a specific camera."""
    pipeline_manager = request.app.state.pipeline_manager
    await pipeline_manager.start_camera(camera_id)
    return {"status": "started", "camera_id": camera_id}


@router.post("/{camera_id}/stop", response_model=dict)
async def stop_camera_pipeline(
    camera_id: str,
    request: Request,
    camera: Camera = Depends(get_camera_or_404),
) -> dict:
    """Stop the CV pipeline for a specific camera."""
    pipeline_manager = request.app.state.pipeline_manager
    await pipeline_manager.stop_camera(camera_id)
    return {"status": "stopped", "camera_id": camera_id}
