"""
Sentinel FastAPI application.

Startup sequence:
  1. init_db()            — create tables (dev mode)
  2. load_property_config — parse property.yaml and populate zone registry
  3. sync cameras         — upsert cameras from YAML into the DB
  4. PipelineManager      — starts CV pipelines for all active cameras

Shutdown: all pipelines are stopped cleanly.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from sentinel.api.routers import cameras, detections, events, property, map as map_router
from sentinel.core.config import settings
from sentinel.core.database import AsyncSessionLocal, init_db
from sentinel.core.events import bus
from sentinel.cv.face_recognizer import FaceRecognizer
from sentinel.cv.face_pipeline import FacePipeline
from sentinel.websocket.manager import ConnectionManager, websocket_endpoint

logger = logging.getLogger(__name__)

ws_manager = ConnectionManager(bus, heartbeat_interval=settings.ws_heartbeat_interval)
face_recognizer = FaceRecognizer()
face_pipeline = FacePipeline(face_recognizer, AsyncSessionLocal, bus)


class PipelineManager:
    """
    Manages one CameraPipeline instance per active camera.
    Shared detector and tracker are created once and reused across cameras.
    """

    def __init__(self) -> None:
        from sentinel.cv.detector import YOLODetector
        from sentinel.cv.tracker import ObjectTracker

        self._detector = YOLODetector(
            model_path=settings.yolo_model_path,
            confidence=settings.yolo_confidence_threshold,
            device=settings.yolo_device,
        )
        self._pipelines: dict[str, "CameraPipeline"] = {}  # camera_id → pipeline  # noqa: F821

    async def start_all(self) -> None:
        """Start pipelines for all active cameras registered in the DB."""
        from sqlalchemy import select

        from sentinel.models.camera import Camera

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Camera).where(Camera.is_active == True)  # noqa: E712
            )
            cameras = result.scalars().all()

        for camera in cameras:
            await self.start_camera(camera.id)

    async def start_camera(self, camera_id: str) -> None:
        if camera_id in self._pipelines:
            logger.info("Pipeline already running for camera %s", camera_id)
            return

        from sqlalchemy import select

        from sentinel.cv.pipeline import CameraPipeline
        from sentinel.cv.tracker import ObjectTracker
        from sentinel.models.camera import Camera
        from sentinel.schemas.camera import CameraRead
        from sentinel.zones.loader import zones_for_camera

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Camera).where(Camera.id == camera_id))
            camera = result.scalar_one_or_none()
            if camera is None:
                logger.error("Camera %s not found", camera_id)
                return
            camera_read = CameraRead.model_validate(camera)

        zones = zones_for_camera(camera_read.config_id or camera_id)
        pipeline = CameraPipeline(
            camera=camera_read,
            detector=self._detector,
            tracker=ObjectTracker(),  # each camera gets its own tracker state
            zones=zones,
            db_factory=AsyncSessionLocal,
            event_bus=bus,
            target_fps=settings.default_frame_rate,
            loitering_threshold_seconds=settings.loitering_threshold_seconds,
            face_recognizer=face_recognizer,
        )
        self._pipelines[camera_id] = pipeline
        await pipeline.start()
        logger.info("Started pipeline for camera %s", camera_id)

    async def stop_camera(self, camera_id: str) -> None:
        pipeline = self._pipelines.pop(camera_id, None)
        if pipeline:
            await pipeline.stop()

    async def stop_all(self) -> None:
        for camera_id in list(self._pipelines.keys()):
            await self.stop_camera(camera_id)


async def _sync_cameras_from_yaml() -> None:
    """
    Upsert cameras defined in property.yaml into the database.
    Cameras added via API are left untouched.
    """
    from sqlalchemy import select

    from sentinel.models.camera import Camera
    from sentinel.zones.loader import get_property_config

    cfg = get_property_config()
    async with AsyncSessionLocal() as session:
        for cam_cfg in cfg.cameras:
            result = await session.execute(
                select(Camera).where(Camera.config_id == cam_cfg.id)
            )
            existing = result.scalar_one_or_none()
            if existing is None:
                camera = Camera(
                    name=cam_cfg.name,
                    stream_url=cam_cfg.stream_url,
                    config_id=cam_cfg.id,
                    is_active=True,
                )
                session.add(camera)
                logger.info("Registered camera from YAML: %s", cam_cfg.id)
        await session.commit()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    from sentinel.zones.loader import load_property_config

    # 1. Init database tables.
    await init_db()

    # 2. Load property config (zones, camera positions).
    load_property_config(settings.property_config_path)
    logger.info("Property config loaded from %s", settings.property_config_path)

    # 3. Sync YAML cameras to DB.
    await _sync_cameras_from_yaml()

    # 3.5 Load face embeddings into recognizer.
    async with AsyncSessionLocal() as fr_session:
        from sqlalchemy import select as sa_select
        from sentinel.models.person import FaceEmbedding as FE, Person as P
        fr_result = await fr_session.execute(
            sa_select(FE.person_id, P.name, P.role, FE.embedding).join(P, P.id == FE.person_id)
        )
        fr_rows = fr_result.all()
        face_recognizer.load_embeddings([(r[0], r[1], r[2], r[3]) for r in fr_rows])
        logger.info("Loaded %d face embeddings", len(fr_rows))

    # 4. Start CV pipelines.
    pipeline_manager = PipelineManager()
    app.state.pipeline_manager = pipeline_manager
    await pipeline_manager.start_all()

    # Start face camera pipeline
    await face_pipeline.start()

    yield

    # Shutdown: stop all pipelines cleanly.
    await pipeline_manager.stop_all()
    await face_pipeline.stop()
    logger.info("All pipelines stopped.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentinel",
        description="AI-powered home situational awareness system",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(cameras.router, prefix="/api/v1/cameras", tags=["cameras"])
    app.include_router(detections.router, prefix="/api/v1/detections", tags=["detections"])
    app.include_router(events.router, prefix="/api/v1/events", tags=["events"])
    app.include_router(property.router, prefix="/api/v1/property", tags=["property"])
    app.include_router(map_router.router, prefix="/api/v1/map", tags=["map"])

    from sentinel.api.routers import persons as persons_router
    app.include_router(persons_router.router, prefix="/api/v1/persons", tags=["persons"])

    from sentinel.api.routers import face_camera as face_camera_router
    app.include_router(face_camera_router.router, prefix="/api/v1/face-camera", tags=["face-camera"])

    @app.websocket("/ws")
    async def ws_route(websocket: WebSocket) -> None:
        await websocket_endpoint(websocket, ws_manager)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
