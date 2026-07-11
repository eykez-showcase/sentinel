from __future__ import annotations

"""
CameraPipeline: orchestrates the full frame-processing loop for one camera.

Data flow per frame:
  VideoCapture → YOLODetector → ObjectTracker → ZoneMapper
  → persist Detections + upsert Tracks → emit Events → publish to EventBus

The capture loop runs as an asyncio Task. cv2.VideoCapture.read() is blocking;
we offload it to a thread via run_in_executor so it doesn't stall the loop.
"""

import asyncio
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sentinel.core.events import EventBus
from sentinel.cv.detector import YOLODetector
from sentinel.cv.face_recognizer import FaceRecognizer
from sentinel.cv.tracker import ObjectTracker
from sentinel.cv.zone_mapper import ZoneMapper
from sentinel.models.camera import Camera
from sentinel.models.detection import Detection
from sentinel.models.event import Event
from sentinel.models.track import Track
from sentinel.schemas.camera import CameraRead
from sentinel.schemas.detection import DetectionRead
from sentinel.schemas.event import EventRead
from sentinel.zones.loader import ZoneConfig

logger = logging.getLogger(__name__)


class CameraPipeline:
    def __init__(
        self,
        camera: CameraRead,
        detector: YOLODetector,
        tracker: ObjectTracker,
        zones: list[ZoneConfig],
        db_factory: async_sessionmaker,
        event_bus: EventBus,
        target_fps: int = 10,
        loitering_threshold_seconds: int = 30,
        face_recognizer: FaceRecognizer | None = None,
    ) -> None:
        self.camera = camera
        self._detector = detector
        self._tracker = tracker
        self._zones = zones
        self._db_factory = db_factory
        self._bus = event_bus
        self._frame_interval = 1.0 / max(target_fps, 1)
        self._loitering_threshold = loitering_threshold_seconds
        self._face_recognizer = face_recognizer

        self._running = False
        self._task: asyncio.Task | None = None

        # Track previous zone assignments to detect zone entry/exit.
        # Key: tracker_id (int), Value: (zone_id | None, entry_time)
        self._track_state: dict[int, tuple[str | None, datetime]] = {}

        self._zone_mapper: ZoneMapper | None = None

        # Face recognition cache: tracker_id → (person_id, name, role, confidence, cached_at)
        self._recognition_cache: dict[int, tuple[str | None, str | None, str | None, float, datetime]] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name=f"pipeline:{self.camera.id}")
        logger.info("Pipeline started for camera %s", self.camera.id)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Pipeline stopped for camera %s", self.camera.id)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        cap: cv2.VideoCapture | None = None

        try:
            cap = await loop.run_in_executor(
                None, cv2.VideoCapture, self.camera.stream_url
            )
            if not cap.isOpened():
                logger.error("Cannot open stream: %s", self.camera.stream_url)
                return

            frame_number = 0
            while self._running:
                tick = loop.time()
                ret, frame = await loop.run_in_executor(None, cap.read)

                if not ret:
                    logger.warning(
                        "Camera %s: failed to read frame, retrying in 2s",
                        self.camera.id,
                    )
                    await asyncio.sleep(2)
                    continue

                await self._process_frame(frame, frame_number)
                frame_number += 1

                # Hard throttle to target FPS.
                elapsed = loop.time() - tick
                sleep_for = self._frame_interval - elapsed
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Pipeline error for camera %s", self.camera.id)
        finally:
            if cap is not None:
                cap.release()

    async def _process_frame(self, frame: np.ndarray, frame_number: int) -> None:
        h, w = frame.shape[:2]

        # Lazy-init the zone mapper when we know the resolution.
        if self._zone_mapper is None and self._zones:
            self._zone_mapper = ZoneMapper(self._zones, (w, h))

        # Run detection + tracking in the thread pool so we don't block asyncio.
        loop = asyncio.get_running_loop()
        detections = await loop.run_in_executor(None, self._detector.detect, frame)
        detections = await loop.run_in_executor(None, self._tracker.update, detections)

        zone_ids: list[str | None]
        if self._zone_mapper is not None:
            zone_ids = await loop.run_in_executor(
                None, self._zone_mapper.map, detections
            )
        else:
            zone_ids = [None] * len(detections)

        now = datetime.now(timezone.utc)

        # Face recognition: run once per tracker_id (cache for 30s)
        face_results: dict[int, tuple[str | None, str | None, str | None, float]] = {}
        if self._face_recognizer and self._face_recognizer.available and detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                if tid is None:
                    continue
                tid_int = int(tid)
                class_name = (
                    detections.data["class_name"][i]
                    if "class_name" in detections.data
                    else str(int(detections.class_id[i]))
                )
                if class_name != "person":
                    continue

                # Check cache
                cached = self._recognition_cache.get(tid_int)
                if cached and (now - cached[4]).total_seconds() < 30:
                    face_results[tid_int] = cached[:4]
                    continue

                # Crop face region from frame
                bbox = detections.xyxy[i]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                h_frame, w_frame = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                pid, pname, prole, conf = await loop.run_in_executor(
                    None, self._face_recognizer.recognize, crop
                )
                face_results[tid_int] = (pid, pname, prole, conf)
                self._recognition_cache[tid_int] = (pid, pname, prole, conf, now)

        async with self._db_factory() as session:
            det_records = await self._persist_detections(
                session, detections, zone_ids, frame_number, now
            )
            track_records = await self._upsert_tracks(
                session, detections, zone_ids, now
            )
            event_records = await self._evaluate_events(
                session, detections, zone_ids, track_records, now, face_results
            )
            await session.commit()

        # Publish to the WebSocket event bus.
        for det in det_records:
            await self._bus.publish(
                "detection.new",
                DetectionRead.model_validate(det).model_dump(mode="json"),
            )
        for evt in event_records:
            await self._bus.publish(
                "event.created",
                EventRead.model_validate(evt).model_dump(mode="json"),
            )

    async def _persist_detections(
        self,
        session: AsyncSession,
        detections,
        zone_ids: list[str | None],
        frame_number: int,
        now: datetime,
    ) -> list[Detection]:
        records: list[Detection] = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            tracker_id = (
                int(detections.tracker_id[i])
                if detections.tracker_id is not None
                else None
            )
            class_name = (
                detections.data["class_name"][i]
                if "class_name" in detections.data
                else str(int(detections.class_id[i]))
            )
            det = Detection(
                camera_id=self.camera.id,
                frame_number=frame_number,
                timestamp=now,
                class_name=class_name,
                confidence=float(detections.confidence[i]),
                bbox_x1=float(bbox[0]),
                bbox_y1=float(bbox[1]),
                bbox_x2=float(bbox[2]),
                bbox_y2=float(bbox[3]),
                tracker_id=tracker_id,
                zone_id=zone_ids[i],
            )
            session.add(det)
            records.append(det)
        return records

    async def _upsert_tracks(
        self,
        session: AsyncSession,
        detections,
        zone_ids: list[str | None],
        now: datetime,
    ) -> list[Track]:
        if detections.tracker_id is None:
            return []

        records: list[Track] = []
        for i, tid in enumerate(detections.tracker_id):
            if tid is None:
                continue
            tid = int(tid)
            class_name = (
                detections.data["class_name"][i]
                if "class_name" in detections.data
                else str(int(detections.class_id[i]))
            )
            zone_id = zone_ids[i]

            stmt = select(Track).where(
                Track.camera_id == self.camera.id,
                Track.tracker_id == tid,
                Track.is_active == True,  # noqa: E712
            )
            result = await session.execute(stmt)
            track = result.scalar_one_or_none()

            if track is None:
                track = Track(
                    camera_id=self.camera.id,
                    tracker_id=tid,
                    class_name=class_name,
                    first_seen=now,
                    last_seen=now,
                    zone_id=zone_id,
                    is_active=True,
                )
                session.add(track)
            else:
                track.last_seen = now
                track.zone_id = zone_id

            records.append(track)

        return records

    async def _evaluate_events(
        self,
        session: AsyncSession,
        detections,
        zone_ids: list[str | None],
        track_records: list[Track],
        now: datetime,
        face_results: dict[int, tuple] | None = None,
    ) -> list[Event]:
        events: list[Event] = []
        if face_results is None:
            face_results = {}

        if detections.tracker_id is None:
            return events

        track_by_tid = {t.tracker_id: t for t in track_records}

        for i, tid in enumerate(detections.tracker_id):
            if tid is None:
                continue
            tid = int(tid)
            current_zone = zone_ids[i]
            track = track_by_tid.get(tid)

            prev_zone, zone_entry_time = self._track_state.get(
                tid, (None, now)
            )

            # Zone entry.
            if current_zone != prev_zone:
                if current_zone is not None:
                    face_meta = face_results.get(tid)
                    evt = Event(
                        camera_id=self.camera.id,
                        track_id=track.id if track else None,
                        event_type="zone_entry",
                        zone_id=current_zone,
                        class_name=track.class_name if track else None,
                        severity=self._severity_for_zone_entry(current_zone, track),
                        timestamp=now,
                        metadata_json={
                            "person_id": face_meta[0] if face_meta else None,
                            "person_name": face_meta[1] if face_meta else None,
                            "person_role": face_meta[2] if face_meta else None,
                            "face_confidence": face_meta[3] if face_meta else 0.0,
                        } if face_meta is not None else None,
                    )
                    session.add(evt)
                    events.append(evt)

                if prev_zone is not None:
                    evt = Event(
                        camera_id=self.camera.id,
                        track_id=track.id if track else None,
                        event_type="zone_exit",
                        zone_id=prev_zone,
                        class_name=track.class_name if track else None,
                        severity="low",
                        timestamp=now,
                    )
                    session.add(evt)
                    events.append(evt)

                self._track_state[tid] = (current_zone, now)

            else:
                # Same zone — check for loitering.
                if current_zone is not None:
                    dwell = (now - zone_entry_time).total_seconds()
                    if dwell >= self._loitering_threshold:
                        # Emit loitering once when threshold is first crossed.
                        # Reset timer so we don't spam events.
                        if dwell < self._loitering_threshold + self._frame_interval + 1:
                            evt = Event(
                                camera_id=self.camera.id,
                                track_id=track.id if track else None,
                                event_type="loitering",
                                zone_id=current_zone,
                                class_name=track.class_name if track else None,
                                severity="medium",
                                timestamp=now,
                                metadata_json={"dwell_seconds": int(dwell)},
                            )
                            session.add(evt)
                            events.append(evt)
                            # Reset entry time so the threshold check resets.
                            self._track_state[tid] = (current_zone, now)

        return events

    def _severity_for_zone_entry(
        self, zone_id: str, track: Track | None
    ) -> str:
        """
        High-sensitivity zones (front_porch, perimeter) get medium severity.
        Unknown persons get upgraded to high. Everything else is low.
        """
        sensitive_zones = {"front_porch", "side_gate", "backyard"}
        if zone_id in sensitive_zones:
            if track and track.class_name == "person":
                return "high"
            return "medium"
        return "low"
