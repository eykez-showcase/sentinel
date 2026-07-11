from __future__ import annotations

"""
FacePipeline: reads from the Jetson's CSI camera via GStreamer subprocess,
runs face detection and recognition, annotates frames, and serves them as MJPEG.

Uses a GStreamer pipeline piped to stdout since OpenCV on this system was built
without GStreamer support.

Events emitted (throttled to once per 60s per identity):
  - face_recognized  → known family member
  - unknown_face     → no match in DB
"""

import asyncio
import logging
import subprocess
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import async_sessionmaker

from sentinel.core.events import EventBus
from sentinel.models.event import Event
from sentinel.schemas.event import EventRead

logger = logging.getLogger(__name__)

EVENT_THROTTLE = 60
TARGET_FPS = 10
WIDTH = 1920
HEIGHT = 1080
FRAME_BYTES = WIDTH * HEIGHT * 3

GST_CMD = [
    "gst-launch-1.0", "-q",
    "nvarguscamerasrc", "sensor-id=0", "!",
    f"video/x-raw(memory:NVMM),width={WIDTH},height={HEIGHT},framerate=30/1", "!",
    "nvvidconv", "!",
    "video/x-raw,format=BGRx", "!",
    "videoconvert", "!",
    f"video/x-raw,format=BGR", "!",
    "fdsink", "fd=1",
]

# RTSP fallback command template
def _rtsp_cmd(url: str) -> list[str]:
    return [
        "gst-launch-1.0", "-q",
        "rtspsrc", f"location={url}", "latency=0", "!",
        "rtph264depay", "!",
        "avdec_h264", "!",
        "videoconvert", "!",
        "video/x-raw,format=BGR", "!",
        "fdsink", "fd=1",
    ]


def _read_frame(proc: subprocess.Popen) -> np.ndarray | None:
    """Read one raw BGR frame from the GStreamer subprocess stdout."""
    raw = b""
    while len(raw) < FRAME_BYTES:
        chunk = proc.stdout.read(FRAME_BYTES - len(raw))
        if not chunk:
            return None
        raw += chunk
    return np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))


class FacePipeline:
    def __init__(
        self,
        face_recognizer,
        db_factory: async_sessionmaker,
        event_bus: EventBus,
    ) -> None:
        self._recognizer = face_recognizer
        self._db_factory = db_factory
        self._bus = event_bus

        self._running = False
        self._task: asyncio.Task | None = None

        self._latest_frame: bytes | None = None
        self._frame_lock = asyncio.Lock()

        self._last_event: dict[str, float] = {}

        self.last_seen_name: str | None = None
        self.last_seen_role: str | None = None
        self.last_seen_confidence: float = 0.0
        self.last_seen_at: str | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="face_pipeline")
        logger.info("FacePipeline started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("FacePipeline stopped")

    async def get_latest_frame(self) -> bytes | None:
        async with self._frame_lock:
            return self._latest_frame

    async def _run(self) -> None:
        from sentinel.core.config import settings
        loop = asyncio.get_running_loop()
        interval = 1.0 / TARGET_FPS

        url = settings.face_camera_url
        cmd = _rtsp_cmd(url) if url.startswith("rtsp://") else GST_CMD
        logger.info("FacePipeline opening camera with: %s", cmd[0])

        proc = None
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            logger.info("FacePipeline GStreamer process started (pid=%d)", proc.pid)

            while self._running:
                tick = loop.time()

                frame = await loop.run_in_executor(None, _read_frame, proc)
                if frame is None:
                    logger.warning("FacePipeline: lost camera feed, restarting in 2s")
                    await asyncio.sleep(2)
                    break

                # Scale down for processing (960x540) while keeping full-res for display
                small = cv2.resize(frame, (960, 540))
                annotated, results = await loop.run_in_executor(
                    None, self._process_frame, small
                )

                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                async with self._frame_lock:
                    self._latest_frame = buf.tobytes()

                for person_id, name, role, confidence in results:
                    await self._maybe_emit_event(person_id, name, role, confidence)

                elapsed = loop.time() - tick
                sleep = interval - elapsed
                if sleep > 0:
                    await asyncio.sleep(sleep)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("FacePipeline error")
        finally:
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def _process_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[str | None, str | None, str | None, float]]]:
        results: list[tuple[str | None, str | None, str | None, float]] = []

        if not self._recognizer.available:
            cv2.putText(
                frame, "Face recognition unavailable — install insightface",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            return frame, results

        try:
            faces = self._recognizer._app.get(frame)
        except Exception:
            return frame, results

        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            emb = face.normed_embedding.astype(np.float32)
            person_id, name, role, confidence = self._recognizer._match_embedding(emb)
            results.append((person_id, name, role, confidence))

            if person_id and role == "family":
                color = (0, 200, 0)
                label = f"{name} ({confidence:.0%})"
            elif person_id:
                color = (0, 200, 200)
                label = f"{name} ({confidence:.0%})"
            else:
                color = (0, 0, 220)
                label = "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.last_seen_name = name or "Unknown"
            self.last_seen_role = role
            self.last_seen_confidence = confidence
            self.last_seen_at = datetime.now(timezone.utc).isoformat()

        return frame, results

    async def _maybe_emit_event(
        self,
        person_id: str | None,
        name: str | None,
        role: str | None,
        confidence: float,
    ) -> None:
        identity_key = person_id or "unknown"
        now = time.monotonic()
        if now - self._last_event.get(identity_key, 0) < EVENT_THROTTLE:
            return
        self._last_event[identity_key] = now

        event_type = "face_recognized" if person_id else "unknown_face"
        severity = "low" if (person_id and role == "family") else "high"

        async with self._db_factory() as session:
            evt = Event(
                camera_id="jetson_camera",
                event_type=event_type,
                class_name="person",
                severity=severity,
                timestamp=datetime.now(timezone.utc),
                metadata_json={
                    "person_id": person_id,
                    "person_name": name,
                    "person_role": role,
                    "confidence": round(confidence, 3),
                },
            )
            session.add(evt)
            await session.commit()
            await session.refresh(evt)

        await self._bus.publish(
            "event.created",
            EventRead.model_validate(evt).model_dump(mode="json"),
        )
        logger.info("Face event: %s — %s (%.0f%%)", event_type, name or "unknown", confidence * 100)
