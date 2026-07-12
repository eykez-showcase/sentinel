from __future__ import annotations

"""
FacePipeline: reads from the Jetson CSI camera via GStreamer subprocess,
streams annotated MJPEG video, and runs face recognition as a separate task
so slow CPU inference never blocks the video stream.
"""

import asyncio
import logging
import subprocess
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import async_sessionmaker

from sentinel.core.events import EventBus
from sentinel.cv.audio_alerter import AudioAlerter
from sentinel.models.event import Event
from sentinel.schemas.event import EventRead

logger = logging.getLogger(__name__)

EVENT_THROTTLE = 60
TARGET_FPS = 15
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
    "video/x-raw,format=BGR", "!",
    "fdsink", "fd=1",
]


def _rtsp_cmd(url: str) -> list[str]:
    return [
        "gst-launch-1.0", "-q",
        "rtspsrc", f"location={url}", "latency=0", "!",
        "rtph264depay", "!", "avdec_h264", "!",
        "videoconvert", "!",
        "video/x-raw,format=BGR", "!",
        "fdsink", "fd=1",
    ]


class FrameReader:
    """Drains GStreamer pipe in a background thread, keeping only the latest frame."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while self._alive:
            raw = b""
            while len(raw) < FRAME_BYTES:
                chunk = self._proc.stdout.read(FRAME_BYTES - len(raw))
                if not chunk:
                    self._alive = False
                    return
                raw += chunk
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            with self._lock:
                self._latest = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    @property
    def alive(self) -> bool:
        return self._alive

    def stop(self) -> None:
        self._alive = False


class FacePipeline:
    def __init__(self, face_recognizer, db_factory: async_sessionmaker, event_bus: EventBus) -> None:
        self._recognizer = face_recognizer
        self._db_factory = db_factory
        self._bus = event_bus

        self._running = False
        self._stream_task: asyncio.Task | None = None
        self._recog_task: asyncio.Task | None = None

        # Shared frame buffer for MJPEG
        self._latest_frame: bytes | None = None
        self._frame_lock = asyncio.Lock()

        # Latest raw frame for recognition task to consume
        self._latest_raw: np.ndarray | None = None
        self._raw_lock = threading.Lock()

        # Recognition overlay: list of (x1,y1,x2,y2,label,color)
        self._overlays: list[tuple] = []
        self._overlay_lock = threading.Lock()

        self._last_event: dict[str, float] = {}
        self._audio = AudioAlerter()

        self.last_seen_name: str | None = None
        self.last_seen_role: str | None = None
        self.last_seen_confidence: float = 0.0
        self.last_seen_at: str | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stream_task = asyncio.create_task(self._stream_loop(), name="face_stream")
        self._recog_task = asyncio.create_task(self._recognition_loop(), name="face_recog")
        logger.info("FacePipeline started")

    async def stop(self) -> None:
        self._running = False
        for task in (self._stream_task, self._recog_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("FacePipeline stopped")

    async def get_latest_frame(self) -> bytes | None:
        async with self._frame_lock:
            return self._latest_frame

    # ------------------------------------------------------------------ #
    # Stream loop — reads camera, annotates with cached overlays, encodes #
    # ------------------------------------------------------------------ #
    async def _stream_loop(self) -> None:
        from sentinel.core.config import settings
        loop = asyncio.get_running_loop()
        interval = 1.0 / TARGET_FPS

        url = settings.face_camera_url
        cmd = _rtsp_cmd(url) if url.startswith("rtsp://") else GST_CMD

        while self._running:
            proc = None
            reader = None
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                logger.info("FacePipeline camera process started (pid=%d)", proc.pid)
                reader = FrameReader(proc)

                while self._running:
                    tick = loop.time()

                    if not reader.alive:
                        logger.warning("FacePipeline: camera died, restarting in 3s")
                        break

                    frame = reader.read()
                    if frame is None:
                        await asyncio.sleep(0.05)
                        continue

                    # Share raw frame with recognition task
                    with self._raw_lock:
                        self._latest_raw = frame

                    # Scale for streaming
                    display = cv2.resize(frame, (960, 540))

                    # Draw cached recognition overlays (from last recognition run)
                    with self._overlay_lock:
                        for x1, y1, x2, y2, label, color in self._overlays:
                            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                            cv2.putText(display, label, (x1 + 2, y1 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    async with self._frame_lock:
                        self._latest_frame = buf.tobytes()

                    elapsed = loop.time() - tick
                    sleep = interval - elapsed
                    if sleep > 0:
                        await asyncio.sleep(sleep)

            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("FacePipeline stream error")
            finally:
                if reader:
                    reader.stop()
                if proc:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()

            if self._running:
                await asyncio.sleep(3)

    # ------------------------------------------------------------------ #
    # Recognition loop — runs insightface every second, updates overlays  #
    # ------------------------------------------------------------------ #
    async def _recognition_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                await asyncio.sleep(1.0)  # run recognition once per second

                if not self._recognizer.available:
                    continue

                with self._raw_lock:
                    frame = self._latest_raw.copy() if self._latest_raw is not None else None

                if frame is None:
                    continue

                small = cv2.resize(frame, (960, 540))
                overlays, results = await loop.run_in_executor(
                    None, self._run_recognition, small
                )

                with self._overlay_lock:
                    self._overlays = overlays

                for person_id, name, role, confidence in results:
                    await self._maybe_emit_event(person_id, name, role, confidence)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Recognition loop error")

    def _run_recognition(
        self, frame: np.ndarray
    ) -> tuple[list[tuple], list[tuple]]:
        """Runs in thread pool. Returns (overlays, events)."""
        overlays = []
        results = []

        try:
            faces = self._recognizer._app.get(frame)
        except Exception:
            return overlays, results

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

            overlays.append((x1, y1, x2, y2, label, color))

            self.last_seen_name = name or "Unknown"
            self.last_seen_role = role
            self.last_seen_confidence = confidence
            self.last_seen_at = datetime.now(timezone.utc).isoformat()

        return overlays, results

    async def _maybe_emit_event(self, person_id, name, role, confidence) -> None:
        identity_key = person_id or "unknown"
        now = time.monotonic()
        if now - self._last_event.get(identity_key, 0) < EVENT_THROTTLE:
            return
        self._last_event[identity_key] = now

        # Audio alert
        if not person_id:
            self._audio.alert_unknown()
        elif role == "family":
            self._audio.alert_family(name)
        else:
            self._audio.alert_known(name)

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
