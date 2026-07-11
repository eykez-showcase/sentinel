from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()


def _get_pipeline():
    from sentinel.api.main import face_pipeline
    return face_pipeline


@router.get("/stream")
async def mjpeg_stream():
    """MJPEG stream from the Jetson camera with face annotation."""
    pipeline = _get_pipeline()

    async def generate():
        while True:
            frame = await pipeline.get_latest_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(1 / 15)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/status")
async def get_status():
    """Current face recognition status."""
    pipeline = _get_pipeline()
    return {
        "available": pipeline._recognizer.available,
        "last_seen_name": pipeline.last_seen_name,
        "last_seen_role": pipeline.last_seen_role,
        "last_seen_confidence": pipeline.last_seen_confidence,
        "last_seen_at": pipeline.last_seen_at,
    }
