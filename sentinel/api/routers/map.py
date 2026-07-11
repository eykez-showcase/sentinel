from __future__ import annotations

import json
import pathlib

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

router = APIRouter()

POSITIONS_FILE = pathlib.Path("config/camera_positions.json")
FLOORPLAN_DIR = pathlib.Path("config")
FLOORPLAN_STEM = "floorplan"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@router.get("/positions")
async def get_positions() -> dict:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text())
    return {}


@router.post("/positions")
async def save_positions(positions: dict) -> dict:
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2))
    return {"saved": True}


@router.get("/floorplan")
async def get_floorplan() -> FileResponse:
    for ext in ALLOWED_EXTS:
        path = FLOORPLAN_DIR / f"{FLOORPLAN_STEM}{ext}"
        if path.exists():
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="No floor plan uploaded yet")


@router.post("/floorplan")
async def upload_floorplan(file: UploadFile = File(...)) -> dict:
    suffix = pathlib.Path(file.filename or "plan.png").suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"File type {suffix} not allowed")
    # Remove any existing floor plan first
    for ext in ALLOWED_EXTS:
        old = FLOORPLAN_DIR / f"{FLOORPLAN_STEM}{ext}"
        if old.exists():
            old.unlink()
    path = FLOORPLAN_DIR / f"{FLOORPLAN_STEM}{suffix}"
    path.write_bytes(await file.read())
    return {"filename": path.name}
