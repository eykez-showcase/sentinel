from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/")
async def get_property() -> dict:
    """Return the loaded property config (zones, cameras) for the dashboard map."""
    from sentinel.zones.loader import get_property_config

    try:
        cfg = get_property_config()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Property config not yet loaded.")

    return {
        "name": cfg.name,
        "cameras": [
            {
                "id": c.id,
                "name": c.name,
                "position": c.position,
                "fov_degrees": c.fov_degrees,
            }
            for c in cfg.cameras
        ],
        "zones": [
            {
                "id": z.id,
                "name": z.name,
                "camera_id": z.camera_id,
                "polygon": z.polygon,
                "alert_classes": z.alert_classes,
            }
            for z in cfg.zones
        ],
    }
