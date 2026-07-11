from __future__ import annotations

"""
Parse property.yaml into strongly-typed config objects.
Called once at startup; results cached in module-level state.
"""

from dataclasses import dataclass, field

import yaml


@dataclass
class ZoneConfig:
    id: str
    name: str
    camera_id: str
    # Normalized [0.0–1.0] (x, y) coordinates, clockwise from top-left.
    polygon: list[tuple[float, float]]
    alert_classes: list[str] = field(default_factory=list)


@dataclass
class CameraConfig:
    id: str
    name: str
    stream_url: str
    position: dict
    fov_degrees: float = 90.0


@dataclass
class PropertyConfig:
    name: str
    address: str
    cameras: list[CameraConfig]
    zones: list[ZoneConfig]


_config: PropertyConfig | None = None


def load_property_config(path: str) -> PropertyConfig:
    global _config
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    prop = raw.get("property", {})
    cameras = [
        CameraConfig(
            id=c["id"],
            name=c["name"],
            stream_url=c["stream_url"],
            position=c.get("position", {}),
            fov_degrees=float(c.get("fov_degrees", 90)),
        )
        for c in raw.get("cameras", [])
    ]
    zones = [
        ZoneConfig(
            id=z["id"],
            name=z["name"],
            camera_id=z["camera_id"],
            polygon=[tuple(p) for p in z["polygon"]],
            alert_classes=z.get("alert_classes", []),
        )
        for z in raw.get("zones", [])
    ]
    _config = PropertyConfig(
        name=prop.get("name", ""),
        address=prop.get("address", ""),
        cameras=cameras,
        zones=zones,
    )
    return _config


def get_property_config() -> PropertyConfig:
    if _config is None:
        raise RuntimeError("Property config not loaded. Call load_property_config first.")
    return _config


def zones_for_camera(camera_id: str) -> list[ZoneConfig]:
    cfg = get_property_config()
    return [z for z in cfg.zones if z.camera_id == camera_id]
