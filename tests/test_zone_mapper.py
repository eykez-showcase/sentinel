"""
Tests for ZoneMapper — the system that maps pixel bounding boxes to named zones.
"""

import numpy as np
import pytest
import supervision as sv

from sentinel.cv.zone_mapper import ZoneMapper
from sentinel.zones.loader import ZoneConfig

FRAME_WH = (1280, 720)

ZONES = [
    ZoneConfig(
        id="front_porch",
        name="Front Porch",
        camera_id="cam1",
        # Normalized: right half, bottom half of frame.
        polygon=[(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)],
        alert_classes=["person"],
    ),
    ZoneConfig(
        id="driveway",
        name="Driveway",
        camera_id="cam1",
        # Normalized: left half, top half of frame.
        polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
        alert_classes=["person", "car"],
    ),
]


def make_detections(bboxes_xyxy: list[list[float]]) -> sv.Detections:
    """Helper: build a minimal sv.Detections from pixel bounding boxes."""
    if not bboxes_xyxy:
        return sv.Detections.empty()
    xyxy = np.array(bboxes_xyxy, dtype=np.float32)
    return sv.Detections(xyxy=xyxy)


def test_centroid_inside_front_porch():
    mapper = ZoneMapper(ZONES, FRAME_WH)
    # Centroid at (960, 540) → normalized (0.75, 0.75) → front_porch
    det = make_detections([[900, 500, 1020, 580]])
    result = mapper.map(det)
    assert result == ["front_porch"]


def test_centroid_inside_driveway():
    mapper = ZoneMapper(ZONES, FRAME_WH)
    # Centroid at (320, 180) → normalized (0.25, 0.25) → driveway
    det = make_detections([[280, 160, 360, 200]])
    result = mapper.map(det)
    assert result == ["driveway"]


def test_centroid_outside_all_zones():
    mapper = ZoneMapper(ZONES, FRAME_WH)
    # Center of frame → (640, 360) → normalized (0.5, 0.5) is on boundary;
    # put it clearly between zones: bottom-left quadrant isn't covered.
    # Bottom-left centroid: (320, 540) → normalized (0.25, 0.75) → no zone
    det = make_detections([[280, 520, 360, 560]])
    result = mapper.map(det)
    assert result == [None]


def test_multiple_detections():
    mapper = ZoneMapper(ZONES, FRAME_WH)
    det = make_detections([
        [900, 500, 1020, 580],  # front_porch
        [280, 160, 360, 200],   # driveway
        [280, 520, 360, 560],   # None
    ])
    result = mapper.map(det)
    assert result == ["front_porch", "driveway", None]


def test_empty_detections():
    mapper = ZoneMapper(ZONES, FRAME_WH)
    det = make_detections([])
    result = mapper.map(det)
    assert result == []


def test_first_match_wins_for_overlapping_zones():
    """When zones overlap, the first one in the list takes priority."""
    overlapping_zones = [
        ZoneConfig(
            id="zone_a",
            name="Zone A",
            camera_id="cam1",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            alert_classes=[],
        ),
        ZoneConfig(
            id="zone_b",
            name="Zone B",
            camera_id="cam1",
            polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            alert_classes=[],
        ),
    ]
    mapper = ZoneMapper(overlapping_zones, FRAME_WH)
    det = make_detections([[580, 340, 700, 400]])
    result = mapper.map(det)
    assert result == ["zone_a"]


def test_resolution_scaling():
    """Zone mapper built for 640×480 should give same logical result as 1280×720."""
    mapper_hd = ZoneMapper(ZONES, (1280, 720))
    mapper_sd = ZoneMapper(ZONES, (640, 480))

    # front_porch centroid in HD: (960, 540); in SD: (480, 360)
    det_hd = make_detections([[920, 520, 1000, 560]])
    det_sd = make_detections([[460, 345, 500, 380]])

    assert mapper_hd.map(det_hd) == ["front_porch"]
    assert mapper_sd.map(det_sd) == ["front_porch"]
