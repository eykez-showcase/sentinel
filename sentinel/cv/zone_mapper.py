from __future__ import annotations

"""
Maps pixel-space bounding boxes to named property zones.

Zone polygons are stored in normalized [0,1] coordinates in property.yaml.
At mapping time we scale them to pixel coordinates using the frame dimensions.
"""

import numpy as np
import supervision as sv

from sentinel.zones.loader import ZoneConfig


class ZoneMapper:
    def __init__(self, zones: list[ZoneConfig], frame_wh: tuple[int, int]) -> None:
        """
        Build PolygonZone objects for a specific resolution.

        frame_wh: (width, height) in pixels. If the stream resolution changes,
        create a new ZoneMapper instance.
        """
        self._zone_configs = zones
        self._frame_wh = frame_wh
        self._sv_zones: list[tuple[str, sv.PolygonZone]] = []

        w, h = frame_wh
        for zone in zones:
            pixel_polygon = np.array(
                [(int(x * w), int(y * h)) for x, y in zone.polygon],
                dtype=np.int32,
            )
            sv_zone = sv.PolygonZone(polygon=pixel_polygon)
            self._sv_zones.append((zone.id, sv_zone))

    def map(self, detections: sv.Detections) -> list[str | None]:
        """
        For each detection return the ID of the first matching zone, or None.

        Zones are checked in declaration order; first match wins.
        """
        if len(detections) == 0:
            return []

        result: list[str | None] = [None] * len(detections)

        for zone_id, sv_zone in self._sv_zones:
            in_zone = sv_zone.trigger(detections)
            for i, hit in enumerate(in_zone):
                if hit and result[i] is None:
                    result[i] = zone_id

        return result
