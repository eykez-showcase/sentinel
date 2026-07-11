import numpy as np
import supervision as sv
from ultralytics import YOLO


class YOLODetector:
    """
    Thin wrapper around Ultralytics YOLO that returns supervision Detections.

    Using supervision's Detections as the common currency means the output
    composes directly with ByteTrack and PolygonZone without conversion.
    """

    def __init__(self, model_path: str, confidence: float, device: str) -> None:
        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        # Filter by confidence threshold.
        mask = detections.confidence >= self.confidence
        return detections[mask]
