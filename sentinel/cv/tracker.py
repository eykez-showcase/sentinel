import supervision as sv


class ObjectTracker:
    """
    Wraps supervision's ByteTrack.

    After update(), the returned sv.Detections has tracker_id populated for
    each tracked object. New detections that ByteTrack is not yet confident
    about will have tracker_id = None.
    """

    def __init__(self) -> None:
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Call when a camera stream is restarted to clear track state."""
        self.tracker.reset()
