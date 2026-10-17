import logging

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from .schemas import DetectionPacket, TrackingPacket


logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, fps=30):
        self.fps = fps
        self.trackers = {}

    def _get_tracker(self, camera_id: str):
        if camera_id not in self.trackers:
            self.trackers[camera_id] = ByteTrackTracker(
                frame_rate=self.fps,
                track_activation_threshold=0.5,
                high_conf_det_threshold=0.6,
                lost_track_buffer=30,
                minimum_consecutive_frames=2,
                minimum_iou_threshold=0.1
            )
            logger.info("Tracker created (fps=%s)", self.fps, extra={"camera_id": camera_id})

        return self.trackers[camera_id]

    def update(self, packet: DetectionPacket) -> TrackingPacket:
        tracker = self._get_tracker(packet.camera_id)

        detections = sv.Detections(
            xyxy=packet.boxes,
            confidence=packet.scores,
            class_id=packet.class_ids
        )

        tracked = tracker.update(detections)

        if len(tracked) == 0 or tracked.tracker_id is None:
            return TrackingPacket(
                camera_id=packet.camera_id,
                generation=packet.generation,
                timestamp=packet.timestamp,
                frame=packet.frame,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                track_ids=np.empty((0,), dtype=np.int32)
            )

        # Filter invalid/unconfirmed tracks
        valid = tracked.tracker_id >= 0
        tracked = tracked[valid]

        return TrackingPacket(
            camera_id=packet.camera_id,
            generation=packet.generation,
            timestamp=packet.timestamp,
            frame=packet.frame,
            boxes=tracked.xyxy.astype(np.float32),
            scores=tracked.confidence.astype(np.float32),
            class_ids=tracked.class_id.astype(np.int32),
            track_ids=tracked.tracker_id.astype(np.int32)
        )

    def reset(self, camera_id: str):
        if self.trackers.pop(camera_id, None) is not None:
            logger.debug("Tracker reset", extra={"camera_id": camera_id})
