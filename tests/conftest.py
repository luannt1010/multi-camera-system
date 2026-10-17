"""Small synthetic tracking packets shared by the EventEngine unit tests."""
import numpy as np
import pytest

from src.schemas import TrackingPacket


@pytest.fixture
def tracking_packet_factory():
    """Build a packet from (track_id, center_x, center_y), without inference."""
    def make(objects=(), *, timestamp=0.0, camera_id="cam_01", frame_shape=(100, 100, 3)):
        objects = list(objects)
        return TrackingPacket(
            camera_id=camera_id,
            timestamp=timestamp,
            frame=np.zeros(frame_shape, dtype=np.uint8),
            boxes=np.asarray([[x - 2, y - 2, x + 2, y + 2] for _, x, y in objects],
                             dtype=np.float32).reshape((-1, 4)),
            scores=np.full(len(objects), .9, dtype=np.float32),
            class_ids=np.zeros(len(objects), dtype=np.int32),
            track_ids=np.asarray([track_id for track_id, _, _ in objects], dtype=np.int32),
        )
    return make
