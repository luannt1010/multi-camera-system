from datetime import datetime
import logging
import cv2
import numpy as np
from .schemas import Event, EventResults, TrackingPacket


logger = logging.getLogger(__name__)


class EventEngine:
    def __init__(self, config: Event):
        self.config = config
        self.previous_side = {}
        self.previous_center = {}
        self.inside = {}
        self.last_seen = {}
        self.left_count = self.right_count = 0

    def geometry(self, shape):
        """Hàm chuyển đổi line polygon"""
        scale = np.array([shape[1], shape[0]]) if self.config.normalized else 1
        if self.config.line is not None:
            if self.config.polygon is not None:
                return (np.asarray(self.config.line, dtype=np.float32) * scale,
                        np.asarray(self.config.polygon, dtype=np.float32) * scale)
            else:
                return (np.asarray(self.config.line, dtype=np.float32) * scale, None)
        else:
            if self.config.polygon is not None:
                return (None, np.asarray(self.config.polygon, dtype=np.float32) * scale)
            else:
                return (None, None)

    @staticmethod
    def _side(point, line):
        """Hàm tính side: AB = AxBy - AyBx
        side: 1, -1, 0"""
        a, b = line
        return float((b[0] - a[0]) * (point[1] - a[1]) -
                     (b[1] - a[1]) * (point[0] - a[0]))

    def reset(self):
        """Hàm xóa trạng thái nếu camera bị gián đoạn vẫn giữ tổng đã đếm"""
        for state in (self.previous_side, self.previous_center, self.inside, self.last_seen):
            state.clear()

    def process(self, packet: TrackingPacket):
        """Hàm xử lý sự kiện"""
        if not self.config.enabled:
            return EventResults(packet.camera_id, datetime.now(), packet.frame, {}, self.left_count, self.right_count)
        line, polygon = self.geometry(packet.frame.shape)
        # Bỏ object xuất hiện lần cuối lâu hơn state_ttl
        for tid, seen in list(self.last_seen.items()):
            if packet.timestamp - seen > self.config.state_ttl:
                for state in (self.last_seen, self.previous_side, self.previous_center, self.inside):
                    state.pop(tid, None)
        events = {}
        for box, tid in zip(packet.boxes, packet.track_ids):
            tid = int(tid)
            x1, y1, x2, y2 = box
            center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
            side = self._side(center, line) if line is not None else 0
            previous = self.previous_side.get(tid, 0)
            crossed = previous * side < 0
            if crossed:
                # Kiểm tra xem hai đâu AB có cắt đường thẳng tạo thành bới tâm cũ và mới
                movement = (self.previous_center[tid], center) # tâm cũ, mới
                crossed = self._side(line[0], movement) * self._side(line[1], movement) <= 0
            direction = None
            if crossed:
                direction = "left_to_right" if side > 0 else "right_to_left"
                self.right_count += int(side > 0)
                self.left_count += int(side < 0)
            inside = (cv2.pointPolygonTest(polygon.astype(np.float32), center, False) >= 0
                      if polygon is not None else False)
            events[tid] = {"in_prohibit": inside,
                           "zone_entry": inside and not self.inside.get(tid, False),
                           "crossed": bool(crossed), "direction": direction}
            if events[tid]["zone_entry"]:
                logger.debug("Zone entry detected: track_id=%d", tid,
                             extra={"camera_id": packet.camera_id})
            if crossed:
                logger.debug("Line crossing detected: track_id=%d direction=%s", tid, direction,
                             extra={"camera_id": packet.camera_id})
            if side != 0:
                self.previous_side[tid] = side
                self.previous_center[tid] = center
            if polygon is not None:
                self.inside[tid] = inside
            self.last_seen[tid] = packet.timestamp
        return EventResults(packet.camera_id, datetime.now(), packet.frame,
                            events, self.left_count, self.right_count)
