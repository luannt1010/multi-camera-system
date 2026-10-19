import base64
from datetime import datetime
import logging
import cv2
import sqlite3
from pathlib import Path
from .schemas import EventResults


logger = logging.getLogger(__name__)


class EventWriter:
    def __init__(self, dp_path):
        self.dp_path = Path(dp_path)
        self.dp_path.parent.mkdir(exist_ok=True, parents=True)
        self.conn = sqlite3.connect(str(self.dp_path), timeout=10)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                snapshot_base64 TEXT NOT NULL)
                """)
        self.conn.commit()
        logger.info("Event database initialized: %s", self.dp_path)

    def _insert_record(self, timestamp, camera_id, track_id, event_type, snapshot):
        """Hàm insert 1 record"""
        timestamp = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        cursor = self.conn.execute("""
        INSERT INTO events (timestamp, camera_id, track_id, event_type, snapshot_base64)
        VALUES (?, ?, ?, ?, ?)
        """, (timestamp, camera_id, track_id, event_type, snapshot))
        return cursor.lastrowid

    def save(self, event_result: EventResults):
        """Hàm save nhiều record"""
        records = []
        for track_id, value in event_result.events.items():
            if value.get("zone_entry", value["in_prohibit"]):
                records.append((int(track_id), "zone_entry"))
            if value["crossed"]:
                direction = value["direction"]
                records.append((int(track_id), f"crossed_{direction}"))
        if not records:
            return []

        success, encoded = cv2.imencode( ".jpg", event_result.frame, [cv2.IMWRITE_JPEG_QUALITY, 80],)
        if not success:
            raise RuntimeError("Không mã hóa được snapshot")
        # encode base64
        snapshot_base64 = base64.b64encode( encoded.tobytes()).decode("ascii")

        saved_ids = []
        with self.conn:
            for track_id, event_type in records:
                lastrowid = self._insert_record(event_result.timestamp, event_result.camera_id,
                                               track_id, event_type, snapshot_base64)
                saved_ids.append(lastrowid)
        logger.info("Committed %d event records", len(saved_ids),
                    extra={"camera_id": event_result.camera_id})
        return saved_ids

    def get(self, event_id):
        """Hàm lấy theo event id"""
        cursor = self.conn.execute("""
        SELECT id, timestamp, camera_id, track_id, event_type, snapshot_base64
        FROM events
        WHERE id = ?
        """, (event_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [column[0] for column in cursor.description]
        return dict(zip(columns, row))

    def list_events(self, *, camera_id=None, start_time: datetime | None = None,
                    end_time: datetime | None = None, limit=100, offset=0):
        """Lấy metadata từ mới đến cũ, bao gồm cả hai mốc thời gian."""
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("limit phải là số nguyên từ 1 đến 500")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset phải là số nguyên không âm")
        if camera_id is not None and (not isinstance(camera_id, str) or not camera_id.strip()):
            raise ValueError("camera_id phải là chuỗi không rỗng")
        for value in (start_time, end_time):
            if value is not None and (not isinstance(value, datetime) or value.utcoffset() is not None):
                raise ValueError("Thời gian phải là datetime theo giờ máy, không kèm múi giờ")
        if start_time is not None and end_time is not None and start_time > end_time:
            raise ValueError("start_time không được lớn hơn end_time")
        start = start_time.isoformat() if start_time is not None else None
        end = end_time.isoformat() if end_time is not None else None

        clauses, parameters = [], []
        for column, operator, value in (("camera_id", "=", camera_id), ("timestamp", ">=", start), ("timestamp", "<=", end)):
            if value is not None:
                clauses.append(f"{column} {operator} ?")
                parameters.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.conn:
            # COUNT và danh sách dùng cùng snapshot dù writer đang thêm dữ liệu.
            if not self.conn.in_transaction:
                self.conn.execute("BEGIN")
            total = self.conn.execute("SELECT COUNT(*) FROM events" + where, parameters).fetchone()[0]
            cursor = self.conn.execute(
                "SELECT id, timestamp, camera_id, track_id, event_type "
                "FROM events" + where + " ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            )
            columns = [column[0] for column in cursor.description]
            items = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return {"items": items, "total": total, "limit": limit, "offset": offset}

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("Event database closed")
