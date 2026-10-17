import base64
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

    def save_one_record(self, timestamp, camera_id, track_id, event_type, snapshot):
        # Lưu thông tin và ảnh vào SQLite.
        with self.conn:
            cursor = self.conn.execute("""
            INSERT INTO events (
            timestamp, camera_id, track_id, event_type, snapshot_base64)
            VALUES (?, ?, ?, ?, ?)
            """, (timestamp, camera_id, track_id, event_type, snapshot))
            record_id = cursor.lastrowid
        logger.info("Committed event: id=%d track_id=%s event_type=%s", record_id, track_id,
                    event_type, extra={"camera_id": camera_id})
        return record_id

    def save(self, event_result: EventResults):

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

        snapshot_base64 = base64.b64encode( encoded.tobytes()).decode("ascii")

        saved_ids = []
        with self.conn:
            for track_id, event_type in records:
                cursor = self.conn.execute(
                    "INSERT INTO events (timestamp, camera_id, track_id, event_type, snapshot_base64) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (event_result.timestamp.isoformat(), event_result.camera_id,
                     track_id, event_type, snapshot_base64),
                )
                saved_ids.append(cursor.lastrowid)
        logger.info("Committed %d event records", len(saved_ids),
                    extra={"camera_id": event_result.camera_id})
        return saved_ids

    def get(self, event_id):
        cursor = self.conn.execute("""
            SELECT id, timestamp, camera_id, track_id,
                   event_type, snapshot_base64
            FROM events
            WHERE id = ?
        """, (event_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        columns = [column[0] for column in cursor.description]
        return dict(zip(columns, row))

    def close(self):
        self.conn.close()
        logger.info("Event database closed")
