"""Kiểm tra lịch sử sự kiện bằng DB tạm, không chạm dữ liệu camera thật."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
import sqlite3
from threading import Barrier

import numpy as np
import pytest

from src.event_writer import EventWriter
from src.output_workers import WriterWorker
from src.schemas import EventResults


EVENT_COLUMNS = [
    "id", "timestamp", "camera_id", "track_id", "event_type", "snapshot_base64",
]


def create_legacy_database(path, rows):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("""
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                snapshot_base64 TEXT NOT NULL
            )
        """)
        connection.executemany(
            "INSERT INTO events(timestamp,camera_id,track_id,event_type,snapshot_base64) "
            "VALUES (?,?,?,?,?)", rows,
        )
        connection.commit()


@pytest.fixture
def event_database(tmp_path):
    path = tmp_path / "events.sqlite3"
    writer = EventWriter(path)
    yield path, writer
    writer.close()


def test_constructor_creates_database_and_empty_event_table(tmp_path):
    path = tmp_path / "nested" / "events.sqlite3"
    with closing(EventWriter(path)) as writer:
        assert path.is_file()
        assert writer.conn is not None
        assert writer.get(1) is None
        assert writer.list_events() == {"items": [], "total": 0, "limit": 100, "offset": 0}


@pytest.mark.parametrize("timestamp", [
    "2026-10-17T12:00:00",
    "2026-10-17T12:00:00.000123",
    "2026-10-17T12:00:00+07:00",
    "2026-10-17T05:00:00Z",
])
def test_writer_and_reader_preserve_timestamp_strings(event_database, timestamp):
    path, writer = event_database
    event_id = writer.save_one_record(timestamp, "cam_01", 12, "zone_entry", "snapshot")
    assert writer.get(event_id) == {
        "id": event_id, "timestamp": timestamp, "camera_id": "cam_01",
        "track_id": 12, "event_type": "zone_entry", "snapshot_base64": "snapshot",
    }
    assert writer.list_events()["items"] == [{
        "id": event_id, "timestamp": timestamp, "camera_id": "cam_01",
        "track_id": 12, "event_type": "zone_entry",
    }]


def test_writer_close_can_be_called_more_than_once(event_database):
    path, writer = event_database
    event_id = writer.save_one_record(datetime.now(), "cam_01", 1, "zone_entry", "image")
    writer.close()
    writer.close()
    with closing(EventWriter(path)) as reader:
        assert reader.get(event_id)["snapshot_base64"] == "image"


def test_get_keeps_snapshot_and_parameterizes_event_id(event_database):
    path, writer = event_database
    event_id = writer.save_one_record(datetime.now(), "cam_01", 12, "zone_entry", "snapshot")
    reader = writer
    assert reader.get(event_id)["snapshot_base64"] == "snapshot"
    assert reader.get(event_id + 1) is None
    assert reader.get("1 OR 1=1") is None


def test_save_one_record_accepts_local_datetime_without_conversion(event_database):
    path, writer = event_database
    timestamp = datetime.now()
    event_id = writer.save_one_record(timestamp, "cam_01", 12, "zone_entry", "snapshot")
    assert writer.get(event_id)["timestamp"] == timestamp.isoformat()
    assert writer.list_events()["items"][0]["timestamp"] == timestamp.isoformat()


def test_writer_saves_datetime_now_for_each_event_without_conversion(event_database):
    path, writer = event_database
    timestamp = datetime.now()
    event = EventResults(
        camera_id="cam_02", timestamp=timestamp, frame=np.zeros((20, 20, 3), dtype=np.uint8),
        events={8: {"zone_entry": True, "in_prohibit": True, "crossed": True, "direction": "left"}},
        left_count=1, right_count=0,
    )
    event_ids = writer.save(event)
    assert len(event_ids) == 2
    result = writer.list_events()
    assert {item["event_type"] for item in result["items"]} == {"zone_entry", "crossed_left"}
    assert {item["timestamp"] for item in result["items"]} == {timestamp.isoformat()}
    assert all(writer.get(event_id)["timestamp"] == timestamp.isoformat() for event_id in event_ids)
    assert all("snapshot_base64" not in item for item in result["items"])


def test_save_rolls_back_all_events_from_frame_if_second_record_fails(event_database):
    _, writer = event_database
    writer.conn.execute("""
        CREATE TRIGGER reject_crossing BEFORE INSERT ON events
        WHEN NEW.event_type = 'crossed_left'
        BEGIN
            SELECT RAISE(ABORT, 'crossing rejected');
        END
    """)
    event = EventResults(
        camera_id="cam_01", timestamp=datetime.now(), frame=np.zeros((20, 20, 3), dtype=np.uint8),
        events={7: {"zone_entry": True, "in_prohibit": True, "crossed": True, "direction": "left"}},
        left_count=1, right_count=0,
    )
    with pytest.raises(sqlite3.IntegrityError, match="crossing rejected"):
        writer.save(event)
    assert writer.list_events()["total"] == 0
    assert not writer.conn.in_transaction
    event_id = writer.save_one_record(datetime.now(), "cam_01", 8, "zone_entry", "image")
    assert writer.get(event_id)["track_id"] == 8


def test_reads_keep_same_connection_open_for_later_writes(event_database):
    _, writer = event_database
    connection = writer.conn
    first = writer.save_one_record(datetime.now(), "cam_01", 1, "zone_entry", "one")
    assert writer.get(first)["track_id"] == 1
    assert writer.get(first + 1) is None
    assert writer.list_events()["total"] == 1
    assert writer.list_events(camera_id="missing")["total"] == 0
    assert writer.conn is connection
    assert not connection.in_transaction
    second = writer.save_one_record(datetime.now(), "cam_01", 2, "zone_entry", "two")
    assert writer.get(second)["track_id"] == 2
    assert writer.list_events()["total"] == 2


def test_total_and_page_share_snapshot_and_next_query_sees_new_commit(event_database):
    path, writer = event_database
    first = writer.save_one_record("2026-10-17T12:00:00", "cam_01", 1, "zone_entry", "one")
    inserted = False

    def insert_between_count_and_page(statement):
        nonlocal inserted
        if not inserted and statement.startswith("SELECT id, timestamp"):
            inserted = True
            with closing(sqlite3.connect(path)) as other:
                other.execute(
                    "INSERT INTO events(timestamp,camera_id,track_id,event_type,snapshot_base64) "
                    "VALUES (?,?,?,?,?)", ("2026-10-17T12:00:01", "cam_01", 2, "zone_entry", "two"),
                )
                other.commit()

    writer.conn.set_trace_callback(insert_between_count_and_page)
    try:
        result = writer.list_events()
    finally:
        writer.conn.set_trace_callback(None)
    assert inserted
    assert result["total"] == 1
    assert [item["id"] for item in result["items"]] == [first]
    assert not writer.conn.in_transaction
    assert writer.list_events()["total"] == 2


def test_camera_and_time_bounds_are_inclusive_to_microsecond(event_database):
    path, writer = event_database
    start = datetime(2026, 10, 17, 12, 0, 0, 123456)
    end = start + timedelta(microseconds=2)
    writer.save_one_record(start - timedelta(microseconds=1), "cam_01", 1, "zone_entry", "")
    first = writer.save_one_record(start, "cam_01", 2, "zone_entry", "")
    writer.save_one_record(start + timedelta(microseconds=1), "cam_02", 3, "zone_entry", "")
    last = writer.save_one_record(end, "cam_01", 4, "zone_entry", "")
    writer.save_one_record(end + timedelta(microseconds=1), "cam_01", 5, "zone_entry", "")
    result = writer.list_events(camera_id="cam_01", start_time=start, end_time=end)
    assert result["total"] == 2
    assert [item["id"] for item in result["items"]] == [last, first]
    assert [item["timestamp"] for item in result["items"]] == [end.isoformat(), start.isoformat()]


def test_zero_microsecond_bounds_do_not_include_later_frames(event_database):
    path, writer = event_database
    timestamp = datetime(2026, 10, 17, 12)
    first = writer.save_one_record(timestamp, "cam_01", 1, "zone_entry", "")
    writer.save_one_record(timestamp + timedelta(microseconds=1), "cam_01", 2, "zone_entry", "")
    result = writer.list_events(start_time=timestamp, end_time=timestamp)
    assert result["total"] == 1
    assert result["items"][0]["id"] == first


@pytest.mark.parametrize("bound", ["start_time", "end_time"])
def test_each_time_bound_can_be_used_alone(event_database, bound):
    path, writer = event_database
    timestamp = datetime(2026, 10, 17, 12)
    ids = [writer.save_one_record(timestamp + timedelta(seconds=i), "cam_01", i, "zone_entry", "")
           for i in range(3)]
    result = writer.list_events(**{bound: timestamp + timedelta(seconds=1)})
    expected = ids[1:] if bound == "start_time" else ids[:2]
    assert [item["id"] for item in result["items"]] == list(reversed(expected))


def test_equal_timestamps_have_stable_pagination_and_total_is_not_page_size(event_database):
    path, writer = event_database
    ids = [writer.save_one_record("2026-10-17T12:00:00", "cam_01", i, "zone_entry", "")
           for i in range(7)]
    reader = writer
    result = reader.list_events(limit=3, offset=2)
    assert result["total"] == 7
    assert (result["limit"], result["offset"]) == (3, 2)
    assert [item["id"] for item in result["items"]] == list(reversed(ids))[2:5]
    assert reader.list_events(offset=100)["items"] == []


def test_local_time_order_uses_timestamp_before_insertion_id(event_database):
    path, writer = event_database
    newer = writer.save_one_record("2026-10-17T12:30:00", "cam_01", 2, "zone_entry", "")
    older = writer.save_one_record("2026-10-17T12:00:00", "cam_01", 1, "zone_entry", "")
    assert newer < older
    assert [item["id"] for item in writer.list_events()["items"]] == [newer, older]


def test_camera_filter_is_parameterized(event_database):
    path, writer = event_database
    writer.save_one_record("2026-10-17T12:00:00", "cam_01", 1, "zone_entry", "")
    reader = writer
    assert reader.list_events(camera_id="cam_01' OR 1=1 --")["total"] == 0
    assert reader.list_events()["total"] == 1


@pytest.mark.parametrize("arguments", [
    {"limit": 0}, {"limit": 501}, {"limit": True}, {"limit": 1.5},
    {"offset": -1}, {"offset": True}, {"offset": 0.5},
    {"camera_id": " "}, {"camera_id": 7},
    {"start_time": datetime(2026, 10, 18), "end_time": datetime(2026, 10, 17)},
    {"start_time": "2026-10-17T12:00:00"},
    {"end_time": "2026-10-17T12:00:00"},
    {"start_time": "not a date"}, {"end_time": 123},
    {"start_time": datetime(2026, 10, 17, tzinfo=timezone.utc)},
    {"end_time": datetime(2026, 10, 17, tzinfo=timezone(timedelta(hours=7)))},
])
def test_invalid_query_arguments_raise_without_leaving_transaction(event_database, arguments):
    _, writer = event_database
    with pytest.raises(ValueError):
        writer.list_events(**arguments)
    assert not writer.conn.in_transaction


def test_original_schema_keeps_six_columns_and_all_snapshots(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    rows = [("2026-10-17T12:00:00.123456", "cam_01", i, "zone_entry", f"image-{i}")
            for i in range(1203)]
    create_legacy_database(path, rows)
    reader = EventWriter(path)
    assert reader.list_events()["total"] == 1203
    reader.close()
    with closing(sqlite3.connect(path)) as connection:
        assert [row[1] for row in connection.execute("PRAGMA table_info(events)")] == EVENT_COLUMNS
        assert connection.execute(
            "SELECT timestamp,camera_id,track_id,event_type,snapshot_base64 FROM events ORDER BY id"
        ).fetchall() == rows


def test_reading_keeps_unparsed_original_timestamps(tmp_path):
    path = tmp_path / "unparsed.sqlite3"
    create_legacy_database(path, [("legacy date", "cam_01", 1, "zone_entry", "original")])
    reader = EventWriter(path)
    assert reader.list_events()["items"][0]["timestamp"] == "legacy date"
    reader.close()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT snapshot_base64 FROM events").fetchone()[0] == "original"


def test_filtering_and_pagination_work_without_creating_indexes(event_database):
    path, writer = event_database
    timestamp = datetime(2026, 10, 17, 12)
    older = writer.save_one_record(timestamp, "cam_01", 1, "zone_entry", "")
    first = writer.save_one_record(timestamp + timedelta(seconds=1), "cam_01", 2, "zone_entry", "")
    writer.save_one_record(timestamp + timedelta(seconds=1), "cam_02", 3, "zone_entry", "")
    second = writer.save_one_record(timestamp + timedelta(seconds=1), "cam_01", 4, "zone_entry", "")
    newest = writer.save_one_record(timestamp + timedelta(seconds=2), "cam_01", 5, "zone_entry", "")

    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA index_list(events)").fetchall() == []


    reader = writer
    for _ in range(2):
        result = reader.list_events(
            camera_id="cam_01", start_time=timestamp,
            end_time=timestamp + timedelta(seconds=1), limit=1, offset=1,
        )
        assert result["total"] == 3
        assert [item["id"] for item in result["items"]] == [first]
        assert [item["id"] for item in reader.list_events(camera_id="cam_01")["items"]] == [
            newest, second, first, older,
        ]
        assert reader.list_events(start_time=timestamp + timedelta(seconds=1))["total"] == 4

    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA index_list(events)").fetchall() == []

def test_submit_keeps_async_writer_separate_from_api_connections(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    barrier = Barrier(4)
    worker = WriterWorker(path)

    def submit_events():
        for batch in range(20):
            for index in range(4):
                track_id = batch * 4 + index
                worker.submit(EventResults(
                    camera_id="cam_01", timestamp=datetime(2026, 10, 17, 12),
                    frame=np.zeros((20, 20, 3), dtype=np.uint8),
                    events={track_id: {"zone_entry": True, "in_prohibit": True, "crossed": False}},
                    left_count=0, right_count=0,
                ))
            barrier.wait(timeout=10)

    def read_events():
        # Mỗi thread tự mở, dùng và đóng connection của chính nó.
        with closing(EventWriter(path)) as reader:
            previous_total = 0
            for _ in range(20):
                barrier.wait(timeout=10)
                result = reader.list_events(camera_id="cam_01", limit=20)
                assert result["total"] >= previous_total
                assert len(result["items"]) == min(result["total"], 20)
                ids = [item["id"] for item in result["items"]]
                assert ids == sorted(set(ids), reverse=True)
                if ids:
                    record = reader.get(ids[0])
                    assert record["camera_id"] == "cam_01"
                    assert record["snapshot_base64"]
                previous_total = result["total"]

    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(submit_events), *(executor.submit(read_events) for _ in range(3))]
            for future in futures:
                future.result(timeout=30)
    finally:
        worker.close()
    with closing(EventWriter(path)) as reader:
        result = reader.list_events()
    assert result["total"] == 80
    assert {record["track_id"] for record in result["items"]} == set(range(80))
    assert not worker.thread.is_alive()
    assert worker.queue.empty()
