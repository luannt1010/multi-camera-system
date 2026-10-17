"""Optional per-camera geometry must not interrupt the rest of the pipeline."""
import sqlite3
import time

import cv2
import numpy as np
import pytest
import yaml

from src.config import load_config
from src.event_engine import EventEngine
from src.event_writer import EventWriter
from src.manager import SystemManager
from src.schemas import DetectionPacket, Event, FramePacket, TrackingPacket
from src.utils import draw_results


LINE = [[.1, .5], [.9, .5]]
POLYGON = [[.1, .3], [.9, .3], [.9, .9], [.1, .9]]
MODES = (
    ("both", LINE, POLYGON),
    ("line", LINE, None),
    ("polygon", None, POLYGON),
    ("bare", None, None),
)


def camera(camera_id, line=None, polygon=None):
    result = {"id": camera_id, "source": {"url": 0}}
    if line is not None:
        result["lines"] = {"point": line}
    if polygon is not None:
        result["prohibit_area"] = {"polygon": polygon}
    return result


def write_configs(tmp_path, cameras):
    # Validation only needs the file to exist; fake inference never reads weights.
    weights = tmp_path / "unused.pt"
    weights.touch()
    camera_path = tmp_path / "cameras.yaml"
    model_path = tmp_path / "model.yaml"
    camera_path.write_text(yaml.safe_dump({"cameras": cameras}), encoding="utf-8")
    model_path.write_text(yaml.safe_dump({
        "model": {"weights": str(weights), "device": "cpu"},
        "runtime": {"database": str(tmp_path / "events.sqlite3"), "batch_size": 2},
    }), encoding="utf-8")
    return camera_path, model_path


def tracked(y, *, camera_id="test", timestamp=1):
    return TrackingPacket(
        camera_id, timestamp, np.zeros((100, 100, 3), dtype=np.uint8),
        np.array([[45, y - 5, 55, y + 5]], dtype=np.float32),
        np.array([.9]), np.array([0]), np.array([1]),
    )


def test_config_accepts_all_four_camera_modes_together(tmp_path):
    paths = write_configs(tmp_path, [camera(*mode) for mode in MODES])
    cameras, _, _ = load_config(*paths)
    assert [item.event.enabled for item in cameras] == [True, True, True, False]
    for setting, (_, line, polygon) in zip(cameras, MODES):
        assert setting.event.line == line
        assert setting.event.polygon == polygon


@pytest.mark.parametrize("fields", [
    {},
    {"lines": None, "prohibit_area": None},
    {"lines": {}, "prohibit_area": {}},
    {"lines": {"point": None}, "prohibit_area": {"polygon": None}},
    {"line": None},
    {"line": {"point": None}},
])
def test_missing_or_null_geometry_disables_events(tmp_path, fields):
    item = camera("bare")
    item.update(fields)
    cameras, _, _ = load_config(*write_configs(tmp_path, [item]))
    assert not cameras[0].event.enabled
    assert cameras[0].event.line is None
    assert cameras[0].event.polygon is None


def test_config_accepts_line_alias_and_pixel_polygon(tmp_path):
    line = camera("line")
    line["line"] = {"point": [[10, 50], [90, 50]]}
    line["coordinates"] = "pixels"
    polygon = camera("polygon", polygon=[[10, 30], [90, 30], [90, 90], [10, 90]])
    polygon["coordinates"] = "pixels"
    cameras, _, _ = load_config(*write_configs(tmp_path, [line, polygon]))
    for item in cameras:
        assert item.event.enabled
        assert not item.event.normalized
    assert cameras[0].event.line == [[10, 50], [90, 50]]
    assert cameras[1].event.polygon == polygon["prohibit_area"]["polygon"]


@pytest.mark.parametrize("kind,points", [
    ("line", []),
    ("line", [[.1, .5]]),
    ("line", [[.1, .5], [.9]]),
    ("line", [[.1, .5], [.1, .5]]),
    ("line", [[.1, .5], [.9, float("nan")]]),
    ("line", [[.1, .5], [1.1, .5]]),
    ("polygon", []),
    ("polygon", [[.1, .3], [.9, .3]]),
    ("polygon", [[.1, .3], [.9, .3], [.9]]),
    ("polygon", [[.1, .1], [.2, .2], [.3, .3]]),
    ("polygon", [[.1, .1], [.1, .1], [.9, .9]]),
    ("polygon", [[.1, .3], [.9, .3], [.9, float("inf")]]),
    ("polygon", [[.1, .3], [.9, .3], [.9, -0.1]]),
])
def test_present_geometry_is_validated_without_the_other_kind(tmp_path, kind, points):
    item = camera("invalid")
    if kind == "line":
        item["lines"] = {"point": points}
    else:
        item["prohibit_area"] = {"polygon": points}
    with pytest.raises(ValueError):
        load_config(*write_configs(tmp_path, [item]))


@pytest.mark.parametrize("fields", [
    {"lines": []},
    {"lines": 1},
    {"prohibit_area": []},
    {"prohibit_area": "invalid"},
])
def test_malformed_geometry_blocks_raise_configuration_errors(tmp_path, fields):
    item = camera("invalid")
    item.update(fields)
    with pytest.raises(ValueError):
        load_config(*write_configs(tmp_path, [item]))


def test_line_only_crossings_do_not_generate_zone_events(tmp_path):
    engine = EventEngine(Event(line=LINE, normalized=True))
    writer = EventWriter(tmp_path / "line.sqlite3")
    try:
        first = engine.process(tracked(40))
        second = engine.process(tracked(60, timestamp=2))
        assert not first.events[1]["crossed"]
        assert second.right_count == 1
        assert second.events[1]["direction"] == "left_to_right"
        for result in (first, second):
            assert not result.events[1]["in_prohibit"]
            assert not result.events[1]["zone_entry"]
            writer.save(result)
        rows = writer.conn.execute("select event_type from events").fetchall()
        assert rows == [("crossed_left_to_right",)]
        line, polygon = engine.geometry((100, 200, 3))
        np.testing.assert_allclose(line, [[20, 50], [180, 50]])
        assert polygon is None
    finally:
        writer.close()


def test_polygon_only_entry_exit_reentry_has_no_crossings(tmp_path):
    engine = EventEngine(Event(polygon=POLYGON, normalized=True))
    writer = EventWriter(tmp_path / "polygon.sqlite3")
    try:
        results = [engine.process(tracked(y, timestamp=i))
                   for i, y in enumerate((10, 40, 60, 10, 40), start=1)]
        assert [item.events[1]["zone_entry"] for item in results] == [False, True, False, False, True]
        for result in results:
            assert not result.events[1]["crossed"]
            assert result.events[1]["direction"] is None
            assert result.left_count == result.right_count == 0
            writer.save(result)
        rows = writer.conn.execute("select event_type from events").fetchall()
        assert rows == [("zone_entry",), ("zone_entry",)]
        line, polygon = engine.geometry((100, 200, 3))
        assert line is None
        np.testing.assert_allclose(polygon, [[20, 30], [180, 30], [180, 90], [20, 90]])
    finally:
        writer.close()


def test_disabled_event_engine_does_not_accumulate_identity_state():
    config = Event()
    assert not config.enabled
    engine = EventEngine(config)
    assert engine.geometry((100, 100, 3)) == (None, None)
    for i, y in enumerate((40, 60), start=1):
        result = engine.process(tracked(y, timestamp=i))
        assert result.events == {}
        assert result.left_count == result.right_count == 0
    assert not engine.last_seen
    assert not engine.previous_side
    assert not engine.previous_center
    assert not engine.inside


@pytest.mark.parametrize("mode,line,polygon", MODES)
def test_drawing_keeps_boxes_and_only_shows_configured_geometry(monkeypatch, mode, line, polygon):
    packet = tracked(40)
    engine = EventEngine(Event(line, polygon, normalized=True))
    result = engine.process(packet) if engine.config.enabled else None
    labels = []
    original_text = cv2.putText

    def record_text(frame, text, *args, **kwargs):
        labels.append(text)
        return original_text(frame, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)
    drawn = draw_results(packet, result, engine.geometry(packet.frame.shape))
    assert drawn.any()
    assert drawn[35, 45].any(), "Bounding boxes must remain visible for bare cameras"
    assert not packet.frame.any(), "Drawing must preserve the raw frame used for snapshots"
    if result is not None:
        assert not result.frame.any()
    assert any("ID 1" in label for label in labels)
    counters = any("Left" in label or "Right" in label for label in labels)
    assert counters == (line is not None)


class SequenceCamera:
    def __init__(self, config):
        self.config = config
        self.thread = None
        self.running = False
        self.positions = iter((40, 60, 70, 10, 40))
        self.generation = 0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def get_status(self):
        return "connected" if self.running else "stopped"

    def get_frame(self):
        y = next(self.positions, None)
        if y is None:
            return None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # One pixel communicates the fake object's position to fake inference.
        frame[0, 0, 0] = y
        return FramePacket(self.config.camera_id, time.perf_counter(), frame,
                           generation=self.generation)


class FakeInference:
    def __init__(self, config):
        self.camera_ids = []

    def predict(self, packets):
        self.camera_ids.extend(packet.camera_id for packet in packets)
        results = []
        for packet in packets:
            y = int(packet.frame[0, 0, 0])
            results.append(DetectionPacket(
                packet.camera_id, packet.timestamp, packet.frame,
                np.array([[45, y - 5, 55, y + 5]], dtype=np.float32),
                np.array([.9]), np.array([0]), generation=packet.generation,
            ))
        return results


class FakeTracker:
    def __init__(self, fps):
        self.camera_ids = []
        self.resets = []

    def update(self, packet):
        self.camera_ids.append(packet.camera_id)
        return TrackingPacket(
            packet.camera_id, packet.timestamp, packet.frame, packet.boxes,
            packet.scores, packet.class_ids, np.array([1]),
            generation=packet.generation,
        )

    def reset(self, camera_id):
        self.resets.append(camera_id)


class FakeStream:
    def __init__(self, **options):
        self.frames = []
        self.dropped = 0
        self.closed = False

    def start(self):
        pass

    def check(self):
        pass

    def submit(self, frame, *, captured_at=None):
        self.frames.append(frame)

    def close(self):
        self.closed = True


def make_manager(paths, **options):
    return SystemManager(*paths, inference_factory=FakeInference, camera_factory=SequenceCamera,
                         tracker_factory=FakeTracker, stream_factory=FakeStream, **options)


def test_manager_mixed_modes_only_persists_configured_events(tmp_path):
    cameras = [camera(*mode) for mode in MODES]
    for item in cameras:
        item["output"] = {"rtsp_url": f"rtsp://localhost:8554/output/{item['id']}"}
    manager = make_manager(write_configs(tmp_path, cameras))
    manager.run(duration=.1)
    assert set(manager.engines) == {"both", "line", "polygon"}
    assert manager.processed == 20
    assert set(manager.inference.camera_ids) == {"both", "line", "polygon", "bare"}
    assert set(manager.tracker.camera_ids) == {"both", "line", "polygon", "bare"}
    for stream in manager.streamers.values():
        assert len(stream.frames) == 5
        assert stream.frames[0][35, 45].any()
        assert stream.closed
    with sqlite3.connect(tmp_path / "events.sqlite3") as db:
        rows = db.execute("select camera_id, event_type from events").fetchall()
    assert len(rows) == 8
    assert sorted(event for cid, event in rows if cid == "line") == [
        "crossed_left_to_right", "crossed_right_to_left",
    ]
    assert [event for cid, event in rows if cid == "polygon"] == ["zone_entry", "zone_entry"]
    assert not any(cid == "bare" for cid, _ in rows)
    assert not manager.writer.thread.is_alive()
    assert all(not worker.running for worker in manager.workers.values())


def test_manager_without_any_geometry_never_initializes_writer(tmp_path):
    def forbidden_writer(*args, **kwargs):
        pytest.fail("No event writer should be initialized when all geometry is disabled")

    manager = make_manager(write_configs(tmp_path, [camera("bare")]),
                           writer_factory=forbidden_writer, stream=False)
    manager.run(duration=.05)
    assert manager.processed == 5
    assert manager.engines == {}
    assert manager.writer is None
    assert manager.status()["event_queue"] == 0
    assert not (tmp_path / "events.sqlite3").exists()


def test_reconnect_without_event_geometry_still_resets_tracker(tmp_path):
    manager = make_manager(write_configs(tmp_path, [camera("bare")]), stream=False)
    try:
        manager._initialize()
        assert len(manager._collect()) == 1
        manager.workers["bare"].generation = 1
        assert len(manager._collect()) == 1
        assert manager.tracker.resets == ["bare"]
        assert manager.engines == {}
    finally:
        manager._close()
