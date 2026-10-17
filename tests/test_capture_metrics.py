import logging
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src import camera_worker
from src.camera_worker import CameraWorker
from src.metrics import RollingRate
from src.schemas import CameraConfig


FRAME = np.zeros((8, 8, 3), dtype=np.uint8)
STOP = object()


class Capture:
    def __init__(self, reads=(), *, opened=True, fps=25.0):
        self.reads = iter(reads)
        self.opened = opened
        self.fps = fps
        self.released = False
        self.harness = None

    def isOpened(self):
        return self.opened

    def get(self, prop):
        assert prop == cv2.CAP_PROP_FPS
        return self.fps

    def read(self):
        result = next(self.reads)
        if result is STOP:
            self.harness.active = self.harness.worker.get_metrics()
            self.harness.worker.stop()
            return False, None
        self.harness.clock.now += 0.25
        return result is not None, result

    def release(self):
        self.released = True


@pytest.fixture
def capture_harness(monkeypatch):
    def build(sessions, *, source="rtsp://camera/live"):
        harness = SimpleNamespace(clock=SimpleNamespace(now=100.0), active=None, waits=[])
        clock = lambda: harness.clock.now
        monkeypatch.setattr(camera_worker, "time", SimpleNamespace(
            perf_counter=clock, monotonic=clock,
        ))
        harness.worker = CameraWorker(CameraConfig("cam_01", source))
        harness.worker._capture_rate = RollingRate(clock=clock)
        pending = iter(sessions)
        for session in sessions:
            session.harness = harness

        def open_capture(*args):
            return next(pending)

        def wait(seconds):
            harness.waits.append(harness.worker.get_metrics())
            harness.clock.now += seconds
            return False

        monkeypatch.setattr(camera_worker.cv2, "VideoCapture", open_capture)
        monkeypatch.setattr(harness.worker.stop_event, "wait", wait)
        harness.worker.running = True
        return harness

    return build


def test_measured_capture_fps_and_stop_reset(capture_harness):
    session = Capture([FRAME, FRAME, FRAME, STOP])
    harness = capture_harness([session])
    harness.worker._update_frame()

    assert harness.active["capture_fps"] == pytest.approx(4.0)
    assert harness.active["captured_frames"] == 3
    assert harness.active["connection_attempts"] == 1
    assert harness.active["reconnect_attempts"] == 0
    assert harness.active["reconnect_count"] == 0
    assert harness.worker.get_metrics() == {**harness.active, "capture_fps": 0.0}
    assert harness.worker.get_status() == "stopped"
    assert session.released
    assert harness.worker.error is None


def test_initial_open_failures_do_not_count_as_reconnect(capture_harness):
    sessions = [Capture(opened=False), Capture(opened=False), Capture([FRAME, FRAME, STOP])]
    harness = capture_harness(sessions)
    harness.worker._update_frame()

    assert harness.active["connection_attempts"] == 3
    assert harness.active["reconnect_attempts"] == 0
    assert harness.active["reconnect_count"] == 0
    assert harness.active["captured_frames"] == 2
    assert [item["capture_fps"] for item in harness.waits] == [0.0, 0.0]
    assert all(session.released for session in sessions)
    assert harness.worker.error is None


def test_reconnect_attempts_require_frames_before_and_after_failure(capture_harness, caplog):
    sessions = [
        Capture([FRAME, FRAME, None]),
        Capture(opened=False),
        Capture([None]),  # An opened connection without frames is not a recovery.
        Capture([FRAME, FRAME, STOP]),
    ]
    harness = capture_harness(sessions)
    with caplog.at_level(logging.INFO, logger="src.camera_worker"):
        harness.worker._update_frame()

    assert harness.active["connection_attempts"] == 4
    assert harness.active["reconnect_attempts"] == 3
    assert harness.active["reconnect_count"] == 1
    assert harness.active["captured_frames"] == 4
    assert harness.active["capture_fps"] == pytest.approx(4.0)
    assert all(item["capture_fps"] == 0 for item in harness.waits)
    assert all(item["reconnect_count"] == 0 for item in harness.waits)
    recovery = [record for record in caplog.records if "connection recovered" in record.message]
    assert len(recovery) == 1
    assert recovery[0].camera_id == "cam_01"
    assert "reconnect_attempts=3" in recovery[0].message
    assert "reconnect_count=1" in recovery[0].message
    assert "connection_attempts=4" in recovery[0].message
    assert all(session.released for session in sessions)
    assert harness.worker.error is None


def test_opened_initial_sessions_without_frames_are_not_reconnects(capture_harness):
    sessions = [Capture([None]), Capture([None]), Capture([FRAME, FRAME, STOP])]
    harness = capture_harness(sessions)
    harness.worker._update_frame()

    assert harness.active["connection_attempts"] == 3
    assert harness.active["reconnect_attempts"] == 0
    assert harness.active["reconnect_count"] == 0
    assert harness.worker.error is None


def test_each_recovered_session_is_counted_once(capture_harness):
    sessions = [
        Capture([FRAME, FRAME, None]),
        Capture([FRAME, FRAME, None]),
        Capture([FRAME, FRAME, FRAME, STOP]),
    ]
    harness = capture_harness(sessions)
    harness.worker._update_frame()

    assert harness.active["connection_attempts"] == 3
    assert harness.active["reconnect_attempts"] == 2
    assert harness.active["reconnect_count"] == 2
    assert harness.active["captured_frames"] == 7
    assert harness.worker.error is None


def test_file_pacing_uses_source_fps_not_measured_capture_fps(capture_harness):
    session = Capture([FRAME, FRAME, STOP], fps=17.0)
    harness = capture_harness([session], source="video.mp4")
    harness.worker._update_frame()

    assert harness.worker.fps == 17.0
    assert harness.active["capture_fps"] > 0
    assert harness.active["capture_fps"] != harness.worker.fps
    assert harness.worker.error is None


def test_disconnect_warning_is_rate_limited_and_has_attempt_counts(capture_harness, caplog):
    harness = capture_harness([])
    worker = harness.worker
    with caplog.at_level(logging.WARNING, logger="src.camera_worker"):
        for offset in (0.0, 1.0, 31.0):
            harness.clock.now = 100.0 + offset
            worker._set_status("connecting")
            worker._begin_connection_attempt()
            worker._set_status("disconnected")

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "connection_attempts=1" in warnings[0].message
    assert "connection_attempts=3" in warnings[1].message
    assert "reconnect_attempts=0" in warnings[1].message
