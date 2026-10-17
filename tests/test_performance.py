"""Performance counters measure completed frames and preserve raw event images."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.metrics import CameraPerformance, RollingLatency, RollingRate
from src.schemas import CameraConfig, DetectionPacket, Event, FramePacket, TrackingPacket
from src.utils import draw_results


class Clock:
    def __init__(self, start=100.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


def test_fps_uses_frame_intervals_and_expires_when_frames_stop():
    clock = Clock(start=0)
    rate = RollingRate(window_seconds=5, clock=clock)
    assert rate.value() == 0
    rate.mark()
    assert rate.value() == 0, "One frame cannot establish an observed frame rate"
    clock.advance(1)
    rate.mark()
    clock.advance(1)
    rate.mark()
    assert rate.value() == pytest.approx(1)
    clock.advance(6)
    assert rate.value() == 0, "A disconnected camera must not retain its old FPS"


def test_fps_reset_starts_new_connection_without_a_stale_interval():
    clock = Clock(start=0)
    rate = RollingRate(clock=clock)
    rate.mark(now=0)
    rate.mark(now=.1)
    assert rate.value(now=.1) == pytest.approx(10)
    rate.reset()
    assert rate.value(now=.1) == 0
    rate.mark(now=4)
    assert rate.value(now=4) == 0
    rate.mark(now=5)
    assert rate.value(now=5) == pytest.approx(1)


def test_latency_reports_recent_last_mean_and_p95_then_expires():
    clock = Clock(start=0)
    latency = RollingLatency(window_seconds=5, clock=clock)
    empty = {"last_ms": None, "avg_ms": None, "p95_ms": None}
    assert latency.snapshot() == empty
    for milliseconds in range(1, 21):
        latency.record(milliseconds)
    snapshot = latency.snapshot()
    assert snapshot["last_ms"] == 20
    assert snapshot["avg_ms"] == pytest.approx(10.5)
    assert 19 <= snapshot["p95_ms"] <= 20
    clock.advance(6)
    assert latency.snapshot() == empty
    latency.record(7)
    assert latency.snapshot() == {"last_ms": 7, "avg_ms": 7, "p95_ms": 7}


def test_latency_window_excludes_expired_slow_samples():
    clock = Clock(start=0)
    latency = RollingLatency(window_seconds=5, clock=clock)
    latency.record(1000)
    clock.advance(4)
    latency.record(10)
    clock.advance(2)
    assert latency.snapshot() == {"last_ms": 10, "avg_ms": 10, "p95_ms": 10}


def test_performance_history_is_bounded_with_concurrent_writers():
    clock = Clock()
    rate = RollingRate(clock=clock)
    latency = RollingLatency(clock=clock)

    def record(_):
        for index in range(3000):
            rate.mark()
            latency.record(12)
            if index % 100 == 0:
                assert rate.value() >= 0
                assert latency.snapshot()["last_ms"] == 12

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(4)))
    for metric in (rate, latency):
        histories = [value for value in vars(metric).values() if isinstance(value, deque)]
        assert histories, "A rolling history must retain a bounded sample buffer"
        assert all(len(history) <= 4096 for history in histories)
    assert latency.snapshot()["avg_ms"] == pytest.approx(12)


def test_camera_performance_measures_capture_age_batch_inference_and_completion():
    clock = Clock()
    metrics = CameraPerformance(clock=clock)
    metrics.record(captured_at=99.80, inference_started_at=99.90,
                   inference_finished_at=99.95, completed_at=100)
    first = metrics.snapshot(now=100)
    assert first["processed_frames"] == 1
    assert first["processing_fps"] == 0
    for name, milliseconds in (("frame_age_ms", 100), ("inference_ms", 50),
                               ("processing_latency_ms", 200)):
        assert first[name]["last_ms"] == pytest.approx(milliseconds)

    metrics.record(captured_at=100.80, inference_started_at=100.90,
                   inference_finished_at=100.95, completed_at=101)
    assert metrics.snapshot(now=101)["processing_fps"] == pytest.approx(1)
    assert metrics.snapshot(now=101)["processed_frames"] == 2
    stale = metrics.snapshot(now=107)
    assert stale["processing_fps"] == 0
    assert stale["processing_latency_ms"]["last_ms"] is None
    assert stale["processed_frames"] == 2, "Totals survive rolling-window expiry"


@pytest.mark.parametrize("shape", [(150, 400, 3), (1080, 1920, 3)])
def test_overlay_displays_per_camera_fps_without_mutating_the_source(monkeypatch, shape):
    packet = TrackingPacket("bare", 1, np.zeros(shape, np.uint8),
                            np.empty((0, 4), np.float32), np.empty(0),
                            np.empty(0, dtype=int), np.empty(0, dtype=int))
    labels = []
    fps_scales = []
    original = cv2.putText

    def put_text(frame, text, *args, **kwargs):
        labels.append(text)
        if "FPS" in text:
            fps_scales.append(args[2])
        return original(frame, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", put_text)
    drawn = draw_results(packet, None, (None, None),
                         performance={"processing_fps": 12.5})
    assert any("FPS" in text and "12.5" in text for text in labels)
    assert drawn.any() and not packet.frame.any()
    assert drawn is not packet.frame
    preview_scale = min(800 / shape[1], 450 / shape[0], 1)
    assert all(scale * preview_scale >= .59 for scale in fps_scales)
    # Existing callers remain valid when they have no performance metadata.
    assert draw_results(packet, None, (None, None)).shape == packet.frame.shape


def deterministic_manager(monkeypatch, *, interval=2):
    """Two cameras run at different completion rates, with shared batch inference."""
    import src.manager as module

    clock = Clock()
    monkeypatch.setattr(module, "time", SimpleNamespace(
        perf_counter=clock, monotonic=clock))
    settings = [SimpleNamespace(capture=CameraConfig(cid, 0), event=Event(),
                                output={"output_url": f"rtsp://localhost:8554/out/{cid}"})
                for cid in ("fast", "slow")]
    runtime = {"batch_size": 2, "tracking_fps": 15, "reset_after_seconds": 3}
    monkeypatch.setattr(module, "load_config", lambda *_: (settings, None, runtime))

    class Camera:
        def __init__(self, config):
            self.config = config
            self.thread = None
            self.running = False
            self.requests = 0

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

        def get_status(self):
            return "connected" if self.running else "stopped"

        def get_frame(self):
            request = self.requests
            self.requests += 1
            if request >= 4 or (self.config.camera_id == "slow" and request % 2):
                return None
            return FramePacket(self.config.camera_id, clock() - .05,
                               np.zeros((150, 400, 3), np.uint8))

        def get_metrics(self):
            return {"capture_fps": 24 if self.config.camera_id == "fast" else 12,
                    "reconnect_count": 2 if self.config.camera_id == "fast" else 0}

    class Inference:
        def __init__(self, _):
            pass

        def predict(self, packets):
            clock.advance(.02)
            return [DetectionPacket(p.camera_id, p.timestamp, p.frame,
                                    np.empty((0, 4), np.float32), np.empty(0),
                                    np.empty(0, dtype=int)) for p in packets]

    class Tracker:
        def __init__(self, **_):
            pass

        def update(self, packet):
            clock.advance(.03)
            return TrackingPacket(packet.camera_id, packet.timestamp, packet.frame,
                                  packet.boxes, packet.scores, packet.class_ids,
                                  np.empty(0, dtype=int))

        def reset(self, _):
            pass

    class Stream:
        def __init__(self, camera_id, **_):
            self.camera_id = camera_id
            self.dropped = 0
            self.frames = []

        def start(self):
            pass

        def check(self):
            pass

        def submit(self, frame, **metadata):
            self.frames.append((frame, metadata))

        def close(self):
            pass

        def get_metrics(self):
            return {"output_fps": 10 if self.camera_id == "fast" else 5,
                    "e2e_latency_ms": {"last_ms": 150, "avg_ms": 140, "p95_ms": 180}}

    manager = module.SystemManager(inference_factory=Inference, camera_factory=Camera,
                                   tracker_factory=Tracker, stream_factory=Stream,
                                   status_interval_seconds=interval)
    collect = manager._collect

    def scheduled_collect():
        clock.advance(1)
        return collect()

    monkeypatch.setattr(manager, "_collect", scheduled_collect)
    return manager


def test_manager_rates_and_latency_are_per_camera_and_reach_overlay(monkeypatch):
    import src.manager as module

    manager = deterministic_manager(monkeypatch)
    overlays = []

    def draw(packet, result, geometry, performance=None):
        overlays.append((packet.camera_id, performance))
        return packet.frame.copy()

    monkeypatch.setattr(module, "draw_results", draw)
    manager.run(duration=4.5)
    summary = manager.status()["camera_metrics"]
    assert manager.processed == 6
    assert summary["fast"]["processed_frames"] == 4
    assert summary["slow"]["processed_frames"] == 2
    assert summary["fast"]["capture_fps"] == 24
    assert summary["slow"]["capture_fps"] == 12
    assert summary["fast"]["reconnect_count"] == 2
    assert summary["slow"]["reconnect_count"] == 0
    assert summary["fast"]["output_fps"] == 10
    assert summary["slow"]["output_fps"] == 5
    assert summary["fast"]["e2e_latency_ms"]["last_ms"] == 150
    assert 0 < summary["slow"]["processing_fps"] < summary["fast"]["processing_fps"] < 2
    for performance in summary.values():
        assert performance["frame_age_ms"]["avg_ms"] == pytest.approx(50)
        assert performance["inference_ms"]["avg_ms"] == pytest.approx(20)
        assert performance["processing_latency_ms"]["last_ms"] >= 100 - 1e-8
    assert len(overlays) == 6
    for _, performance in overlays:
        assert performance is not None and "processing_fps" in performance
    for cid in ("fast", "slow"):
        rates = [performance["processing_fps"] for camera_id, performance in overlays if camera_id == cid]
        assert rates[0] == 0 and rates[-1] > 0
    for stream in manager.streamers.values():
        assert stream.frames
        assert all(metadata for _, metadata in stream.frames), "RTSP needs capture timestamps for end-to-end delay"


def test_manager_logs_camera_performance_periodically_and_at_shutdown(monkeypatch, caplog):
    manager = deterministic_manager(monkeypatch, interval=2)
    with caplog.at_level(logging.INFO, logger="src.manager"):
        manager.run(duration=4.5)
    records = [record for record in caplog.records
               if record.name == "src.manager" and record.getMessage().startswith("Camera performance:")]
    assert {record.camera_id for record in records} == {"fast", "slow"}
    assert all(record.levelno == logging.INFO for record in records)
    for cid in ("fast", "slow"):
        assert sum(record.camera_id == cid for record in records) >= 2
    # Final metrics must still be emitted when no periodic deadline occurs.
    caplog.clear()
    manager = deterministic_manager(monkeypatch, interval=100)
    with caplog.at_level(logging.INFO, logger="src.manager"):
        manager.run(duration=4.5)
    final = [record for record in caplog.records
             if record.name == "src.manager" and record.getMessage().startswith("Camera performance:")]
    assert len(final) == 2
    assert {record.camera_id for record in final} == {"fast", "slow"}
