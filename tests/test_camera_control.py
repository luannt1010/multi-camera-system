"""Kiểm tra bật/tắt camera thật sự trong luồng manager, không cần RTSP hay GPU."""
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import CameraSettings
from src.schemas import CameraConfig, DetectionPacket, Event, FramePacket, TrackingPacket


def wait_for(predicate, timeout=3):
    """Chờ điều kiện với deadline để test không phụ thuộc tốc độ máy."""
    deadline = time.monotonic() + timeout
    tick = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return
        tick.wait(.005)
    assert predicate(), "Điều kiện không hoàn thành trước deadline"


class FakeCamera:
    def __init__(self, config, harness):
        self.config = config
        self.harness = harness
        self.frames = deque()
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.error = None
        self.stopping = threading.Event()
        self.starts = self.stops = 0
        self.metrics = {"capture_fps": 25.0, "captured_frames": 7,
                        "connection_attempts": 4, "reconnect_attempts": 3,
                        "reconnect_count": 3}

    def start(self):
        self.harness.lifecycle.append(("capture_start", self.config.camera_id, threading.get_ident()))
        self.starts += 1
        self.running = True
        self.stopping.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.stopping.wait()
        self._clear_queue()

    def stop(self):
        self.harness.lifecycle.append(("capture_stop", self.config.camera_id, threading.get_ident()))
        self.stops += 1
        self.running = False
        self.stopping.set()

    def _clear_queue(self):
        with self.lock:
            self.frames.clear()

    def get_frame(self):
        with self.lock:
            return self.frames.popleft() if self.frames else None

    def get_status(self):
        return "connected" if self.running else "stopped"

    def get_metrics(self):
        return dict(self.metrics)


class FakeStream:
    def __init__(self, camera_id, harness, **options):
        self.camera_id = camera_id
        self.harness = harness
        self.options = options
        self.closed = False
        self.starts = 0
        self.dropped = 0
        self.frames = []
        self.close_gate = None
        self.close_entered = threading.Event()
        self.close_error = None

    def start(self):
        self.starts += 1
        self.harness.lifecycle.append(("stream_start", self.camera_id, threading.get_ident()))

    def close(self):
        self.harness.lifecycle.append(("stream_close", self.camera_id, threading.get_ident()))
        self.close_entered.set()
        if self.close_gate is not None:
            assert self.close_gate.wait(5)
        self.closed = True
        if self.close_error is not None:
            raise self.close_error

    def check(self):
        pass

    def submit(self, frame, *, captured_at=None):
        assert not self.closed
        self.frames.append(captured_at)

    def get_metrics(self):
        return {"output_fps": 24.0, "sent_frames": len(self.frames),
                "dropped_frames": self.dropped}


class FakeModel:
    def __init__(self, harness):
        self.harness = harness
        self.calls = []

    def predict(self, packets):
        self.calls.append(list(packets))
        self.harness.inference_entered.set()
        assert self.harness.inference_release.wait(5)
        if self.harness.inference_error is not None:
            raise self.harness.inference_error
        return [DetectionPacket(
            item.camera_id, item.timestamp, item.frame,
            np.empty((0, 4), np.float32), np.empty(0), np.empty(0, dtype=int),
            generation=item.generation,
        ) for item in packets]


class FakeTracker:
    def __init__(self, harness):
        self.harness = harness
        self.received = []
        self.resets = []

    def update(self, detection):
        self.received.append(detection)
        return TrackingPacket(
            detection.camera_id, detection.timestamp, detection.frame,
            detection.boxes, detection.scores, detection.class_ids,
            np.empty(0, dtype=int), generation=detection.generation,
        )

    def reset(self, camera_id):
        self.harness.lifecycle.append(("tracker_reset", camera_id, threading.get_ident()))
        self.resets.append(camera_id)


class FakeWriter:
    def __init__(self):
        self.queue = Queue()
        self.closed = False
        self.results = []

    def start(self):
        pass

    def check(self):
        pass

    def submit(self, result):
        self.results.append(result)

    def close(self):
        self.closed = True


class Harness:
    def __init__(self):
        self.captures = defaultdict(list)
        self.streams = defaultdict(list)
        self.models = []
        self.lifecycle = []
        self.inference_entered = threading.Event()
        self.inference_release = threading.Event()
        self.inference_release.set()
        self.inference_error = None
        self.model_error = None
        self.model_gate = None
        self.model_entered = threading.Event()
        self.run_errors = []
        self.manager = self.thread = None
        self.writer = FakeWriter()

    def camera(self, config):
        result = FakeCamera(config, self)
        self.captures[config.camera_id].append(result)
        return result

    def stream(self, camera_id, **options):
        result = FakeStream(camera_id, self, **options)
        self.streams[camera_id].append(result)
        return result

    def model(self, config):
        self.model_entered.set()
        if self.model_gate is not None:
            assert self.model_gate.wait(5)
        if self.model_error is not None:
            raise self.model_error
        result = FakeModel(self)
        self.models.append(result)
        return result

    def start(self, *, wait=True):
        def run():
            try:
                self.manager.run()
            except Exception as exc:
                self.run_errors.append(exc)

        self.thread = threading.Thread(target=run, name="test-manager", daemon=True)
        self.thread.start()
        if wait:
            assert self.manager.ready_event.wait(3)
            assert not self.run_errors

    def emit(self, camera_id, *, generation=1, timestamp=None):
        worker = self.captures[camera_id][-1]
        packet = FramePacket(camera_id, time.perf_counter() if timestamp is None else timestamp,
                             np.zeros((8, 8, 3), np.uint8), generation=generation)
        with worker.lock:
            worker.frames.append(packet)
        return packet

    def seen(self, packet):
        return any(item.timestamp == packet.timestamp and item.camera_id == packet.camera_id
                   for item in self.manager.tracker.received)

    def close(self):
        self.inference_release.set()
        if self.model_gate is not None:
            self.model_gate.set()
        for sinks in self.streams.values():
            for sink in sinks:
                if sink.close_gate is not None:
                    sink.close_gate.set()
        self.manager.stop()
        if self.thread is not None:
            self.thread.join(5)
            assert not self.thread.is_alive(), "Luồng manager chưa dừng"
        for captures in self.captures.values():
            for camera in captures:
                if camera.thread is not None:
                    assert not camera.thread.is_alive(), "Luồng capture chưa dừng"


@pytest.fixture
def harness_factory(monkeypatch, tmp_path):
    import src.manager as module

    harnesses = []
    settings = [CameraSettings(CameraConfig(cid, "unused"),
                               Event(line=[[0, 4], [8, 4]]),
                               {"output_url": f"rtsp://unused/{cid}"}) for cid in ("a", "b")]
    runtime = {"batch_size": 2, "tracking_fps": 15, "event_queue_size": 8,
               "reset_after_seconds": 3, "database": tmp_path / "unused.db"}
    monkeypatch.setattr(module, "load_config", lambda *_: (settings, SimpleNamespace(), dict(runtime)))
    monkeypatch.setattr(module, "draw_results", lambda tracking, *_args, **_kwargs: tracking.frame)

    def make(*, mode="shared", **options):
        harness = Harness()
        harness.manager = module.SystemManager(
            model_mode=mode, inference_factory=harness.model,
            camera_factory=harness.camera, stream_factory=harness.stream,
            tracker_factory=lambda **_: FakeTracker(harness),
            writer_factory=lambda *_args, **_kwargs: harness.writer,
            **options,
        )
        harnesses.append(harness)
        return harness

    yield make
    for harness in harnesses:
        harness.close()


def test_controls_reject_requests_until_initialization_completes(harness_factory):
    harness = harness_factory()
    harness.model_gate = threading.Event()
    harness.start(wait=False)
    assert harness.model_entered.wait(3)
    assert not harness.manager.ready_event.is_set()
    with pytest.raises(RuntimeError):
        harness.manager.set_camera_enabled("a", False)
    harness.model_gate.set()
    assert harness.manager.ready_event.wait(3)
    assert harness.manager.error is None
    assert harness.manager.get_camera_status("a")["enabled"] is True


def test_unknown_camera_has_clear_error_for_reads_and_controls(harness_factory):
    harness = harness_factory()
    harness.start()
    with pytest.raises(KeyError):
        harness.manager.get_camera_status("missing")
    with pytest.raises(KeyError):
        harness.manager.set_camera_enabled("missing", False)


def test_single_camera_reads_and_control_responses_skip_unrelated_metrics(harness_factory, monkeypatch):
    harness = harness_factory()
    harness.start()
    manager = harness.manager

    def unrelated_read(*_args, **_kwargs):
        pytest.fail("Đọc một camera không được quét trạng thái và metrics camera khác")

    with monkeypatch.context() as patch:
        patch.setattr(manager.performance["b"], "snapshot", unrelated_read)
        patch.setattr(manager.workers["b"], "get_metrics", unrelated_read)
        patch.setattr(manager.workers["b"], "get_status", unrelated_read)
        patch.setattr(manager.streamers["b"], "get_metrics", unrelated_read)
        assert manager.get_camera_status("a")["camera_id"] == "a"
        assert manager.set_camera_enabled("a", False)["enabled"] is False
        assert manager.set_camera_enabled("a", True)["enabled"] is True


def test_initialization_failure_is_exposed_without_accepting_controls(harness_factory):
    harness = harness_factory()
    harness.model_error = RuntimeError("weights unavailable")
    harness.start(wait=False)
    harness.thread.join(3)
    assert not harness.thread.is_alive()
    assert harness.manager.ready_event.is_set()
    assert harness.manager.error is harness.model_error
    assert not harness.manager.running
    with pytest.raises(RuntimeError):
        harness.manager.set_camera_enabled("a", False)


@pytest.mark.parametrize("enabled", [None, 0, 1, "false", [], {}])
def test_enabled_requires_a_real_boolean(harness_factory, enabled):
    harness = harness_factory()
    harness.start()
    with pytest.raises(ValueError):
        harness.manager.set_camera_enabled("a", enabled)
    assert harness.captures["a"][0].starts == 1


@pytest.mark.parametrize("mode,model_count", [("shared", 1), ("per_camera", 2)])
def test_disable_enable_releases_resources_and_keeps_other_camera_running(harness_factory, mode, model_count):
    harness = harness_factory(mode=mode)
    harness.start()
    manager = harness.manager
    first = harness.emit("a", generation=9)
    other = harness.emit("b")
    wait_for(lambda: harness.seen(first) and harness.seen(other))
    capture = manager.workers["a"]
    previous_thread = capture.thread
    previous_stream = manager.streamers["a"]
    capture_metrics = capture.get_metrics()
    engine = manager.engines["a"]
    engine.left_count, engine.right_count = 4, 6
    engine.previous_side[10] = 2
    engine.previous_center[10] = (2, 2)
    engine.last_seen[10] = first.timestamp

    disabled = manager.set_camera_enabled("a", False)
    assert disabled["camera_id"] == "a"
    assert disabled["enabled"] is False
    assert disabled["status"] == "disabled"
    assert "a" not in manager.workers and "a" not in manager.streamers
    assert not previous_thread.is_alive()
    assert previous_stream.closed
    for field in ("processing_fps", "capture_fps", "output_fps"):
        assert disabled["metrics"][field] == 0
    for field in ("captured_frames", "connection_attempts", "reconnect_attempts", "reconnect_count"):
        assert disabled["metrics"][field] == capture_metrics[field]
    assert disabled["metrics"]["processed_frames"] >= 1
    assert not engine.previous_side and not engine.previous_center and not engine.last_seen
    assert (engine.left_count, engine.right_count) == (4, 6)

    other = harness.emit("b")
    wait_for(lambda: harness.seen(other))
    # Frame còn sót trong worker đã tắt phải bị bỏ trước khi bật lại.
    stale = harness.emit("a", generation=9)
    enabled = manager.set_camera_enabled("a", True)
    assert enabled["enabled"] is True
    assert len(harness.models) == model_count
    assert manager.workers["a"].thread is not previous_thread
    assert manager.streamers["a"] is not previous_stream
    assert "a" not in manager.last_frame and "a" not in manager.generations
    fresh = harness.emit("a", generation=1)
    wait_for(lambda: harness.seen(fresh))
    assert not harness.seen(stale)
    assert manager.get_camera_status("a")["metrics"]["reconnect_count"] == 3
    assert (engine.left_count, engine.right_count) == (4, 6)
    assert all(identity == harness.thread.ident for _, _, identity in harness.lifecycle)
    if mode == "per_camera":
        for cid, model in manager.inferences.items():
            assert all(item.camera_id == cid for batch in model.calls for item in batch)


def test_idempotent_controls_do_not_create_duplicate_threads(harness_factory):
    harness = harness_factory()
    harness.start()
    manager = harness.manager
    capture, stream = manager.workers["a"], manager.streamers["a"]
    for _ in range(2):
        assert manager.set_camera_enabled("a", True)["enabled"] is True
    assert capture.starts == stream.starts == 1
    for _ in range(2):
        assert manager.set_camera_enabled("a", False)["enabled"] is False
    assert capture.stops == 1
    for _ in range(2):
        assert manager.set_camera_enabled("a", True)["enabled"] is True
    assert capture.starts == 2
    assert len(harness.streams["a"]) == 2


def test_all_cameras_can_be_disabled_and_enabled_without_restarting_pipeline(harness_factory):
    harness = harness_factory()
    harness.start()
    manager = harness.manager
    for cid in ("a", "b"):
        manager.set_camera_enabled(cid, False)
    assert manager.running and not manager.workers
    assert {item["status"] for item in manager.list_cameras()} == {"disabled"}
    manager.set_camera_enabled("a", True)
    packet = harness.emit("a")
    wait_for(lambda: harness.seen(packet))
    assert manager.get_camera_status("b")["enabled"] is False


def test_status_reads_remain_available_while_camera_close_is_blocked(harness_factory):
    harness = harness_factory()
    harness.start()
    sink = harness.manager.streamers["a"]
    sink.close_gate = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as executor:
        command = executor.submit(harness.manager.set_camera_enabled, "a", False)
        try:
            assert sink.close_entered.wait(3)
            snapshot = executor.submit(harness.manager.list_cameras).result(timeout=1)
            assert {item["camera_id"] for item in snapshot} == {"a", "b"}
            assert not command.done()
        finally:
            sink.close_gate.set()
        assert command.result(timeout=3)["enabled"] is False


def test_concurrent_status_readers_can_run_during_repeated_controls(harness_factory):
    harness = harness_factory()
    harness.start()
    finished = threading.Event()
    reader_started = threading.Event()

    def read():
        reader_started.set()
        while not finished.is_set():
            cameras = harness.manager.list_cameras()
            assert {item["camera_id"] for item in cameras} == {"a", "b"}
            assert all(isinstance(item["enabled"], bool) and isinstance(item["metrics"], dict)
                       for item in cameras)
            assert set(harness.manager.status()["camera_metrics"]) == {"a", "b"}

    with ThreadPoolExecutor(max_workers=2) as executor:
        readers = [executor.submit(read) for _ in range(2)]
        assert reader_started.wait(3)
        try:
            for _ in range(4):
                harness.manager.set_camera_enabled("a", False)
                harness.manager.set_camera_enabled("a", True)
        finally:
            finished.set()
        for reader in readers:
            reader.result(timeout=3)


def test_failed_output_cleanup_keeps_camera_disabled_without_stopping_others(harness_factory):
    harness = harness_factory()
    harness.start()
    manager = harness.manager
    sink = manager.streamers["a"]
    sink.close_error = RuntimeError("encoder close failed")
    try:
        with pytest.raises(RuntimeError, match="encoder close failed"):
            manager.set_camera_enabled("a", False)
        status = manager.get_camera_status("a")
        assert status["enabled"] is False
        assert status["status"] == "disabled"
        assert "encoder close failed" in status["error"]
        other = harness.emit("b")
        wait_for(lambda: harness.seen(other))
        assert manager.running
        # Chưa dọn xong encoder cũ thì không được mở thêm encoder cho cùng camera.
        with pytest.raises(RuntimeError):
            manager.set_camera_enabled("a", True)
        assert len(harness.streams["a"]) == 1
    finally:
        sink.close_error = None
    manager.set_camera_enabled("a", True)
    assert len(harness.streams["a"]) == 2
    assert manager.get_camera_status("a")["error"] is None
    fresh = harness.emit("a")
    wait_for(lambda: harness.seen(fresh))


def test_timed_out_pending_control_is_cancelled_and_never_applied(harness_factory):
    harness = harness_factory()
    harness.start()
    harness.inference_release.clear()
    harness.emit("a")
    assert harness.inference_entered.wait(3)
    capture = harness.manager.workers["a"]
    try:
        with pytest.raises(TimeoutError):
            harness.manager.set_camera_enabled("a", False, timeout=.03)
    finally:
        harness.inference_release.set()
    packet = harness.emit("b")
    wait_for(lambda: harness.seen(packet))
    assert harness.manager.get_camera_status("a")["enabled"] is True
    assert capture.stops == 0


@pytest.mark.parametrize("fail", [False, True])
def test_shutdown_and_pipeline_errors_fail_pending_controls(harness_factory, fail):
    harness = harness_factory()
    harness.start()
    harness.inference_release.clear()
    harness.emit("a")
    assert harness.inference_entered.wait(3)
    with ThreadPoolExecutor(max_workers=1) as executor:
        command = executor.submit(harness.manager.set_camera_enabled, "b", False, timeout=3)
        wait_for(lambda: harness.manager._commands.qsize() == 1)
        if fail:
            harness.inference_error = RuntimeError("model failed")
        else:
            harness.manager.stop()
        harness.inference_release.set()
        with pytest.raises(RuntimeError):
            command.result(timeout=3)
    harness.thread.join(3)
    assert not harness.thread.is_alive()
    assert not harness.manager.running
    assert len(harness.run_errors) == int(fail)
    if fail:
        assert harness.manager.error is not None
    with pytest.raises(RuntimeError):
        harness.manager.set_camera_enabled("a", True)
    assert harness.writer.closed


def test_stopped_manager_does_not_restart_camera_for_new_commands(harness_factory):
    harness = harness_factory()
    harness.start()
    harness.close()
    starts = harness.captures["a"][0].starts
    with pytest.raises(RuntimeError):
        harness.manager.set_camera_enabled("a", True)
    assert harness.captures["a"][0].starts == starts
