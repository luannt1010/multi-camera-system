"""Output telemetry measured at the actual asynchronous send boundary."""
import threading
from types import SimpleNamespace

import pytest

import src.output_workers as module


class FakeClock:
    def __init__(self, initial=10.0):
        self.now = initial
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def set(self, value):
        with self.lock:
            self.now = value


def make_worker(monkeypatch, clock, send):
    class FakeOutput:
        fps = 100

        def __init__(self, **options):
            self.closed = False
            self.frames = []

        def send(self, frame):
            self.frames.append(frame)
            return send(frame)

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "StreamOutput", FakeOutput)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=clock))
    return module.StreamWorker("camera-a", output_url="unused")


def assert_unavailable(stat):
    assert stat == {"last_ms": None, "avg_ms": None, "p95_ms": None}


def test_stream_replacement_keeps_latest_frame_and_its_timestamps(monkeypatch):
    clock = FakeClock()
    worker = make_worker(monkeypatch, clock, lambda frame: True)
    first, latest = object(), object()
    worker.submit(first, captured_at=8)
    clock.set(11)
    worker.submit(latest, captured_at=9)
    metrics = worker.get_metrics()
    assert metrics["dropped_frames"] == 1
    assert metrics["pending_frames"] == 1
    item = worker.queue.get_nowait()
    worker.queue.task_done()
    assert item.frame is latest
    assert item.captured_at == 9
    assert item.submitted_at == 11
    assert metrics["sent_frames"] == 0
    assert_unavailable(metrics["e2e_ms"])
    worker.close()


def test_stream_close_discards_pending_frame_without_start_and_is_idempotent(monkeypatch):
    clock = FakeClock()
    worker = make_worker(monkeypatch, clock, lambda frame: True)
    worker.submit(object(), captured_at=8)
    assert worker.get_metrics()["pending_frames"] == 1
    worker.close()
    assert worker.output.closed
    metrics = worker.get_metrics()
    assert metrics["pending_frames"] == 0
    assert metrics["dropped_frames"] == 1
    assert metrics["sent_frames"] == 0
    assert metrics["send_failures"] == 0
    joined = threading.Thread(target=worker.queue.join, daemon=True)
    joined.start()
    joined.join(timeout=1)
    assert not joined.is_alive()
    worker.close()
    assert worker.get_metrics()["dropped_frames"] == 1
    assert worker.get_metrics()["pending_frames"] == 0


def test_stream_success_measures_queue_send_and_local_e2e(monkeypatch):
    clock = FakeClock()
    finish_times = iter([12, 15])

    def send(frame):
        clock.set(next(finish_times))
        return True

    worker = make_worker(monkeypatch, clock, send)
    first, second = object(), object()
    worker.submit(first, captured_at=5)
    clock.set(11)
    worker.start()
    try:
        worker.queue.join()
        first_metrics = worker.get_metrics()
        assert first_metrics["output_queue_ms"]["last_ms"] == pytest.approx(1000)
        assert first_metrics["send_ms"]["last_ms"] == pytest.approx(1000)
        assert first_metrics["e2e_ms"]["last_ms"] == pytest.approx(7000)
        clock.set(14)
        worker.submit(second, captured_at=12.5)
        worker.queue.join()
        metrics = worker.get_metrics()
        assert worker.output.frames == [first, second]
        assert metrics["sent_frames"] == 2
        assert metrics["send_failures"] == 0
        assert metrics["output_fps"] > 0
        assert metrics["output_queue_ms"]["last_ms"] == pytest.approx(0)
        assert metrics["send_ms"]["last_ms"] == pytest.approx(1000)
        assert metrics["e2e_ms"]["last_ms"] == pytest.approx(2500)
        assert metrics["e2e_ms"]["avg_ms"] == pytest.approx(4750)
        assert metrics["pending_frames"] == 0
        clock.set(21)
        stale = worker.get_metrics()
        assert stale["output_fps"] == 0
        assert_unavailable(stale["e2e_ms"])
    finally:
        worker.close()
    assert worker.output.closed


def test_stream_failed_send_does_not_count_delivery_or_latency(monkeypatch):
    clock = FakeClock()

    def fail(frame):
        clock.set(12)
        worker.stopping.set()
        return False

    worker = make_worker(monkeypatch, clock, fail)
    worker.submit(object(), captured_at=8)
    clock.set(11)
    worker.start()
    worker.thread.join(timeout=2)
    worker.close()
    assert not worker.thread.is_alive()
    metrics = worker.get_metrics()
    assert metrics["send_failures"] == 1
    assert metrics["sent_frames"] == 0
    assert metrics["output_fps"] == 0
    assert_unavailable(metrics["output_queue_ms"])
    assert_unavailable(metrics["send_ms"])
    assert_unavailable(metrics["e2e_ms"])
    assert worker.output.closed


def test_stream_legacy_submit_has_no_capture_latency(monkeypatch):
    clock = FakeClock()

    def send(frame):
        clock.set(12)
        return True

    worker = make_worker(monkeypatch, clock, send)
    worker.submit(object())
    clock.set(11)
    worker.start()
    try:
        worker.queue.join()
        metrics = worker.get_metrics()
        assert metrics["sent_frames"] == 1
        assert metrics["send_ms"]["last_ms"] == pytest.approx(1000)
        assert_unavailable(metrics["e2e_ms"])
    finally:
        worker.close()


def test_stream_send_exception_reports_failure_and_closes_output(monkeypatch):
    clock = FakeClock()

    def send(frame):
        raise OSError("encoder unavailable")

    worker = make_worker(monkeypatch, clock, send)
    worker.submit(object(), captured_at=0)
    worker.start()
    worker.thread.join(timeout=2)
    assert not worker.thread.is_alive()
    assert worker.get_metrics()["send_failures"] == 1
    assert worker.get_metrics()["sent_frames"] == 0
    assert_unavailable(worker.get_metrics()["e2e_ms"])
    assert worker.output.closed
    with pytest.raises(RuntimeError, match="RTSP output worker failed"):
        worker.close()
