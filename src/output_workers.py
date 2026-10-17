"""Bounded asynchronous sinks. SQLite belongs exclusively to its writer thread."""
import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue

from .event_writer import EventWriter
from .metrics import RollingLatency, RollingRate
from .stream_ouput import StreamOutput

logger = logging.getLogger(__name__)


class WriterWorker:
    def __init__(self, path, capacity=128):
        self.path = path
        self.queue = Queue(maxsize=capacity)
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="event-writer", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(15):
            raise TimeoutError("Event writer initialization timed out")
        self.check()
        logger.info("Event writer worker ready: queue_capacity=%d", self.queue.maxsize)

    def check(self):
        if self.error is not None:
            raise RuntimeError("Event writer failed") from self.error

    def submit(self, result):
        self.check()
        if any(v.get("zone_entry", False) or v["crossed"] for v in result.events.values()):
            try:
                self.queue.put_nowait(result)
            except Full as exc:
                logger.error("Event queue full: capacity=%d", self.queue.maxsize,
                             extra={"camera_id": result.camera_id})
                raise RuntimeError("Event queue full; stopping to avoid silent event loss") from exc

    def _run(self):
        writer = None
        try:
            writer = EventWriter(self.path)
            self.ready.set()
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    result = self.queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    writer.save(result)
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.error = exc
            self.ready.set()
            logger.exception("Event writer worker failed")
        finally:
            if writer is not None:
                writer.close()

    def close(self):
        self.stopping.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise TimeoutError("Event writer did not finish draining")
        self.check()
        logger.info("Event writer worker stopped; pending=%d", self.queue.qsize())


@dataclass(frozen=True)
class _StreamFrame:
    frame: object
    captured_at: float | None
    submitted_at: float


class StreamWorker:
    def __init__(self, camera_id="-", **options):
        self.camera_id = camera_id
        self.output = StreamOutput(camera_id=camera_id, **options)
        self.queue = Queue(maxsize=1)
        self.stopping = threading.Event()
        self.dropped = 0
        self._sent_frames = 0
        self._send_failures = 0
        self._metrics_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._clock = time.perf_counter
        # fps gửi thành công
        self._output_rate = RollingRate(clock=self._clock)
        # thời gian chờ trong queue
        self._queue_latency = RollingLatency(clock=self._clock)
        # thời gian gửi
        self._send_latency = RollingLatency(clock=self._clock)
        # độ trễ từ nhận frame -> gửi xong
        self._e2e_latency = RollingLatency(clock=self._clock)
        self.error = None
        self.thread = threading.Thread(target=self._run, name=f"rtsp-{camera_id}", daemon=True)

    def start(self):
        self.thread.start()
        logger.info("RTSP worker started", extra={"camera_id": self.camera_id})

    def submit(self, frame, *, captured_at=None):
        """Queue the latest frame; capture timestamps use ``perf_counter``.

        Without a capture timestamp the output still works, but cannot benchmark
        capture-to-FFmpeg latency for that frame.
        """
        self.check()
        item = _StreamFrame(frame, captured_at, self._clock())
        with self._submit_lock:
            try:
                self.queue.put_nowait(item)
            except Full:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                    with self._metrics_lock:
                        self.dropped += 1
                except Empty:
                    pass
                self.queue.put_nowait(item)

    def get_metrics(self):
        """Return output health without waiting for the encoder.

        ``e2e_ms`` ends when ``StreamOutput.send`` has flushed Python's pipe
        to FFmpeg. It does not measure network transport or viewer playback.
        """
        now = self._clock()
        with self._metrics_lock:
            return {
                "output_fps": self._output_rate.value(now=now),
                "sent_frames": self._sent_frames,
                "send_failures": self._send_failures,
                "dropped_frames": self.dropped,
                "pending_frames": self.queue.qsize(),
                "output_queue_ms": self._queue_latency.snapshot(now=now),
                "send_ms": self._send_latency.snapshot(now=now),
                "e2e_ms": self._e2e_latency.snapshot(now=now),
            }

    def check(self):
        if self.error is not None:
            raise RuntimeError("RTSP output worker failed") from self.error

    def _run(self):
        try:
            while not self.stopping.is_set():
                try:
                    item = self.queue.get(timeout=0.1)
                except Empty:
                    continue
                started = self._clock()
                try:
                    try:
                        sent = self.output.send(item.frame)
                    except Exception:
                        with self._metrics_lock:
                            self._send_failures += 1
                        raise
                    finished = self._clock()
                    with self._metrics_lock:
                        if sent:
                            self._sent_frames += 1
                            self._output_rate.mark(now=finished)
                            # ms = giây * 1000
                            # bắt đầu gửi - lúc enqueue -> thời gian chờ queue
                            self._queue_latency.record((started - item.submitted_at) * 1000, now=finished)
                            # gửi xong - bắt đầu gửi -> thời gian send
                            self._send_latency.record((finished - started) * 1000, now=finished)
                            # chỉ tính e2e khi biết mốc nhận frame gốc
                            if item.captured_at is not None:
                                self._e2e_latency.record(
                                    (finished - item.captured_at) * 1000, now=finished)
                        else:
                            self._send_failures += 1
                finally:
                    self.queue.task_done()
                if not sent:
                    self.stopping.wait(2)
                # 1 / self.output.fps -> thời gian mục tiêu giữa 2 lần gửi
                # self._clock() - started -> thời gian đã gửi 1 lần
                # trừ nhau ra thời gian còn lại
                self.stopping.wait(max(0, 1 / self.output.fps - (self._clock() - started)))
        except Exception as exc:
            self.error = exc
            logger.exception("RTSP output worker failed", extra={"camera_id": self.camera_id})
        finally:
            self.output.close()

    def close(self):
        self.stopping.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise TimeoutError("RTSP worker did not stop")
        else:
            self.output.close()
        discarded = 0
        with self._submit_lock:
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
                self.queue.task_done()
                discarded += 1
        with self._metrics_lock:
            self.dropped += discarded
        self.check()
        logger.info("RTSP worker stopped: sent_frames=%d send_failures=%d dropped_frames=%d",
                    self._sent_frames, self._send_failures, self.dropped,
                    extra={"camera_id": self.camera_id})
