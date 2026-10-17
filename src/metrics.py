from collections import deque
import math
import threading
import time


class RollingRate:
    def __init__(self, window_seconds=5.0, clock=None):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        self.window_seconds = window_seconds
        self._clock = clock or time.perf_counter
        self._timestamps = deque(maxlen=4096)
        self._first = None
        self._lock = threading.Lock()

    def mark(self, now=None):
        """Hàm ghi nhận thêm 1 frame"""
        now = self._clock() if now is None else now
        with self._lock:
            if self._first is None:
                self._first = now
            self._timestamps.append(now)
            self._prune(now)

    def _prune(self, now):
        """Xóa các mốc thời gian đã cũ"""
        while self._timestamps and self._timestamps[0] < now - self.window_seconds:
            self._timestamps.popleft()

    def value(self, now=None):
        """Hàm tính FPS"""
        now = self._clock() if now is None else now
        with self._lock:
            self._prune(now)
            if not self._timestamps:
                return 0.0
            start = max(self._first, now - self.window_seconds)
            elapsed = now - start
            if elapsed <= 0:
                return 0.0
            # True -> 1, False -> 0 nếu first còn trong timestamp
            count = len(self._timestamps) - int(self._first >= now - self.window_seconds)
            return count / elapsed

    def reset(self):
        with self._lock:
            self._timestamps.clear()
            self._first = None


class RollingLatency:
    def __init__(self, window_seconds=5.0, clock=None):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        self.window_seconds = window_seconds
        self._clock = clock or time.perf_counter
        self._samples = deque(maxlen=4096)
        self._lock = threading.Lock()

    def record(self, milliseconds, now=None):
        """Hàm ghi lại độ trễ ở thời điểm hiện tại"""
        if not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("Latency must be finite and nonnegative")
        now = self._clock() if now is None else now
        with self._lock:
            self._samples.append((now, milliseconds))
            self._prune(now)

    def _prune(self, now):
        """Hàm xóa mẫu quá hạn"""
        # samples[0] -> mẫu cũ nhất, sample[0][1] thời điểm của nó, sample[0][1] đỗ trễ của mẫu
        while self._samples and self._samples[0][0] < now - self.window_seconds:
            self._samples.popleft()

    def snapshot(self, now=None):
        """Hàm lấy thống kê"""
        now = self._clock() if now is None else now
        with self._lock:
            self._prune(now) # Xóa mẫu hết hạn tại now
            if not self._samples:
                return {"last_ms": None, "avg_ms": None, "p95_ms": None}
            latest = self._samples[-1][1]
            values = sorted(sample[1] for sample in self._samples)
        # Tính độ trễ tại phân vị 95
        rank = .95 * (len(values) - 1)
        lower, upper = math.floor(rank), math.ceil(rank)
        p95 = values[lower] + (values[upper] - values[lower]) * (rank - lower)
        return {"last_ms": latest, "avg_ms": sum(values) / len(values), "p95_ms": p95}


class CameraPerformance:
    """
    frame_age: độ trễ tự khi nhận frame -> infer
    inference: thời gian infer
    processing_latency: độ trễ tự khi nhận frame -> xong tracker/event
    """
    def __init__(self, clock=None):
        self._clock = clock or time.perf_counter
        self.rate = RollingRate(clock=self._clock)
        self.frame_age = RollingLatency(clock=self._clock)
        self.inference = RollingLatency(clock=self._clock)
        self.processing_latency = RollingLatency(clock=self._clock)
        self._processed_frames = 0
        self._lock = threading.Lock()

    def record(self, captured_at, inference_started_at, inference_finished_at, completed_at=None):
        """Hàm xử lý 4 mốc thời gian"""
        # thời điểm xử lý xong frame
        completed_at = self._clock() if completed_at is None else completed_at
        # Ghi 1 frame vào
        self.rate.mark(completed_at)
        # bắt đầu infer - lúc nhận frame, *1000: giây->ms và thời điểm ghi mẫu
        self.frame_age.record(max(0, inference_started_at - captured_at) * 1000, completed_at)
        self.inference.record(max(0, inference_finished_at - inference_started_at) * 1000, completed_at)
        # finish - lúc nhận frame
        self.processing_latency.record(max(0, completed_at - captured_at) * 1000, completed_at)
        with self._lock:
            self._processed_frames += 1 # tăng số frame đã xử lý

    def snapshot(self, now=None):
        """Hàm đọc kết quả"""
        now = self._clock() if now is None else now
        with self._lock:
            count = self._processed_frames
        return {"processing_fps": self.rate.value(now), "processed_frames": count,
                "frame_age_ms": self.frame_age.snapshot(now),
                "inference_ms": self.inference.snapshot(now),
                "processing_latency_ms": self.processing_latency.snapshot(now)}

    def reset_rate(self):
        self.rate.reset()
