import cv2
import time
import threading
import logging
import numpy as np
from queue import Queue, Empty, Full
from .metrics import RollingRate
from .schemas import FramePacket, CameraConfig


logger = logging.getLogger(__name__)


class CameraWorker:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
        self.thread = None
        self.lock = threading.Lock()
        self.running = False
        self.stop_event = threading.Event()

        #Only keep the latest frame
        self.frame_queue = Queue(maxsize=1)
        self.status = "disconnected"
        self.fps = 30.0
        self.generation = 0
        self.error = None
        self._last_disconnect_warning = float("-inf")
        self._capture_rate = RollingRate()
        self._captured_frames = 0
        self._connection_attempts = 0
        self._reconnect_attempts = 0
        self._reconnect_count = 0
        # check xem camera đã từng có hình chưa
        self._has_captured_frame = False
        self._awaiting_reconnect_frame = False

        #Auto extract type
        source = str(config.source).lower()
        if source.isdigit():
            self.type = "webcam"
        elif source.startswith(("rtsp://", "rtsps://")):
            self.type = "rtsp"
        else:
            self.type = "file"

    def _set_status(self, status: str) -> None:
        with self.lock:
            previous = self.status
            self.status = status
            if status in {"disconnected", "stopped"}:
                self._capture_rate.reset()
            connection_attempts = self._connection_attempts
            reconnect_attempts = self._reconnect_attempts
            reconnect_count = self._reconnect_count
        if previous == status:
            return
        context = {"camera_id": self.config.camera_id}
        if status == "connecting":
            logger.debug("Connecting to camera", extra=context)
        elif status == "connected":
            if reconnect_count:
                logger.info(
                    "Camera connection recovered (generation=%d, reconnect_count=%d, "
                    "reconnect_attempts=%d, connection_attempts=%d)",
                    self.generation, reconnect_count, reconnect_attempts, connection_attempts,
                    extra=context,
                )
            else:
                logger.info("Camera connected (generation=%d)", self.generation, extra=context)
        elif status == "stopped":
            logger.info("Camera stopped", extra=context)
        elif status == "disconnected":
            now = time.monotonic()
            if now - self._last_disconnect_warning >= 30:
                self._last_disconnect_warning = now
                logger.warning(
                    "Camera disconnected; retrying in 2 seconds (connection_attempts=%d, "
                    "reconnect_attempts=%d, reconnect_count=%d)",
                    connection_attempts, reconnect_attempts, reconnect_count, extra=context,
                )

    def get_status(self) -> str:
        with self.lock:
            return self.status

    def get_metrics(self) -> dict[str, float | int]:
        """Hàm lấy thông kê số liệu"""
        with self.lock:
            return {
                "capture_fps": self._capture_rate.value(),
                "captured_frames": self._captured_frames,
                "connection_attempts": self._connection_attempts,
                "reconnect_attempts": self._reconnect_attempts,
                "reconnect_count": self._reconnect_count,
            }

    def _begin_connection_attempt(self) -> None:
        """Hàm tính lần mở nguồn"""
        with self.lock:
            self._connection_attempts += 1
            self._awaiting_reconnect_frame = self._has_captured_frame
            if self._awaiting_reconnect_frame:
                self._reconnect_attempts += 1

    def _init_cap(self) -> bool:
        source = self.config.source

        if self.type == "webcam":
            cap = cv2.VideoCapture(int(source))
        elif self.type == "rtsp":
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
            ])
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            cap.release()
            return False

        self.cap = cap
        self.generation += 1

        if self.type == "file":
            fps = cap.get(cv2.CAP_PROP_FPS) # k prop video bị phát nhanh
            self.fps = fps if 1 <= fps <= 120 else 30.0
            logger.debug(
                "Source video frame rate: %.2f FPS", self.fps,
                extra={"camera_id": self.config.camera_id},
            )

        return True

    def _put_latest(self, frame: np.ndarray) -> None:

        packet = FramePacket(
            camera_id=self.config.camera_id,
            frame=frame,
            timestamp=time.perf_counter(), # thời điểm nhận frame
            generation=self.generation,
        )
        with self.lock:
            # ghi lại 1 frame đã đọc thành công
            self._capture_rate.mark(packet.timestamp)
            # tăng 1 khi đã xử lý
            self._captured_frames += 1
            # Xác nhận đã từng có frame
            self._has_captured_frame = True
            # nếu đợi có frame khi _has_captured_frame = True thì tính là reconnect
            if self._awaiting_reconnect_frame:
                self._reconnect_count += 1
                self._awaiting_reconnect_frame = False
        try:
            self.frame_queue.put_nowait(packet)
        except Full:
            # nếu full xóa frame cũ
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
            # thêm mới
            self.frame_queue.put_nowait(packet)

    def get_frame(self) -> FramePacket | None:
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None

    def _clear_queue(self) -> None:
        while True:
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

    def start(self) -> None:
        if self.running:
            return
        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(
            target=self._update_frame,
            name=f"capture-{self.config.camera_id}",
            daemon=True
        )
        self.thread.start()

    def _update_frame(self) -> None:
        try:
            while self.running:

                # Connect / Reconnect
                if self.cap is None:
                    self._set_status("connecting")
                    self._begin_connection_attempt()
                    try:
                        opened = self._init_cap()
                    except cv2.error:
                        opened = False

                    if not opened:
                        self._set_status("disconnected")
                        self._clear_queue()
                        self.stop_event.wait(2)
                        continue

                start_time = time.perf_counter()

                try:
                    ret, frame = self.cap.read()
                except cv2.error:
                    ret, frame = False, None

                if not self.running:
                    break

                if not ret or frame is None:
                    self._set_status("disconnected")
                    self._clear_queue()
                    self._release_cap()
                    # For files, this restarts playback
                    self.stop_event.wait(2)
                    continue

                # Put newest frame into queue
                self._put_latest(frame)
                self._set_status("connected")

                # MP4: limit reading to source FPS
                if self.type == "file":
                    elapsed = time.perf_counter() - start_time
                    delay = max(0.0, (1.0 / self.fps) - elapsed)
                    self.stop_event.wait(delay)

        except Exception as exc:
            self.error = exc
            logger.exception("Capture worker failed", extra={"camera_id": self.config.camera_id})
        finally:
            self.running = False
            self._release_cap()
            self._clear_queue()
            self._set_status("stopped")

    def _release_cap(self) -> None:
        # Called only by the capture thread
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stop(self) -> None:
        self.running = False
        self.stop_event.set()
        with self.lock:
            self._capture_rate.reset()


