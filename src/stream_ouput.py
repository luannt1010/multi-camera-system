import logging
import subprocess
import time
import threading
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .logging_config import configure_logging

logger = logging.getLogger("src.stream_ouput")


def _read_ffmpeg_log(process):
    previous, last_logged, suppressed = None, 0.0, 0
    try:
        for raw in process.stderr:
            message = raw.decode("utf-8", errors="replace").strip()
            if not message:
                continue
            now = time.monotonic()
            if message == previous and now - last_logged < 30:
                suppressed += 1
                continue
            context = {"camera_id": getattr(process, "camera_id", "-")}
            if suppressed:
                logger.warning("Suppressed %d repeated FFmpeg messages", suppressed, extra=context)
                suppressed = 0
            level = logging.ERROR if any(tag in message.lower() for tag in
                                         ("[error]", "[fatal]", "error ", "failed", "broken pipe")) else logging.WARNING
            logger.log(level, "FFmpeg: %s", message, extra=context)
            previous, last_logged = message, now
    except (OSError, ValueError):
        logger.debug("FFmpeg log pipe closed", extra={"camera_id": getattr(process, "camera_id", "-")})
    finally:
        if suppressed:
            logger.warning("Suppressed %d repeated FFmpeg messages", suppressed,
                           extra={"camera_id": getattr(process, "camera_id", "-")})


def start_stream(output_url, width, height, fps, camera_id="-"):
    """Khởi động một process FFmpeg."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "level+warning",
        "-nostats",

        # Nhận frame BGR từ Python qua stdin.
        "-probesize", "32",
        "-analyzeduration", "0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",

        # Nén thành H.264.
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(fps),
        "-bf", "0",

        # Publish video đến MediaMTX.
        "-rtsp_transport", "tcp",
        "-f", "rtsp",
        output_url,
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    process.camera_id = camera_id
    reader = threading.Thread(target=_read_ffmpeg_log, args=(process,),
                              name=f"ffmpeg-log-{camera_id}", daemon=True)
    process.log_thread = reader
    reader.start()
    return process


def stop_stream(process):
    """Dừng FFmpeg và đóng pipe."""
    if process is None:
        return

    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    reader = getattr(process, "log_thread", None)
    if reader is not None:
        reader.join(timeout=1)
        if not reader.is_alive() and process.stderr is not None:
            process.stderr.close()


class StreamOutput:
    def __init__(self, output_url, width=800, height=650, fps=15, camera_id="-"):
        self.camera_id = camera_id
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self._last_error = float("-inf")
        self._started_once = False

    def send(self, annotated_frame):
        if annotated_frame is None:
            return False
        try:
            # Chưa có FFmpeg hoặc FFmpeg đã chết → mở lại.
            if self.process is None or self.process.poll() is not None:
                self.close()  # Dọn process cũ nếu có.
                self.process = start_stream(self.output_url, self.width, self.height, self.fps,
                                            camera_id=self.camera_id)
                logger.log(logging.DEBUG if self._started_once else logging.INFO,
                           "FFmpeg started: pid=%s size=%dx%d fps=%d", self.process.pid,
                           self.width, self.height, self.fps, extra={"camera_id": self.camera_id})
                self._started_once = True

            # Gửi frame vào FFmpeg.
            source_h, source_w = annotated_frame.shape[:2]
            scale = min(self.width / source_w, self.height / source_h)
            resized = cv2.resize(annotated_frame, (max(1, round(source_w * scale)),
                                                  max(1, round(source_h * scale))))
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            h, w = resized.shape[:2]
            top, left = (self.height - h) // 2, (self.width - w) // 2
            frame[top:top + h, left:left + w] = resized
            process = self.process
            def abort_write():
                try:
                    if process.poll() is None:
                        logger.warning("FFmpeg pipe stalled for 3 seconds; terminating encoder",
                                       extra={"camera_id": self.camera_id})
                        process.kill()
                except OSError:
                    pass
            watchdog = threading.Timer(3.0, abort_write)
            watchdog.daemon = True
            watchdog.start()
            try:
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
            finally:
                watchdog.cancel()
            return True

        except (BrokenPipeError, OSError):
            now = time.monotonic()
            if now - self._last_error >= 30:
                logger.warning("FFmpeg send failed; output worker will retry",
                               extra={"camera_id": self.camera_id})
                self._last_error = now
            logger.debug("FFmpeg send failure details", exc_info=True, extra={"camera_id": self.camera_id})
            self.close()
            return False

    def close(self):
        if self.process is not None:
            logger.debug("Closing FFmpeg: pid=%s", self.process.pid, extra={"camera_id": self.camera_id})
        stop_stream(self.process)
        self.process = None


def main():
    controller = configure_logging()

    # Dùng webcam để thử. Có thể thay 0 bằng URL RTSP nguồn.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        logger.error("Cannot open demo camera")
        controller.close()
        raise RuntimeError("Không mở được camera")

    # Chỉ tạo MỘT LẦN trước vòng lặp.
    streamer = StreamOutput(
        output_url="rtsp://127.0.0.1:8554/cam_01",
        width=800,
        height=650,
        fps=15,
    )

    next_frame_time = 0.0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                logger.error("Cannot read demo camera frame")
                break

            # Tiếp tục đọc nguồn, chỉ gửi tối đa theo FPS đã chọn.
            now = time.monotonic()

            if now < next_frame_time:
                continue

            next_frame_time = now + 1.0 / streamer.fps

            # Trong project của bạn, đây là frame sau khi đã vẽ
            # box, track ID, polygon, line và bộ đếm.
            annotated_frame = frame.copy()

            cv2.putText(
                annotated_frame,
                "Camera 01",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            streamer.send(annotated_frame)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt; stopping stream demo")

    finally:
        streamer.close()
        cap.release()
        logger.info("Demo camera and FFmpeg closed")
        controller.close()


if __name__ == "__main__":
    main()
