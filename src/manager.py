"""Run with python -m src.manager (or python src/manager.py)."""
import argparse
from concurrent.futures import Future
from dataclasses import dataclass
import logging
import math
from queue import Empty, Full, Queue
import signal
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import cv2

from .camera_worker import CameraWorker
from .config import MODEL_MODES, ROOT, load_config
from .event_engine import EventEngine
from .inference_worker import InferenceWorker
from .logging_config import DEFAULT_LOG_CONFIG, configure_logging
from .metrics import CameraPerformance
from .output_workers import StreamWorker, WriterWorker
from .tracker import Tracker
from .utils import draw_results, make_status_frame, resize_frame
from .api import create_app

CAMERA_CONFIG = ROOT / "configs/cameras.yaml"
MODEL_CONFIG = ROOT / "configs/model.yaml"
logger = logging.getLogger("src.manager")


@dataclass
class _CameraCommand:
    camera_id: str
    enabled: bool
    result: Future


class SystemManager:
    def __init__(self, camera_config=CAMERA_CONFIG, model_config=MODEL_CONFIG, *,
                 display=False, stream=True, inference_factory=InferenceWorker,
                 camera_factory=CameraWorker, tracker_factory=Tracker,
                 writer_factory=WriterWorker, stream_factory=StreamWorker,
                 status_interval_seconds=30, model_mode=None):
        self.settings, self.model_config, self.runtime = load_config(camera_config, model_config)
        self.model_mode = self.runtime.get("model_mode", "shared") if model_mode is None else model_mode
        if self.model_mode not in MODEL_MODES:
            raise ValueError(f"model_mode must be one of {MODEL_MODES}")
        self.runtime["model_mode"] = self.model_mode
        self.performance = {config.capture.camera_id: CameraPerformance(clock=lambda: time.perf_counter())
                            for config in self.settings}
        self.display = display
        self.stream = stream
        self.factories = (inference_factory, camera_factory, tracker_factory, writer_factory, stream_factory)
        self.workers = {}
        self.engines = {}
        self.streamers = {}
        self.writer = None
        self.tracker = None
        self.inference = None
        self.inferences = {}
        self._settings_by_id = {config.capture.camera_id: config for config in self.settings}
        self._state_lock = threading.RLock()
        self._commands = Queue(maxsize=128)
        self._accept_commands = False
        self._enabled = {cid: True for cid in self._settings_by_id}
        self._camera_errors = {}
        self._inactive_workers = {}
        self._inactive_metrics = {}
        self._retired_streamers = {}
        self._stream_dropped = {}
        self.ready_event = threading.Event()
        self.error = None
        self.stop_event = threading.Event()
        self.running = False
        self._used = False
        self.last_frame = {}
        self.generations = {}
        self.processed = 0
        self._cursor = 0 # điểm bắt đầu duyệt lấy batch
        if status_interval_seconds <= 0:
            raise ValueError("status_interval_seconds must be positive")
        self.status_interval_seconds = status_interval_seconds

    def stop(self):
        self.stop_event.set()

    def status(self):
        return self._status_snapshot(self.performance)

    def _status_snapshot(self, camera_ids):
        # Chỉ khóa để sao chép registry, không chặn API trong lúc inference hoặc dừng camera.
        with self._state_lock:
            workers = {cid: self.workers[cid] for cid in camera_ids if cid in self.workers}
            inactive_workers = {cid: self._inactive_workers[cid] for cid in camera_ids if cid in self._inactive_workers}
            streamers = {cid: self.streamers[cid] for cid in camera_ids if cid in self.streamers}
            enabled = {cid: self._enabled[cid] for cid in camera_ids}
            camera_errors = {cid: self._camera_errors[cid] for cid in camera_ids if cid in self._camera_errors}
            inactive_metrics = {cid: self._inactive_metrics[cid] for cid in camera_ids if cid in self._inactive_metrics}
            stream_dropped = {cid: self._stream_dropped[cid] for cid in camera_ids if cid in self._stream_dropped}
            running = self.running
            error = str(self.error) if self.error is not None else None
        metrics = {}
        states = {}
        for cid in camera_ids:
            performance = self.performance[cid]
            # lấy FPS và latency xử lý của camera
            item = performance.snapshot()
            item.update(inactive_metrics.get(cid, {}))
            worker = workers.get(cid, inactive_workers.get(cid))
            if worker is not None and hasattr(worker, "get_metrics"):
                # thêm capture FPS và các bộ đếm reconnect
                item.update(worker.get_metrics())
            sink = streamers.get(cid)
            if sink is not None and hasattr(sink, "get_metrics"):
                # thêm output FPS, queue/send/E2E
                item.update(sink.get_metrics())
            item.setdefault("capture_fps", 0.0)
            item.setdefault("output_fps", 0.0)
            if not enabled[cid]:
                item.update(processing_fps=0.0, capture_fps=0.0, output_fps=0.0,
                            pending_frames=0)
                states[cid] = "disabled"
            elif cid in workers:
                states[cid] = workers[cid].get_status()
            else:
                states[cid] = "starting" if running else "stopped"
            # lưu theo camera id
            metrics[cid] = item
        return {"running": running, "processed_frames": self.processed,
                "model_mode": self.model_mode,
                "error": error, "cameras": states, "camera_enabled": enabled,
                "camera_errors": camera_errors,
                "event_queue": self.writer.queue.qsize() if self.writer else 0,
                "stream_dropped": {cid: stream_dropped.get(cid, 0) + getattr(streamers.get(cid), "dropped", 0)
                                   for cid in stream_dropped.keys() | streamers.keys()},
                "camera_metrics": metrics}

    @staticmethod
    def _camera_snapshot(snapshot, camera_id):
        return {"camera_id": camera_id, "enabled": snapshot["camera_enabled"][camera_id],
                "status": snapshot["cameras"][camera_id],
                "metrics": snapshot["camera_metrics"][camera_id],
                "error": snapshot["camera_errors"].get(camera_id)}

    def list_cameras(self):
        snapshot = self.status()
        return [self._camera_snapshot(snapshot, cid) for cid in self._settings_by_id]

    def get_camera_status(self, camera_id):
        if camera_id not in self._settings_by_id:
            raise KeyError(camera_id)
        return self._camera_snapshot(self._status_snapshot((camera_id,)), camera_id)

    def set_camera_enabled(self, camera_id, enabled, timeout=20):
        """Gửi lệnh cho thread manager và đợi trạng thái sau khi áp dụng."""
        if camera_id not in self._settings_by_id:
            raise KeyError(camera_id)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        command = _CameraCommand(camera_id, enabled, Future())
        with self._state_lock:
            if not self._accept_commands or self.stop_event.is_set():
                raise RuntimeError("Pipeline is not ready or is stopping")
            try:
                self._commands.put_nowait(command)
            except Full as exc:
                raise RuntimeError("Camera control queue is full") from exc
        try:
            return command.result.result(timeout=timeout)
        except TimeoutError as exc:
            if command.result.done():
                return command.result.result()
            # Lệnh chưa chạy được hủy để không tự đổi trạng thái sau khi client đã timeout.
            cancelled = command.result.cancel()
            message = ("Camera control timed out before execution" if cancelled else
                       "Camera control is still finishing; query camera status")
            raise TimeoutError(message) from exc

    def _process_commands(self):
        # Giới hạn số lệnh mỗi lượt để các request liên tục không bỏ đói inference.
        for _ in range(self._commands.qsize()):
            if self.stop_event.is_set():
                break
            try:
                command = self._commands.get_nowait()
            except Empty:
                break
            try:
                if not command.result.set_running_or_notify_cancel():
                    continue
                try:
                    if self._enabled[command.camera_id] != command.enabled:
                        if command.enabled:
                            self._activate_camera(self._settings_by_id[command.camera_id])
                        else:
                            self._deactivate_camera(command.camera_id)
                    command.result.set_result(self.get_camera_status(command.camera_id))
                except Exception as exc:
                    with self._state_lock:
                        self._camera_errors[command.camera_id] = str(exc)
                    logger.exception("Camera control failed", extra={"camera_id": command.camera_id})
                    command.result.set_exception(exc)
            finally:
                self._commands.task_done()

    def _fail_pending_commands(self):
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                break
            if command.result.set_running_or_notify_cancel():
                command.result.set_exception(RuntimeError("Pipeline stopped before applying camera control"))
            self._commands.task_done()

    def _log_performance(self):
        status = self.status()
        for cid, metrics in status["camera_metrics"].items():
            readable = {key: {stat: round(value, 2) if isinstance(value, float) else value
                              for stat, value in item.items()} if isinstance(item, dict)
                        else round(item, 2) if isinstance(item, float) else item
                        for key, item in metrics.items()}
            logger.info("Camera performance: %s", readable, extra={"camera_id": cid})
        # k lấy lại cid vì đã log chi tiết
        return {key: value for key, value in status.items() if key != "camera_metrics"}

    def _initialize(self):
        inference, camera, tracker, writer, stream = self.factories
        if self.model_mode == "shared":
            self.inference = inference(self.model_config)
        else:
            # Mỗi camera giữ một model riêng trong suốt phiên chạy.
            for config in self.settings:
                cid = config.capture.camera_id
                logger.info("Loading camera inference model", extra={"camera_id": cid})
                self.inferences[cid] = inference(self.model_config)
        self.tracker = tracker(fps=self.runtime["tracking_fps"])
        if any(config.event.enabled for config in self.settings):
            self.writer = writer(self.runtime["database"], capacity=self.runtime["event_queue_size"])
            self.writer.start()
        for config in self.settings:
            cid = config.capture.camera_id
            if config.event.enabled:
                self.engines[cid] = EventEngine(config.event)
            logger.info("Event rules: line=%s polygon=%s", config.event.line is not None,
                        config.event.polygon is not None, extra={"camera_id": cid})
            self._activate_camera(config)

    def _activate_camera(self, config):
        """Chỉ thread manager được tạo hoặc khởi động lại worker."""
        cid = config.capture.camera_id
        previous_sink = self._retired_streamers.get(cid)
        if previous_sink is not None:
            previous_sink.close()
            with self._state_lock:
                self._retired_streamers.pop(cid, None)
        worker = self._inactive_workers.get(cid)
        if worker is not None:
            if worker.thread is not None and worker.thread.is_alive():
                raise TimeoutError(f"Capture worker is still stopping: {cid}")
            # Giữ các bộ đếm capture/reconnect, nhưng không dùng lại frame còn tồn.
            if hasattr(worker, "_clear_queue"):
                worker._clear_queue()
            if hasattr(worker, "error"):
                worker.error = None
        else:
            worker = self.factories[1](config.capture)
        sink = None
        with self._state_lock:
            self.workers[cid] = worker
        try:
            if self.stream and config.output:
                sink = self.factories[4](camera_id=cid, **config.output)
                with self._state_lock:
                    self.streamers[cid] = sink
                sink.start()
            worker.start()
            if self.display:
                cv2.namedWindow(cid, cv2.WINDOW_AUTOSIZE)
        except Exception:
            try:
                self._deactivate_camera(cid)
            except Exception:
                logger.exception("Camera startup cleanup failed", extra={"camera_id": cid})
            raise
        with self._state_lock:
            self._enabled[cid] = True
            self._camera_errors.pop(cid, None)
            self._inactive_workers.pop(cid, None)
            self._inactive_metrics.pop(cid, None)
        logger.info("Camera enabled", extra={"camera_id": cid})

    def _deactivate_camera(self, cid):
        """Ngừng riêng capture/output, giữ bộ đếm sự kiện và model đã nạp."""
        with self._state_lock:
            worker = self.workers.pop(cid, None)
            sink = self.streamers.pop(cid, None)
            self._enabled[cid] = False
            if worker is not None:
                self._inactive_workers[cid] = worker
        errors = []
        if worker is not None:
            try:
                worker.stop()
                if worker.thread is not None:
                    worker.thread.join(timeout=8)
                    if worker.thread.is_alive():
                        raise TimeoutError(f"Capture worker did not stop: {cid}")
            except Exception as exc:
                errors.append(exc)
        output_metrics = {}
        if sink is not None:
            try:
                sink.close()
            except Exception as exc:
                errors.append(exc)
                with self._state_lock:
                    self._retired_streamers[cid] = sink
            if hasattr(sink, "get_metrics"):
                output_metrics = sink.get_metrics()
            with self._state_lock:
                self._stream_dropped[cid] = self._stream_dropped.get(cid, 0) + getattr(sink, "dropped", 0)
                self._inactive_metrics[cid] = output_metrics
        self.performance[cid].reset_rate()
        self.last_frame.pop(cid, None)
        self.generations.pop(cid, None)
        if self.tracker is not None:
            self.tracker.reset(cid)
        if cid in self.engines:
            self.engines[cid].reset()
        if self.display:
            cv2.destroyWindow(cid)
        with self._state_lock:
            if errors:
                self._camera_errors[cid] = str(errors[0])
            else:
                self._camera_errors.pop(cid, None)
        logger.info("Camera disabled", extra={"camera_id": cid})
        if errors:
            raise errors[0]

    def _collect_model_separate(self):
        """Gom frame chung; _predict sẽ chuyển tới model riêng của từng camera."""
        return self._collect_frames()

    def _collect_model_together(self):
        """Gom frame để một model dùng chung xử lý cả batch."""
        return self._collect_frames()

    def _collect(self):
        if self.model_mode == "per_camera":
            return self._collect_model_separate()
        return self._collect_model_together()

    def _collect_frames(self):

        ids = list(self.workers) # list cam id
        batch = []
        now = time.perf_counter()
        for offset in range(len(ids)):
            index = (self._cursor + offset) % len(ids)
            cid = ids[index]
            if getattr(self.workers[cid], "error", None) is not None:
                raise RuntimeError(f"Capture worker failed: {cid}") from self.workers[cid].error
            packet = self.workers[cid].get_frame()
            if packet is None:
                if self.display and self.workers[cid].get_status() != "connected":
                    cv2.imshow(cid, make_status_frame(cid, self.workers[cid].get_status()))
                continue
            if now - packet.timestamp > self.runtime["reset_after_seconds"]:
                continue
            previous = self.last_frame.get(cid)
            if previous is not None and (packet.timestamp - previous > self.runtime["reset_after_seconds"] or
                                         self.generations[cid] != packet.generation):
                logger.info("Reset tracking after reconnect or frame gap", extra={"camera_id": cid})
                self.tracker.reset(cid)
                self.performance[cid].reset_rate()
                if cid in self.engines:
                    self.engines[cid].reset()
            self.last_frame[cid] = packet.timestamp
            self.generations[cid] = packet.generation
            batch.append(packet)
            if len(batch) >= self.runtime["batch_size"]:
                self._cursor = (index + 1) % len(ids)
                break
        return batch

    def _predict(self, batch):
        """Trả kết quả theo thứ tự đầu vào và mốc thời gian inference từng camera."""
        if not batch:
            return [], {}
        if self.model_mode == "shared":
            groups = {None: list(enumerate(batch))}
        else:
            groups = {}
            for index, packet in enumerate(batch):
                groups.setdefault(packet.camera_id, []).append((index, packet))

        detections = [None] * len(batch)
        timings = {}
        for cid, entries in groups.items():
            worker = self.inference if cid is None else self.inferences[cid]
            packets = [packet for _, packet in entries]
            started = time.perf_counter()
            results = worker.predict(packets)
            finished = time.perf_counter()
            if len(results) != len(packets):
                raise RuntimeError("Inference result count does not match input batch")
            for (index, packet), detection in zip(entries, results):
                if detection.camera_id != packet.camera_id:
                    raise RuntimeError("Inference result camera does not match input frame")
                detections[index] = detection
                # Model riêng chỉ ghi thời gian của chính nó, không cộng các model khác.
                timings[packet.camera_id] = (started, finished)
        return detections, timings

    def run(self, duration=None):
        if self._used:
            raise RuntimeError("Create a new SystemManager for each run")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive")
        with self._state_lock:
            self._used = True
            self.running = True
        logger.info("Starting pipeline: cameras=%d model_mode=%s batch_size=%d display=%s stream=%s",
                    len(self.settings), self.model_mode, self.runtime["batch_size"], self.display, self.stream)
        try:
            self._initialize()
            with self._state_lock:
                self._accept_commands = not self.stop_event.is_set()
            self.ready_event.set()
            deadline = time.monotonic() + duration if duration is not None else float("inf")
            next_status = time.monotonic() + self.status_interval_seconds
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                self._process_commands()
                if self.stop_event.is_set():
                    break
                if self.writer is not None:
                    self.writer.check()
                for sink in self.streamers.values():
                    sink.check()
                batch = self._collect()
                detections, inference_times = self._predict(batch)
                for detection in detections:
                    inference_started, inference_finished = inference_times[detection.camera_id]
                    tracking = self.tracker.update(detection)
                    engine = self.engines.get(tracking.camera_id)
                    result = engine.process(tracking) if engine is not None else None
                    if result is not None:
                        self.writer.submit(result)
                    performance = self.performance[tracking.camera_id]
                    performance.record(tracking.timestamp, inference_started, inference_finished,
                                       time.perf_counter())
                    if self.display or tracking.camera_id in self.streamers:
                        geometry = engine.geometry(tracking.frame.shape) if engine is not None else (None, None)
                        # lấy fps tử camera performance từ camera đang vẽ bằng camera id
                        overlay = {"processing_fps": performance.rate.value()}
                        annotated = draw_results(tracking, result, geometry, performance=overlay)
                        if tracking.camera_id in self.streamers:
                            self.streamers[tracking.camera_id].submit(annotated, captured_at=tracking.timestamp)
                        if self.display:
                            resized_frame = resize_frame(annotated)
                            cv2.imshow(tracking.camera_id, resized_frame)
                    self.processed += 1
                if self.display:
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        self.stop()
                    for cid in self.workers:
                        if cv2.getWindowProperty(cid, cv2.WND_PROP_VISIBLE) < 1:
                            self.stop()
                if not batch:
                    self.stop_event.wait(0.005)
                if time.monotonic() >= next_status: # check thời gian benchmark
                    logger.info("Pipeline status: %s", self._log_performance())
                    next_status = time.monotonic() + self.status_interval_seconds
        except Exception as exc:
            with self._state_lock:
                self.error = exc
            logger.exception("Pipeline failed; stopping workers")
            raise
        finally:
            with self._state_lock:
                self.running = False
                self._accept_commands = False
            self.ready_event.set()
            try:
                self._close()
            except Exception as exc:
                with self._state_lock:
                    if self.error is None:
                        self.error = exc
                raise

    start = run

    def _close(self):
        self.stop()
        with self._state_lock:
            self._accept_commands = False
            workers = dict(self._inactive_workers)
            workers.update(self.workers)
            sinks = [*self.streamers.values(), *self._retired_streamers.values(), self.writer]
        self._fail_pending_commands()
        logger.info("Stopping pipeline and draining event queue")
        errors = []
        for worker in workers.values():
            try:
                worker.stop()
            except Exception as exc:
                errors.append(exc)
        deadline = time.monotonic() + 8
        for worker in workers.values():
            if worker.thread is not None:
                try:
                    worker.thread.join(timeout=max(0, deadline - time.monotonic()))
                    if worker.thread.is_alive():
                        errors.append(RuntimeError("Capture thread did not stop before deadline"))
                except Exception as exc:
                    errors.append(exc)
        for sink in sinks:
            if sink is not None:
                try:
                    sink.close()
                except Exception as exc:
                    errors.append(exc)
        if self.display:
            cv2.destroyAllWindows()
        if errors:
            for error in errors:
                logger.error("Shutdown error: %s", error)
            if sys.exc_info()[0] is None:
                raise RuntimeError("Pipeline shutdown failed") from errors[0]
        logger.info("Pipeline stopped: %s", self._log_performance())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cameras", type=Path, default=CAMERA_CONFIG)
    parser.add_argument("--model", type=Path, default=MODEL_CONFIG)
    parser.add_argument("--model_mode", choices=MODEL_MODES, default=MODEL_MODES[0])
    parser.add_argument("--display", action="store_true", help="Mở preview cùng API")
    parser.add_argument("--no_stream", action="store_true")
    parser.add_argument("--log_config", type=Path, default=DEFAULT_LOG_CONFIG)
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log_file", type=Path)
    args = parser.parse_args()
    controller = configure_logging(args.log_config, level=args.log_level, log_file=args.log_file)
    previous_loggers = {}
    try:
        import uvicorn

        # Dùng chung hàng đợi, định dạng và cơ chế che thông tin nhạy cảm của ứng dụng.
        source_logger = logging.getLogger("src")
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            server_logger = logging.getLogger(name)
            previous_loggers[name] = (server_logger.handlers[:], server_logger.level, server_logger.propagate)
            server_logger.handlers = source_logger.handlers[:]
            server_logger.setLevel(source_logger.level)
            server_logger.propagate = False
        def stop_api():
            # Đóng preview hoặc nhấn Q/Esc sẽ kết thúc cả lệnh chạy API.
            server.should_exit = True

        application = create_app(args.cameras,
                                 args.model,
                                 display=args.display,
                                 manager_factory=SystemManager,
                                 stream=not args.no_stream,
                                 model_mode=args.model_mode,
                                 status_interval_seconds=controller.status_interval_seconds,
                                 on_pipeline_stopped=stop_api if args.display else None)
        server = uvicorn.Server(uvicorn.Config(application, host=args.host, port=args.port,
                                             workers=1, log_config=None))
        try:
            server.run()
        except KeyboardInterrupt:
            pass
        if not server.started:
            raise SystemExit(3)
    finally:
        for name, (handlers, level, propagate) in previous_loggers.items():
            server_logger = logging.getLogger(name)
            server_logger.handlers, server_logger.level, server_logger.propagate = handlers, level, propagate
        controller.close()


if __name__ == "__main__":
    main()