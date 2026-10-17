"""Run with python -m src.manager (or python src/manager.py)."""
import argparse
import logging
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

CAMERA_CONFIG = ROOT / "configs/cameras.yaml"
MODEL_CONFIG = ROOT / "configs/model.yaml"
logger = logging.getLogger("src.manager")


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
        self.stop_event = threading.Event()
        self.running = False
        self._used = False
        self.last_frame = {}
        self.generations = {}
        self.processed = 0
        self._cursor = 0
        if status_interval_seconds <= 0:
            raise ValueError("status_interval_seconds must be positive")
        self.status_interval_seconds = status_interval_seconds

    def stop(self):
        self.stop_event.set()

    def status(self):
        metrics = {}
        for cid, performance in self.performance.items():
            # lấy FPS và latency xử lý của camera
            item = performance.snapshot()
            worker = self.workers.get(cid)
            if worker is not None and hasattr(worker, "get_metrics"):
                # thêm capture FPS và các bộ đếm reconnect
                item.update(worker.get_metrics())
            sink = self.streamers.get(cid)
            if sink is not None and hasattr(sink, "get_metrics"):
                # thêm output FPS, queue/send/E2E
                item.update(sink.get_metrics())
            # lưu theo camera id
            metrics[cid] = item
        return {"running": self.running, "processed_frames": self.processed,
                "model_mode": self.model_mode,
                "cameras": {cid: w.get_status() for cid, w in self.workers.items()},
                "event_queue": self.writer.queue.qsize() if self.writer else 0,
                "stream_dropped": {cid: w.dropped for cid, w in self.streamers.items()},
                "camera_metrics": metrics}

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
            if self.stream and config.output:
                self.streamers[cid] = stream(camera_id=cid, **config.output)
                self.streamers[cid].start()
            self.workers[cid] = camera(config.capture)
            self.workers[cid].start()
            if self.display:
                cv2.namedWindow(cid, cv2.WINDOW_AUTOSIZE)

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
        # Dùng cùng chính sách lấy frame để so sánh hai chế độ công bằng.
        ids = list(self.workers)
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
        self._used = True
        self.running = True
        logger.info("Starting pipeline: cameras=%d model_mode=%s batch_size=%d display=%s stream=%s",
                    len(self.settings), self.model_mode, self.runtime["batch_size"], self.display, self.stream)
        try:
            self._initialize()
            deadline = time.monotonic() + duration if duration is not None else float("inf")
            next_status = time.monotonic() + self.status_interval_seconds
            while not self.stop_event.is_set() and time.monotonic() < deadline:
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
                            cv2.imshow(tracking.camera_id, resize_frame(annotated))
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
        except Exception:
            logger.exception("Pipeline failed; stopping workers")
            raise
        finally:
            self.running = False
            self._close()

    start = run

    def _close(self):
        self.stop()
        logger.info("Stopping pipeline and draining event queue")
        errors = []
        for worker in self.workers.values():
            worker.stop()
        deadline = time.monotonic() + 8
        for worker in self.workers.values():
            if worker.thread is not None:
                worker.thread.join(timeout=max(0, deadline - time.monotonic()))
                if worker.thread.is_alive():
                    errors.append(RuntimeError("Capture thread did not stop before deadline"))
        for sink in [*self.streamers.values(), self.writer]:
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
    parser = argparse.ArgumentParser(description="Multi-camera detection, tracking, events and RTSP")
    parser.add_argument("--cameras", type=Path, default=CAMERA_CONFIG)
    parser.add_argument("--model", type=Path, default=MODEL_CONFIG)
    parser.add_argument("--model-mode", choices=MODEL_MODES)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--log-config", type=Path, default=DEFAULT_LOG_CONFIG)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    controller = configure_logging(args.log_config, level=args.log_level, log_file=args.log_file)
    try:
        if args.validate_config:
            cameras, _, _ = load_config(args.cameras, args.model)
            logger.info("Configuration valid: %d cameras", len(cameras))
            return
        manager = SystemManager(args.cameras, args.model, display=args.display, stream=not args.no_stream,
                                status_interval_seconds=controller.status_interval_seconds,
                                model_mode=args.model_mode)
        previous = {sig: signal.signal(sig, lambda *_: manager.stop()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            manager.run(duration=args.duration)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    except Exception:
        logger.exception("Application failed")
        raise SystemExit(1) from None
    finally:
        controller.close()


if __name__ == "__main__":
    main()
