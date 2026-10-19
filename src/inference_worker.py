import logging
import time

import numpy as np
from typing import List
from ultralytics import YOLO
from .schemas import FramePacket, ModelConfig, DetectionPacket


logger = logging.getLogger(__name__)


class InferenceWorker:
    def __init__(self, config: ModelConfig):
        self.config = config
        started = time.perf_counter()
        logger.info("Loading inference model (device=%s)", self.config.device)
        self.model = YOLO(self.config.weight_path)
        logger.info("Inference model loaded in %.2f seconds", time.perf_counter() - started)

    def predict(self, frame_packets: List[FramePacket]) -> List[DetectionPacket]:
        """Hàm predict theo batch"""
        output = []
        if not frame_packets:
            return output
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        started = time.perf_counter() if debug_enabled else None
        frames = [frame_packet.frame for frame_packet in frame_packets]
        results = self.model.predict(
            source=frames,
            conf=self.config.conf,
            classes=self.config.classes,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False
        )

        if len(results) != len(frame_packets):
            raise RuntimeError("Inference result count does not match input batch")

        for packet, result in zip(frame_packets, results):
            bbox_lst = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls_lst = result.boxes.cls.int().cpu().numpy()
            score_lst = result.boxes.conf.cpu().numpy()

            output.append(DetectionPacket(
                camera_id=packet.camera_id,
                generation=packet.generation,
                timestamp=packet.timestamp,
                frame=packet.frame,
                boxes=bbox_lst,
                scores=score_lst,
                class_ids=cls_lst
            ))
        if debug_enabled:
            logger.debug(
                "Inference batch completed: frames=%d detections=%d elapsed_ms=%.2f",
                len(frame_packets), sum(len(packet.boxes) for packet in output),
                (time.perf_counter() - started) * 1000,
            )
        return output

