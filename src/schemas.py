from dataclasses import dataclass, field
import numpy as np
from datetime import datetime

@dataclass
class FramePacket:
    camera_id: str
    timestamp: float
    frame: np.ndarray
    generation: int = field(default=0, kw_only=True)

@dataclass
class CameraConfig:
    camera_id: str
    source: str

@dataclass
class DetectionPacket(FramePacket):
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

@dataclass
class TrackingPacket(DetectionPacket):
    track_ids: np.ndarray

@dataclass
class Event:
    line: list[list[float]] | None = None
    polygon: list[list[float]] | None = None
    normalized: bool = False
    state_ttl: float = 60.0

    @property
    def enabled(self) -> bool:
        return self.line is not None or self.polygon is not None

@dataclass
class EventResults:
    camera_id: str
    timestamp: datetime
    frame: np.ndarray
    events: dict
    left_count: int
    right_count: int

@dataclass
class ModelConfig:
    weight_path: str
    imgsz: int
    device: str
    conf: float
    classes: list[int]
