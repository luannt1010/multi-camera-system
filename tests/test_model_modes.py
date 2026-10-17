"""Kiểm tra chọn model khi benchmark mà không cần GPU hoặc camera thật."""
from collections import deque
from types import SimpleNamespace
import time

import numpy as np
import pytest
import yaml

from src.config import CameraSettings
from src.schemas import CameraConfig, DetectionPacket, Event, FramePacket, TrackingPacket


def packet(camera_id, timestamp=100.0, generation=0):
    return FramePacket(camera_id, timestamp, np.zeros((8, 8, 3), np.uint8),
                       generation=generation)


class FakeCamera:
    def __init__(self, config):
        self.config = config
        self.frames = deque()
        self.running = False
        self.thread = None
        self.error = None

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def get_status(self):
        return "connected" if self.running else "stopped"

    def get_frame(self):
        return self.frames.popleft() if self.frames else None


class FakeTracker:
    def __init__(self, fps):
        self.resets = []
        self.received = []

    def reset(self, camera_id):
        self.resets.append(camera_id)

    def update(self, detection):
        self.received.append(detection)
        return TrackingPacket(
            detection.camera_id, detection.timestamp, detection.frame,
            detection.boxes, detection.scores, detection.class_ids,
            np.arange(len(detection.boxes), dtype=int), generation=detection.generation,
        )


class RecordingModel:
    def __init__(self, identity):
        self.identity = identity
        self.calls = []

    def predict(self, packets):
        self.calls.append(list(packets))
        return [DetectionPacket(
            item.camera_id, item.timestamp, item.frame,
            np.zeros((1, 4), np.float32), np.array([.9]),
            np.full(1, self.identity, dtype=int), generation=item.generation,
        ) for item in packets]


@pytest.fixture
def manager_factory(monkeypatch, tmp_path):
    import src.manager as module

    created = []
    managers = []
    runtime = {"batch_size": 2, "tracking_fps": 15, "event_queue_size": 8,
               "reset_after_seconds": 3, "database": tmp_path / "unused.db"}
    settings = [CameraSettings(CameraConfig(cid, "unused"), Event(), {})
                for cid in ("a", "b", "c")]

    def inference_factory(config):
        model = RecordingModel(len(created))
        created.append(model)
        return model

    monkeypatch.setattr(module, "load_config", lambda *_: (settings, SimpleNamespace(), dict(runtime)))

    def make(*, runtime_mode="shared", **kwargs):
        runtime["model_mode"] = runtime_mode
        manager = module.SystemManager(stream=False, inference_factory=inference_factory,
                                       camera_factory=FakeCamera, tracker_factory=FakeTracker,
                                       **kwargs)
        managers.append(manager)
        return manager, created

    yield make
    for manager in managers:
        manager._close()


@pytest.mark.parametrize("mode,expected_models", [("shared", 1), ("per_camera", 3)])
def test_initialization_loads_only_models_needed_by_selected_mode(manager_factory, mode, expected_models):
    manager, models = manager_factory(model_mode=mode)
    manager._initialize()
    assert len(models) == expected_models
    if mode == "shared":
        assert manager.inference is models[0]
        assert not manager.inferences
    else:
        assert manager.inference is None
        assert set(manager.inferences) == {"a", "b", "c"}
        assert len({id(model) for model in manager.inferences.values()}) == 3


def test_default_mode_keeps_existing_shared_model_behavior(manager_factory):
    manager, models = manager_factory()
    manager._initialize()
    assert manager.model_mode == "shared"
    assert manager.inference is models[0]
    assert len(models) == 1


def test_runtime_mode_is_used_when_no_explicit_override(manager_factory):
    manager, models = manager_factory(runtime_mode="per_camera")
    manager._initialize()
    assert manager.model_mode == "per_camera"
    assert len(models) == 3


def test_explicit_mode_overrides_runtime_setting(manager_factory):
    manager, models = manager_factory(runtime_mode="per_camera", model_mode="shared")
    manager._initialize()
    assert manager.model_mode == "shared"
    assert len(models) == 1


@pytest.mark.parametrize("mode", ["", "separate", "SHARED", 42])
def test_invalid_model_mode_is_rejected_before_models_load(manager_factory, mode):
    with pytest.raises(ValueError, match="model_mode"):
        manager_factory(model_mode=mode)


@pytest.fixture
def config_paths(tmp_path):
    weights = tmp_path / "fake.pt"
    weights.touch()
    cameras = tmp_path / "cameras.yaml"
    cameras.write_text(yaml.safe_dump({"cameras": [{"id": "a", "source": {"url": 0}}]}),
                       encoding="utf-8")
    model = tmp_path / "model.yaml"

    def write(runtime):
        model.write_text(yaml.safe_dump({"model": {"weights": str(weights)}, "runtime": runtime}),
                         encoding="utf-8")
        return cameras, model

    return write


@pytest.mark.parametrize("runtime,expected", [({}, "shared"), ({"model_mode": "shared"}, "shared"),
                                               ({"model_mode": "per_camera"}, "per_camera")])
def test_yaml_config_accepts_benchmark_modes_and_preserves_default(config_paths, runtime, expected):
    from src.config import load_config

    _, _, loaded = load_config(*config_paths(runtime))
    assert loaded["model_mode"] == expected


@pytest.mark.parametrize("mode", [None, "", "separate", "SHARED", 42])
def test_yaml_config_rejects_invalid_benchmark_mode(config_paths, mode):
    from src.config import load_config

    with pytest.raises(ValueError, match="model_mode"):
        load_config(*config_paths({"model_mode": mode}))


@pytest.mark.parametrize("mode", ["shared", "per_camera"])
def test_collectors_keep_batch_limit_and_rotate_cameras(manager_factory, monkeypatch, mode):
    import src.manager as module

    manager, _ = manager_factory(model_mode=mode)
    manager._initialize()
    monkeypatch.setattr(module.time, "perf_counter", lambda: 100.0)
    for cid, worker in manager.workers.items():
        worker.frames.extend([packet(cid), packet(cid)])
    batches = [[item.camera_id for item in manager._collect()] for _ in range(3)]
    assert batches == [["a", "b"], ["c", "a"], ["b", "c"]]


@pytest.mark.parametrize("mode", ["shared", "per_camera"])
def test_collectors_skip_stale_frames_without_blocking_other_cameras(manager_factory, monkeypatch, mode):
    import src.manager as module

    manager, _ = manager_factory(model_mode=mode)
    manager._initialize()
    monkeypatch.setattr(module.time, "perf_counter", lambda: 100.0)
    manager.workers["a"].frames.append(packet("a", timestamp=95.0))
    manager.workers["b"].frames.append(packet("b"))
    manager.workers["c"].frames.append(packet("c"))
    assert [item.camera_id for item in manager._collect()] == ["b", "c"]
    assert "a" not in manager.last_frame


@pytest.mark.parametrize("mode", ["shared", "per_camera"])
def test_reconnect_and_frame_gap_reset_only_affected_camera(manager_factory, monkeypatch, mode):
    import src.manager as module

    manager, _ = manager_factory(model_mode=mode)
    manager._initialize()
    clock = [100.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    worker = manager.workers["a"]
    worker.frames.append(packet("a", generation=1))
    manager._collect()
    worker.frames.append(packet("a", generation=2))
    manager._collect()
    clock[0] = 104.0
    worker.frames.append(packet("a", timestamp=104.0, generation=2))
    manager._collect()
    assert manager.tracker.resets == ["a", "a"]


@pytest.mark.parametrize("mode", ["shared", "per_camera"])
def test_empty_batch_does_not_invoke_any_model(manager_factory, mode):
    manager, models = manager_factory(model_mode=mode)
    manager._initialize()
    assert manager._predict([]) == ([], {})
    assert all(not model.calls for model in models)


def test_shared_prediction_uses_one_mixed_batch_and_preserves_packet_metadata(manager_factory, monkeypatch):
    import src.manager as module

    manager, models = manager_factory(model_mode="shared")
    manager._initialize()
    batch = [packet("b", 90.0, 2), packet("a", 91.0, 3), packet("b", 92.0, 4)]
    clock = iter([100.0, 102.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock, 110.0))
    detections, timings = manager._predict(batch)
    assert len(models[0].calls) == 1
    assert [id(item) for item in models[0].calls[0]] == [id(item) for item in batch]
    assert timings == {"a": (100.0, 102.0), "b": (100.0, 102.0)}
    for source, result in zip(batch, detections, strict=True):
        assert (result.camera_id, result.timestamp, result.generation) == (
            source.camera_id, source.timestamp, source.generation)
        assert result.frame is source.frame


def test_separate_prediction_groups_by_camera_and_restores_original_order(manager_factory, monkeypatch):
    import src.manager as module

    manager, _ = manager_factory(model_mode="per_camera")
    manager._initialize()
    batch = [packet("b", 90.0, 2), packet("a", 91.0, 3),
             packet("b", 92.0, 4), packet("c", 93.0, 5)]
    clock = iter([100.0, 101.0, 102.0, 106.0, 107.0, 109.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock, 110.0))
    detections, timings = manager._predict(batch)
    assert timings == {"b": (100.0, 101.0), "a": (102.0, 106.0), "c": (107.0, 109.0)}
    for cid, model in manager.inferences.items():
        assert len(model.calls) == 1
        expected = [item for item in batch if item.camera_id == cid]
        assert [id(item) for item in model.calls[0]] == [id(item) for item in expected]
    for source, result in zip(batch, detections, strict=True):
        assert (result.camera_id, result.timestamp, result.generation) == (
            source.camera_id, source.timestamp, source.generation)
        assert result.frame is source.frame
        assert result.class_ids.tolist() == [manager.inferences[source.camera_id].identity]


@pytest.mark.parametrize("mode", ["shared", "per_camera"])
def test_prediction_count_mismatch_is_reported_before_tracking(manager_factory, monkeypatch, mode):
    manager, models = manager_factory(model_mode=mode)
    manager._initialize()
    for model in models:
        monkeypatch.setattr(model, "predict", lambda _: [])
    with pytest.raises(RuntimeError, match="count"):
        manager._predict([packet("a")])
    assert not manager.tracker.received


@pytest.mark.parametrize("mode", ["shared", "per_camera"])
def test_prediction_cannot_be_assigned_to_another_camera(manager_factory, monkeypatch, mode):
    manager, models = manager_factory(model_mode=mode)
    manager._initialize()
    wrong_result = models[0].predict([packet("wrong")])
    for model in models:
        monkeypatch.setattr(model, "predict", lambda _: wrong_result)
    with pytest.raises(RuntimeError, match="camera"):
        manager._predict([packet("a")])
    assert not manager.tracker.received


@pytest.mark.parametrize("mode", ["shared", "per_camera"])
def test_run_routes_predictions_to_tracker_and_stops_all_workers(manager_factory, monkeypatch, mode):
    manager, models = manager_factory(model_mode=mode)
    original_collect = manager._collect

    def collect_once():
        for cid, worker in manager.workers.items():
            worker.frames.append(packet(cid, timestamp=time.perf_counter()))
        manager.stop()
        return original_collect()

    monkeypatch.setattr(manager, "_collect", collect_once)
    manager.run(duration=1)
    assert manager.processed == 2
    assert [item.camera_id for item in manager.tracker.received] == ["a", "b"]
    assert not manager.running
    assert all(not worker.running for worker in manager.workers.values())
    if mode == "shared":
        assert len(models[0].calls) == 1
        assert [item.camera_id for item in models[0].calls[0]] == ["a", "b"]
    else:
        assert len(manager.inferences["a"].calls) == len(manager.inferences["b"].calls) == 1
        assert not manager.inferences["c"].calls


@pytest.mark.parametrize("mode", ["shared", "per_camera"])
def test_cli_passes_selected_mode_to_manager(monkeypatch, mode):
    import src.manager as module

    calls = {}

    class Manager:
        def __init__(self, *args, **kwargs):
            calls.update(kwargs)

        def stop(self):
            pass

        def run(self, duration):
            calls["duration"] = duration

    controller = SimpleNamespace(status_interval_seconds=30, close=lambda: None)
    monkeypatch.setattr(module, "SystemManager", Manager)
    monkeypatch.setattr(module, "configure_logging", lambda *args, **kwargs: controller)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(module.sys, "argv", ["manager", "--model-mode", mode,
                                           "--no-stream", "--duration", "1"])
    module.main()
    assert calls["model_mode"] == mode
    assert calls["stream"] is False
    assert calls["duration"] == 1
