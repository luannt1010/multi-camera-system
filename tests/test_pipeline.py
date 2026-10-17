import base64
import sqlite3
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from src.camera_worker import CameraWorker
from src.config import ROOT, load_config
from src.event_engine import EventEngine
from src.event_writer import EventWriter
from src.manager import SystemManager
from src.output_workers import StreamWorker, WriterWorker
from src.schemas import CameraConfig, DetectionPacket, Event, FramePacket, TrackingPacket
from src.tracker import Tracker
from src.utils import draw_results


def tracked(y, x=50, tid=1, timestamp=1, camera='a'):
    return TrackingPacket(camera, timestamp, np.zeros((100, 100, 3), np.uint8),
                          np.array([[x-5, y-5, x+5, y+5]], np.float32),
                          np.array([.9]), np.array([0]), np.array([tid]))


def engine():
    return EventEngine(Event([[.1, .5], [.9, .5]], [[.1,.3],[.9,.3],[.9,.9],[.1,.9]], True))


def configs(tmp_path, count=2):
    cameras = {'cameras': [{'id': str(i), 'source': {'url': 0},
        'lines': {'point': [[.1,.5],[.9,.5]]},
        'prohibit_area': {'polygon': [[.1,.3],[.9,.3],[.9,.9],[.1,.9]]}}
        for i in range(count)]}
    model = {'model': {'weights': str(ROOT/'src/yolov8n.pt'), 'device': 'cpu'},
             'runtime': {'database': str(tmp_path/'events.sqlite3'), 'batch_size': 1}}
    a, b = tmp_path/'cameras.yaml', tmp_path/'model.yaml'
    a.write_text(yaml.safe_dump(cameras)); b.write_text(yaml.safe_dump(model))
    return a, b


def test_geometry_crossing_entry_and_segment():
    e = engine()
    assert e.process(tracked(40)).events[1]['zone_entry']
    assert not e.process(tracked(45)).events[1]['zone_entry']
    assert not e.process(tracked(50)).events[1]['crossed']
    result = e.process(tracked(60))
    assert result.right_count == 1
    assert result.events[1]['direction'] == 'left_to_right'
    assert e.process(tracked(40)).left_count == 1
    e.process(tracked(10))
    assert e.process(tracked(40)).events[1]['zone_entry']
    outside = engine()
    outside.process(tracked(40, x=99))
    assert not outside.process(tracked(60, x=99)).events[1]['crossed']


def test_state_expiry_and_camera_isolation():
    e, other = engine(), engine()
    e.process(tracked(40))
    assert other.process(tracked(60)).right_count == 0
    assert e.process(tracked(60, timestamp=100)).right_count == 0
    assert len(e.last_seen) == 1
    e.reset()
    assert not e.last_seen


def test_drawing_does_not_mutate_snapshot():
    packet = tracked(40)
    e = engine(); result = e.process(packet)
    drawn = draw_results(packet, result, e.geometry(packet.frame.shape))
    assert drawn.any() and not result.frame.any()


def test_sqlite_and_snapshot(tmp_path):
    e = engine()
    writer = EventWriter(tmp_path/'nested'/'events.db')
    try:
        ids = writer.save(e.process(tracked(40)))
        assert len(ids) == 1
        row = writer.get(ids[0])
        image = cv2.imdecode(np.frombuffer(base64.b64decode(row['snapshot_base64']), np.uint8), 1)
        assert image.shape == (100,100,3)
        assert row['event_type'] == 'zone_entry'
        assert writer.save(e.process(tracked(45))) == []
        assert len(writer.save(e.process(tracked(60)))) == 1
    finally:
        writer.close()


def test_writer_drains_and_reports_failure(tmp_path):
    writer = WriterWorker(tmp_path/'events.db')
    writer.start()
    writer.submit(engine().process(tracked(40)))
    writer.close()
    with sqlite3.connect(tmp_path/'events.db') as db:
        assert db.execute('select count(*) from events').fetchone()[0] == 1
    broken = WriterWorker(tmp_path)  # a directory cannot be opened as a database
    with pytest.raises(RuntimeError, match='writer failed'):
        broken.start()
    with pytest.raises(RuntimeError):
        broken.close()
    assert not broken.thread.is_alive()


def test_writer_overload_is_explicit(tmp_path):
    writer = WriterWorker(tmp_path/'unused.db', capacity=1)
    result = engine().process(tracked(40))
    writer.submit(result)
    with pytest.raises(RuntimeError, match='queue full'):
        writer.submit(result)


def test_camera_latest_only_and_interruptible_reconnect(monkeypatch):
    worker = CameraWorker(CameraConfig('a', 'missing.mp4'))
    frame = np.zeros((10,10,3), np.uint8)
    worker._put_latest(frame); worker._put_latest(frame+1)
    assert worker.frame_queue.qsize() == 1
    assert worker.get_frame().frame.all()
    monkeypatch.setattr(worker, '_init_cap', lambda: False)
    worker.start(); worker.stop(); worker.thread.join(timeout=1)
    assert not worker.thread.is_alive()


def test_stream_queue_keeps_latest():
    worker = StreamWorker(output_url='rtsp://localhost:8554/test')
    worker.submit(1); worker.submit(2)
    assert worker.queue.get_nowait().frame == 2
    assert worker.dropped == 1
    worker.close()


def test_config_validation(tmp_path):
    a, b = configs(tmp_path)
    cameras, model, runtime = load_config(a, b)
    assert len(cameras) == 2 and model.device == 'cpu'
    data = yaml.safe_load(a.read_text())
    data['cameras'][1]['id'] = '0'
    a.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='Duplicate'):
        load_config(a, b)


@pytest.mark.parametrize('source,output', [
    ('rtsp://localhost:8554/live1.sdp', 'rtsp://localhost:8554/live1.sdp'),
    ('rtsp://localhost:8554/live1.sdp', 'rtsp://127.0.0.1:8554/live1.sdp'),
    ('rtsp://[::1]:8554/live1.sdp', 'rtsp://localhost:8554/live1.sdp'),
    ('rtsp://LOCALHOST/live1.sdp', 'rtsp://127.0.0.1:554/live1.sdp'),
    ('rtsp://reader:secret@localhost:8554/live1.sdp',
     'rtsp://writer:secret@127.0.0.1:8554/live1.sdp'),
])
def test_config_rejects_output_overwriting_source(tmp_path, source, output):
    a, b = configs(tmp_path)
    data = yaml.safe_load(a.read_text())
    data['cameras'][0]['source']['url'] = source
    data['cameras'][0]['output'] = {'rtsp_url': output}
    a.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='conflicts with a camera source'):
        load_config(a, b)


@pytest.mark.parametrize('source_index', [0, 1])
def test_config_rejects_overwriting_another_cameras_source(tmp_path, source_index):
    a, b = configs(tmp_path)
    data = yaml.safe_load(a.read_text())
    data['cameras'][source_index]['source']['url'] = 'rtsp://localhost:8554/live1.sdp'
    data['cameras'][1-source_index]['output'] = {'rtsp_url': 'rtsp://127.0.0.1:8554/live1.sdp'}
    a.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='conflicts with a camera source'):
        load_config(a, b)


def test_config_accepts_separate_output_paths(tmp_path):
    a, b = configs(tmp_path)
    data = yaml.safe_load(a.read_text())
    for index, camera in enumerate(data['cameras']):
        camera['source']['url'] = f'rtsp://localhost:8554/live{index}.sdp'
        camera['output'] = {'rtsp_url': f'rtsp://127.0.0.1:8554/output/cam_{index}'}
    a.write_text(yaml.safe_dump(data))
    cameras, _, _ = load_config(a, b)
    assert cameras[0].output['output_url'].endswith('/output/cam_0')
    assert cameras[1].capture.source.endswith('/live1.sdp')


def test_config_rejects_output_aliases_publishing_same_path(tmp_path):
    a, b = configs(tmp_path)
    data = yaml.safe_load(a.read_text())
    data['cameras'][0]['output'] = {'rtsp_url': 'rtsp://localhost:8554/output/cam'}
    data['cameras'][1]['output'] = {'rtsp_url': 'rtsp://127.0.0.1:8554/output/cam'}
    a.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='valid and unique'):
        load_config(a, b)


@pytest.mark.parametrize('source,output', [
    ('rtsp://localhost:8554/live', 'rtsp://localhost:8555/live'),
    ('rtsp://localhost:8554/channel?id=1', 'rtsp://localhost:8554/channel?id=2'),
    (0, 'rtsp://localhost:8554/output/cam'),
    ('input.mp4', 'rtsp://localhost:8554/output/cam'),
])
def test_config_accepts_distinct_streams_and_local_sources(tmp_path, source, output):
    a, b = configs(tmp_path, count=1)
    data = yaml.safe_load(a.read_text())
    data['cameras'][0]['source']['url'] = source
    data['cameras'][0]['output'] = {'rtsp_url': output}
    a.write_text(yaml.safe_dump(data))
    cameras, _, _ = load_config(a, b)
    assert cameras[0].output['output_url'] == output


def test_tracker_real_empty_and_isolated():
    tracker = Tracker()
    for cid in ('a', 'b'):
        packet = DetectionPacket(cid, 1, np.zeros((100,100,3),np.uint8),
                                  np.empty((0,4),np.float32), np.empty(0), np.empty(0,dtype=int))
        assert tracker.update(packet).boxes.shape == (0,4)
    assert tracker.trackers['a'] is not tracker.trackers['b']
    tracker.reset('a'); assert 'a' not in tracker.trackers


class FakeCamera:
    def __init__(self, config):
        self.config = config; self.thread = None; self.running = False
    def start(self): self.running = True
    def stop(self): self.running = False
    def get_status(self): return 'connected' if self.running else 'stopped'
    def get_frame(self):
        return FramePacket(self.config.camera_id, time.perf_counter(), np.zeros((100,100,3),np.uint8))


class FakeInference:
    def __init__(self, config): self.ids = []
    def predict(self, packets):
        self.ids.extend(p.camera_id for p in packets)
        return [DetectionPacket(p.camera_id, p.timestamp, p.frame, np.array([[40,35,60,45]],np.float32),
                                np.array([.9]), np.array([0])) for p in packets]


class FakeTracker:
    def __init__(self, fps): self.resets = []
    def update(self, p):
        return TrackingPacket(p.camera_id, p.timestamp, p.frame,p.boxes,p.scores,p.class_ids,np.array([1]))
    def reset(self, cid): self.resets.append(cid)


def test_manager_complete_flow_fairness_and_shutdown(tmp_path):
    a,b = configs(tmp_path)
    manager = SystemManager(a,b,stream=False, inference_factory=FakeInference,
                            camera_factory=FakeCamera, tracker_factory=FakeTracker)
    manager.run(duration=.1)
    assert set(manager.inference.ids) == {'0','1'}
    assert manager.inference.ids[:4] == ['0','1','0','1']
    assert manager.processed > 2 and not manager.running
    assert not manager.writer.thread.is_alive()
    assert all(not w.running for w in manager.workers.values())
    with sqlite3.connect(tmp_path/'events.sqlite3') as db:
        assert db.execute('select count(*) from events').fetchone()[0] == 2
    with pytest.raises(RuntimeError, match='new SystemManager'):
        manager.run()


def test_manager_cleanup_after_inference_failure(tmp_path):
    class Broken(FakeInference):
        def predict(self, packets): raise ValueError('inference failed')
    a,b = configs(tmp_path)
    manager = SystemManager(a,b,stream=False,inference_factory=Broken,camera_factory=FakeCamera)
    with pytest.raises(ValueError, match='inference failed'):
        manager.run(duration=.1)
    assert not manager.writer.thread.is_alive()
    assert all(not w.running for w in manager.workers.values())


def test_reconnect_resets_identity_even_without_long_gap(tmp_path):
    a,b = configs(tmp_path, count=1)
    manager = SystemManager(a,b,stream=False,inference_factory=FakeInference,
                            camera_factory=FakeCamera,tracker_factory=FakeTracker)
    try:
        manager._initialize()
        manager._collect()
        worker = manager.workers['0']
        worker.get_frame = lambda: FramePacket('0',time.perf_counter(),np.zeros((100,100,3),np.uint8),generation=2)
        manager._collect()
        assert manager.tracker.resets == ['0']
    finally:
        manager._close()


def test_ffmpeg_real_pipe_and_stalled_process(monkeypatch):
    import shutil
    import subprocess
    import sys
    import src.stream_ouput as module
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg unavailable')
    processes = []
    def start_null(url, width, height, fps, camera_id="-"):
        process = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
            '-i', 'pipe:0', '-f', 'null', '-'], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        processes.append(process)
        return process
    monkeypatch.setattr(module, 'start_stream', start_null)
    stream = module.StreamOutput('unused', width=320, height=240)
    try:
        assert stream.send(np.zeros((100,200,3),np.uint8))
        stream.process.stdin.close()
        assert stream.process.wait(timeout=5) == 0
    finally:
        stream.close()
    def start_stalled(*args, **kwargs):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
            stdin=subprocess.PIPE, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        processes.append(process)
        return process
    monkeypatch.setattr(module, 'start_stream', start_stalled)
    stream = module.StreamOutput('unused',width=320,height=240)
    started = time.monotonic()
    try:
        assert not stream.send(np.zeros((240,320,3),np.uint8))
        assert time.monotonic()-started < 8
    finally:
        stream.close()
    assert all(p.poll() is not None for p in processes)
