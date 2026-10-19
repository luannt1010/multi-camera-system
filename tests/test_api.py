"""Kiểm tra API bằng manager giả, không mở camera hoặc nạp model."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from src import api


class FakeManager:
    def __init__(self, path, *, startup="ready"):
        self.runtime = {"database": path}
        self.ready_event = threading.Event()
        self.stop_event = threading.Event()
        self.finished = threading.Event()
        self.running = False
        self.error = None
        self.startup = startup
        self.stop_calls = 0
        self.control_calls = []
        self.enabled = {"cam_01": True, "cam_02": True}
        self.control_error = None

    def run(self):
        try:
            if self.startup == "error":
                raise RuntimeError("Lỗi nạp model")
            self.running = True
            if self.startup == "ready":
                self.ready_event.set()
            self.stop_event.wait()
        finally:
            self.running = False
            self.finished.set()

    def stop(self):
        self.stop_calls += 1
        self.stop_event.set()

    def get_camera_status(self, camera_id):
        enabled = self.enabled[camera_id]
        return {"camera_id": camera_id, "enabled": enabled,
                "status": "running" if enabled else "disabled",
                "metrics": {"capture_fps": 25.5, "processing_fps": 24.0}, "error": None}

    def list_cameras(self):
        return [self.get_camera_status(cid) for cid in self.enabled]

    def status(self):
        return {"running": self.running, "processed_frames": 100, "model_mode": "shared",
                "cameras": {cid: self.get_camera_status(cid)["status"] for cid in self.enabled},
                "event_queue": 0, "stream_dropped": {},
                "camera_metrics": {cid: self.get_camera_status(cid)["metrics"] for cid in self.enabled}}

    def set_camera_enabled(self, camera_id, enabled, timeout=20):
        self.control_calls.append((camera_id, enabled, timeout))
        if self.control_error:
            raise self.control_error
        self.enabled[camera_id] = enabled
        return self.get_camera_status(camera_id)


class FakeEventWriter:
    instances = []
    query_error = None

    def __init__(self, path):
        self.path = path
        self.close_calls = 0
        self.calls = []
        self.owner_thread = threading.get_ident()
        self.instances.append(self)

    def close(self):
        assert threading.get_ident() == self.owner_thread
        self.close_calls += 1

    def list_events(self, **filters):
        assert threading.get_ident() == self.owner_thread
        self.calls.append(filters)
        if self.query_error:
            raise self.query_error
        return {"items": [{"id": 7, "timestamp": "2026-10-17T10:00:00",
                           "camera_id": "cam_old", "track_id": 9, "event_type": "zone_entry",
                           "snapshot_base64": "Không gửi ảnh trong danh sách"}],
                "total": 1, "limit": filters["limit"], "offset": filters["offset"]}


@pytest.fixture
def api_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "EventWriter", FakeEventWriter)
    monkeypatch.setattr(FakeEventWriter, "instances", [])
    monkeypatch.setattr(FakeEventWriter, "query_error", None)
    manager = FakeManager(tmp_path / "events.sqlite3")
    manager.runtime["database"].touch()
    app = api.create_app(manager=manager, startup_timeout=1, shutdown_timeout=1, control_timeout=.5)
    return app, manager


def test_app_creation_does_not_create_manager_or_load_model():
    calls = []
    api.create_app(manager_factory=lambda *args, **kwargs: calls.append((args, kwargs)))
    assert calls == []


def test_lifespan_starts_and_stops_only_one_pipeline(api_setup):
    app, manager = api_setup
    assert not manager.running
    with TestClient(app) as client:
        assert manager.running
        assert app.state.database_path == manager.runtime["database"]
        assert FakeEventWriter.instances == []
        assert app.state.pipeline_thread.name == "SystemManager"
        assert client.get("/status").status_code == 200
    assert manager.stop_calls == 1
    assert FakeEventWriter.instances == []
    assert manager.finished.is_set()
    assert not app.state.pipeline_thread.is_alive()


def test_factory_receives_config_and_model_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "EventWriter", FakeEventWriter)
    manager = FakeManager(tmp_path / "events.db")
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return manager

    app = api.create_app("cameras.yaml", "model.yaml", manager_factory=factory,
                         model_mode="per_camera", stream=False, status_interval_seconds=12)
    assert calls == []
    with TestClient(app):
        assert calls == [(("cameras.yaml", "model.yaml"),
                          {"display": False, "stream": False, "model_mode": "per_camera",
                           "status_interval_seconds": 12})]


def test_status_and_camera_endpoints_expose_fps(api_setup):
    app, _ = api_setup
    with TestClient(app) as client:
        status = client.get("/status").json()
        assert status["running"]
        assert status["processed_frames"] == 100
        assert status["camera_metrics"]["cam_01"]["processing_fps"] == 24
        cameras = client.get("/cameras").json()
        assert [item["camera_id"] for item in cameras] == ["cam_01", "cam_02"]
        assert client.get("/cameras/cam_01").json()["metrics"]["capture_fps"] == 25.5


@pytest.mark.parametrize("method", ["get", "patch"])
def test_unknown_camera_returns_404(api_setup, method):
    app, _ = api_setup
    with TestClient(app) as client:
        kwargs = {"json": {"enabled": False}} if method == "patch" else {}
        assert getattr(client, method)("/cameras/missing", **kwargs).status_code == 404


def test_control_uses_requested_state_and_can_repeat(api_setup):
    app, manager = api_setup
    with TestClient(app) as client:
        for enabled in (False, False, True, True):
            response = client.patch("/cameras/cam_01", json={"enabled": enabled})
            assert response.status_code == 200
            assert response.json()["enabled"] is enabled
        assert len(manager.control_calls) == 4
        assert all(timeout == .5 for _, _, timeout in manager.control_calls)
        assert client.get("/cameras/cam_02").json()["enabled"] is True


@pytest.mark.parametrize("body", [{}, {"enabled": "false"}, {"enabled": 0},
                                  {"enabled": None}, {"enabled": True, "unknown": 1}])
def test_control_requires_boolean_body(api_setup, body):
    app, manager = api_setup
    with TestClient(app) as client:
        assert client.patch("/cameras/cam_01", json=body).status_code == 422
        assert manager.control_calls == []


@pytest.mark.parametrize("error,status", [(TimeoutError("pending"), 504),
                                          (RuntimeError("failed"), 503),
                                          (OSError("capture failed"), 503),
                                          (ValueError("invalid"), 422)])
def test_control_errors_have_clear_http_codes(api_setup, error, status):
    app, manager = api_setup
    manager.control_error = error
    with TestClient(app) as client:
        assert client.patch("/cameras/cam_01", json={"enabled": False}).status_code == status


def test_blocking_control_does_not_block_status_requests(api_setup):
    app, manager = api_setup
    entered, release = threading.Event(), threading.Event()
    original = manager.set_camera_enabled

    def slow_control(*args, **kwargs):
        entered.set()
        if not release.wait(3):
            raise TimeoutError("Test không nhận được tín hiệu tiếp tục")
        return original(*args, **kwargs)

    manager.set_camera_enabled = slow_control
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.patch, "/cameras/cam_01", json={"enabled": False})
        try:
            assert entered.wait(1)
            assert client.get("/status").status_code == 200
        finally:
            release.set()
        assert pending.result(timeout=2).status_code == 200


def test_pipeline_failure_remains_visible_and_history_still_works(api_setup):
    app, manager = api_setup
    with TestClient(app) as client:
        manager.error = RuntimeError("rtsp://admin:password@localhost/live?token=secret")
        manager.stop_event.set()
        assert manager.finished.wait(1)
        state = client.get("/status").json()
        assert state["running"] is False
        assert "REDACTED" in state["error"]
        assert "password" not in state["error"]
        assert "secret" not in state["error"]
        assert client.patch("/cameras/cam_01", json={"enabled": True}).status_code == 503
        assert client.get("/events").status_code == 200


def test_camera_errors_are_redacted_and_status_keeps_control_state(api_setup):
    app, manager = api_setup
    secret = "rtsp://admin:password@localhost/live?token=secret"
    original_camera = manager.get_camera_status
    original_status = manager.status

    def camera_state(cid):
        return {**original_camera(cid), "error": secret}

    def pipeline_state():
        return {**original_status(), "camera_enabled": {"cam_01": True, "cam_02": False},
                "camera_errors": {"cam_01": secret, "cam_02": None}}

    manager.get_camera_status = camera_state
    manager.status = pipeline_state
    with TestClient(app) as client:
        for url in ("/cameras", "/cameras/cam_01", "/status"):
            response = client.get(url)
            assert response.status_code == 200
            assert "password" not in response.text
            assert "secret" not in response.text
        assert client.get("/status").json()["camera_enabled"]["cam_02"] is False


def test_events_forward_local_time_filters_and_allow_historical_cameras(api_setup):
    app, _ = api_setup
    with TestClient(app) as client:
        response = client.get("/events", params={"camera_id": "cam_old",
                              "start_time": "2026-10-17T10:00:00",
                              "end_time": "2026-10-17T11:00:00", "limit": 10, "offset": 2})
        assert response.status_code == 200
        page = response.json()
        assert page["total"] == 1
        assert page["limit"] == 10
        assert page["offset"] == 2
        assert "snapshot_base64" not in page["items"][0]
        database = FakeEventWriter.instances[-1]
        assert database.close_calls == 1
        filters = database.calls[-1]
        assert filters["camera_id"] == "cam_old"
        assert filters["start_time"] == datetime(2026, 10, 17, 10)
        assert filters["end_time"] == datetime(2026, 10, 17, 11)
        assert filters["start_time"].tzinfo is None
        assert filters["end_time"].tzinfo is None


def test_api_reads_real_sqlite_history_without_a_writer_thread(tmp_path):
    from src.event_writer import EventWriter

    path = tmp_path / "events.sqlite3"
    writer = EventWriter(path)
    timestamp = "2026-10-17T10:00:00.123456"
    writer.save_one_record(timestamp, "cam_old", 5, "zone_entry", "ảnh")
    writer.save_one_record("2026-10-17T11:00:00", "cam_01", 6, "crossed_in", "ảnh")
    writer.close()
    app = api.create_app(manager=FakeManager(path))
    with TestClient(app) as client:
        response = client.get("/events", params={"camera_id": "cam_old",
                              "start_time": timestamp,
                              "end_time": timestamp})
        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert response.json()["items"][0]["camera_id"] == "cam_old"
        assert response.json()["items"][0]["timestamp"] == timestamp
        assert "snapshot_base64" not in response.json()["items"][0]


def test_api_does_not_create_database_when_events_have_not_started(tmp_path):
    path = tmp_path / "not-created" / "events.sqlite3"
    app = api.create_app(manager=FakeManager(path))
    with TestClient(app) as client:
        response = client.get("/events")
        assert response.status_code == 200
        assert response.json() == {"items": [], "total": 0, "limit": 100, "offset": 0}
        assert not path.parent.exists()
    assert not path.parent.exists()


@pytest.mark.parametrize("params", [
    {"start_time": "2026-10-17T10:00:00+07:00"}, {"end_time": "2026-10-17T10:00:00Z"},
    {"start_time": "not-a-date"}, {"limit": 0}, {"limit": 501}, {"offset": -1},
    {"camera_id": ""},
    {"start_time": "2026-10-18T00:00:00", "end_time": "2026-10-17T00:00:00"},
])
def test_event_invalid_filters_return_422_without_query(api_setup, params):
    app, _ = api_setup
    with TestClient(app) as client:
        assert client.get("/events", params=params).status_code == 422
        assert FakeEventWriter.instances == []


@pytest.mark.parametrize("error", [sqlite3.OperationalError("locked"), OSError("disk"),
                                  RuntimeError("unavailable")])
def test_event_store_unavailable_returns_503(api_setup, error):
    app, _ = api_setup
    FakeEventWriter.query_error = error
    with TestClient(app) as client:
        assert client.get("/events").status_code == 503
        assert FakeEventWriter.instances[-1].close_calls == 1


@pytest.mark.parametrize("error", [sqlite3.OperationalError("locked"), OSError("disk"),
                                  RuntimeError("unavailable")])
def test_event_database_open_failure_returns_503(api_setup, monkeypatch, error):
    app, _ = api_setup

    def fail_open(path):
        raise error

    monkeypatch.setattr(api, "EventWriter", fail_open)
    with TestClient(app) as client:
        assert client.get("/events").status_code == 503


def test_event_requests_open_and_close_separate_connections(api_setup):
    app, manager = api_setup
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(pool.map(lambda _: client.get("/events"), range(3)))
    assert all(response.status_code == 200 for response in responses)
    assert len(FakeEventWriter.instances) == 3
    assert all(database.path == manager.runtime["database"] and database.close_calls == 1
               for database in FakeEventWriter.instances)


def test_routes_without_lifespan_report_unavailable(api_setup):
    app, _ = api_setup
    client = TestClient(app)
    assert client.get("/status").status_code == 503
    assert client.get("/events").status_code == 503


@pytest.mark.parametrize("startup,message", [("error", "Lỗi nạp model"),
                                            ("not-ready", "Hết thời gian")])
def test_startup_failure_stops_background_thread(tmp_path, monkeypatch, startup, message):
    monkeypatch.setattr(api, "EventWriter", FakeEventWriter)
    manager = FakeManager(tmp_path / "events.db", startup=startup)
    app = api.create_app(manager=manager, startup_timeout=.05, shutdown_timeout=1)
    with pytest.raises(RuntimeError, match=message):
        with TestClient(app):
            pytest.fail("Ứng dụng không được phục vụ khi startup lỗi")
    assert manager.stop_calls == 1
    assert app.state.database_path == manager.runtime["database"]
    assert manager.finished.is_set()
    assert not app.state.pipeline_thread.is_alive()


@pytest.mark.parametrize("name,value", [("startup_timeout", 0), ("shutdown_timeout", -1),
                                       ("control_timeout", float("nan")),
                                       ("startup_timeout", float("inf"))])
def test_lifecycle_timeouts_must_be_positive_finite(name, value):
    with pytest.raises(ValueError, match=name):
        api.create_app(**{name: value})
