"""Kiểm tra chạy API và preview cùng một manager, không mở cửa sổ thật."""
import logging
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
import uvicorn

from src import api


class PreviewManager:
    def __init__(self, path, *, startup_error=False):
        self.runtime = {"database": path}
        self.ready_event = threading.Event()
        self.stop_event = threading.Event()
        self.startup_error = startup_error
        self.running = False
        self.error = None
        self.enabled = True
        self.stop_calls = 0
        self.run_thread = None

    def run(self):
        self.run_thread = threading.get_ident()
        try:
            if self.startup_error:
                raise RuntimeError("Không mở được preview")
            self.running = True
            self.ready_event.set()
            self.stop_event.wait()
        finally:
            self.running = False

    def stop(self):
        self.stop_calls += 1
        self.stop_event.set()

    def status(self):
        return {"running": self.running, "processed_frames": 7, "model_mode": "shared",
                "cameras": {"cam_01": "running" if self.enabled else "disabled"},
                "event_queue": 0, "stream_dropped": {},
                "camera_metrics": {"cam_01": {"processing_fps": 24.5}}}

    def get_camera_status(self, camera_id):
        assert camera_id == "cam_01"
        return {"camera_id": camera_id, "enabled": self.enabled,
                "status": "running" if self.enabled else "disabled",
                "metrics": {"processing_fps": 24.5}, "error": None}

    def set_camera_enabled(self, camera_id, enabled, timeout):
        self.enabled = enabled
        return self.get_camera_status(camera_id)


@pytest.mark.parametrize("display", [False, True])
def test_api_display_flag_reaches_the_same_single_manager(tmp_path, display):
    manager = PreviewManager(tmp_path / "events.sqlite3")
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return manager

    options = {"display": True} if display else {}
    application = api.create_app("camera.yaml", "model.yaml", manager_factory=factory,
                                 startup_timeout=1, shutdown_timeout=1, **options)
    assert not calls
    with TestClient(application) as client:
        assert len(calls) == 1
        assert calls[0][0] == ("camera.yaml", "model.yaml")
        assert calls[0][1]["display"] is display
        assert application.state.manager is manager
        assert manager.run_thread != threading.get_ident()
        assert client.get("/status").json()["camera_metrics"]["cam_01"]["processing_fps"] == 24.5
        response = client.patch("/cameras/cam_01", json={"enabled": False})
        assert response.status_code == 200
        assert response.json()["enabled"] is False
        assert manager.enabled is False
    assert manager.stop_calls == 1
    assert not application.state.pipeline_thread.is_alive()


@pytest.mark.parametrize("startup_error", [False, True])
def test_pipeline_stopped_callback_runs_once_on_normal_stop_or_error(tmp_path, startup_error):
    manager = PreviewManager(tmp_path / "events.sqlite3", startup_error=startup_error)
    callback_threads = []
    stopped = threading.Event()

    def on_stopped():
        callback_threads.append(threading.get_ident())
        stopped.set()

    application = api.create_app(manager=manager, display=True,
                                 on_pipeline_stopped=on_stopped,
                                 startup_timeout=1, shutdown_timeout=1)
    if startup_error:
        with pytest.raises(RuntimeError, match="Không mở được preview"):
            with TestClient(application):
                pytest.fail("Startup lỗi phải ngăn API bắt đầu phục vụ")
    else:
        with TestClient(application):
            # Mô phỏng manager dừng khi người dùng đóng preview hoặc nhấn Q.
            manager.stop_event.set()
            assert stopped.wait(1)
    assert stopped.is_set()
    assert callback_threads == [manager.run_thread]
    assert manager.stop_calls == 1
    assert not application.state.pipeline_thread.is_alive()


@pytest.mark.parametrize("display,outcome", [(False, "ready"), (True, "ready"),
                                            (False, "error"), (True, "error"),
                                            (True, "not-started"), (True, "interrupt")])
def test_cli_display_controls_server_and_restores_logging(monkeypatch, display, outcome):
    closed = []
    controller = SimpleNamespace(status_interval_seconds=15, close=lambda: closed.append(True))
    monkeypatch.setattr(api, "configure_logging", lambda *args, **kwargs: controller)
    argv = ["api", "--host", "0.0.0.0", "--port", "8765", "--no-stream",
            "--cameras", "camera.yaml", "--model", "model.yaml", "--model-mode", "per_camera"]
    if display:
        argv.append("--display")
    monkeypatch.setattr(sys, "argv", argv)
    application = object()
    app_calls = []
    servers = []
    configs = []

    def create_app(*args, **kwargs):
        app_calls.append((args, kwargs))
        return application

    def config(app, **kwargs):
        configs.append((app, kwargs))
        return SimpleNamespace(app=app)

    class Server:
        def __init__(self, config):
            self.should_exit = False
            self.started = outcome != "not-started"
            servers.append(self)

        def run(self):
            callback = app_calls[0][1].get("on_pipeline_stopped")
            if display:
                assert callable(callback)
                if self.started:
                    callback()
                    assert self.should_exit is True
            else:
                assert callback is None
                assert self.should_exit is False
            if outcome == "error":
                raise RuntimeError("Không mở được HTTP server")
            if outcome == "interrupt":
                raise KeyboardInterrupt

    monkeypatch.setattr(api, "create_app", create_app)
    monkeypatch.setattr(uvicorn, "Config", config)
    monkeypatch.setattr(uvicorn, "Server", Server)
    previous = {name: (logger.handlers[:], logger.level, logger.propagate)
                for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
                for logger in [logging.getLogger(name)]}
    if outcome == "error":
        with pytest.raises(RuntimeError, match="Không mở được HTTP server"):
            api.main()
    elif outcome == "not-started":
        with pytest.raises(SystemExit) as error:
            api.main()
        assert error.value.code == 3
    else:
        api.main()
    assert len(servers) == len(app_calls) == 1
    assert configs == [(application, {"host": "0.0.0.0", "port": 8765,
                                     "workers": 1, "log_config": None})]
    args, kwargs = app_calls[0]
    assert args == (Path("camera.yaml"), Path("model.yaml"))
    assert kwargs["display"] is display
    assert kwargs["stream"] is False
    assert kwargs["model_mode"] == "per_camera"
    assert kwargs["status_interval_seconds"] == 15
    assert closed == [True]
    for name, original in previous.items():
        logger = logging.getLogger(name)
        assert (logger.handlers, logger.level, logger.propagate) == original
