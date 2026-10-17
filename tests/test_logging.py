import ast
import io
import logging
import re
from pathlib import Path
from queue import Queue
import threading
from types import SimpleNamespace

import pytest
import yaml

from src.config import ROOT
from src.logging_config import configure_logging


@pytest.fixture
def setup_logging(tmp_path):
    """Keep every test's sinks isolated and always release listener threads."""
    controllers = []
    sequence = 0

    def configure(*, overrides=None, level=None, log_file=None):
        nonlocal sequence
        sequence += 1
        output = tmp_path / f"system-{sequence}.log"
        settings = {"level": "INFO", "console": False, "file": str(output),
                    "max_bytes": 10485760, "backup_count": 5,
                    "status_interval_seconds": 30, "queue_size": 4096}
        settings.update(overrides or {})
        config = tmp_path / f"logging-{sequence}.yaml"
        config.write_text(yaml.safe_dump({"logging": settings}), encoding="utf-8")
        controller = configure_logging(config, level=level, log_file=log_file)
        controllers.append(controller)
        return controller, Path(log_file) if log_file else output

    yield configure
    for controller in reversed(controllers):
        controller.close()


def test_file_and_console_respect_level(setup_logging, capsys):
    controller, path = setup_logging(overrides={"level": "WARNING", "console": True})
    logger = logging.getLogger("src.camera_worker")
    logger.debug("hidden-debug")
    logger.info("hidden-info")
    logger.warning("camera-warning", extra={"camera_id": "cam_01"})
    logger.error("camera-error")
    controller.close()

    console = capsys.readouterr()
    assert not console.out
    saved = path.read_text(encoding="utf-8")
    for text in (console.err, saved):
        assert "hidden-debug" not in text and "hidden-info" not in text
        assert text.count("camera-warning") == 1
        assert text.count("camera-error") == 1
        assert "WARNING" in text and "ERROR" in text
        assert "src.camera_worker" in text and "MainThread" in text
        assert "cam_01" in text
        # Every timestamp includes an explicit UTC offset or UTC marker.
        assert re.search(r"\d{4}-\d{2}-\d{2}.*(?:[+-]\d{2}:?\d{2}|Z)", text)


def test_missing_camera_uses_default_and_utf8_is_preserved(setup_logging):
    controller, path = setup_logging()
    logging.getLogger("src.event_writer").info("Đã lưu sự kiện")
    controller.close()
    text = path.read_text(encoding="utf-8")
    assert "Đã lưu sự kiện" in text
    assert re.search(r"(?:camera(?:_id)?[=: ]+|\[)-(?:\s|\]|$)", text) or " - " in text


def test_cli_overrides_yaml_level_and_file(setup_logging, tmp_path):
    override_path = tmp_path / "overridden" / "cli.log"
    controller, path = setup_logging(overrides={"level": "ERROR"},
                                     level="DEBUG", log_file=override_path)
    logging.getLogger("src.manager").debug("cli-level-takes-effect")
    controller.close()
    assert path == override_path
    assert "cli-level-takes-effect" in path.read_text(encoding="utf-8")
    assert not (tmp_path / "system-1.log").exists()


@pytest.mark.parametrize("setting,value", [
    ("level", "NOT_A_LEVEL"),
    ("max_bytes", 0),
    ("max_bytes", -1),
    ("backup_count", -1),
    ("queue_size", 0),
    ("queue_size", -1),
    ("status_interval_seconds", -1),
])
def test_invalid_configuration_fails_at_startup(setup_logging, setting, value):
    with pytest.raises(ValueError):
        setup_logging(overrides={setting: value})


def test_status_interval_is_exposed(setup_logging):
    controller, _ = setup_logging(overrides={"status_interval_seconds": 7.5})
    assert controller.status_interval_seconds == 7.5


def test_rotation_keeps_configured_backups(setup_logging):
    controller, path = setup_logging(overrides={"max_bytes": 300, "backup_count": 2})
    logger = logging.getLogger("src.manager")
    for index in range(30):
        logger.info("rotation-message-%02d %s", index, "x" * 80)
    controller.close()
    assert path.exists()
    assert path.with_name(path.name + ".1").exists()
    assert path.with_name(path.name + ".2").exists()
    assert not path.with_name(path.name + ".3").exists()
    assert "rotation-message-29" in path.read_text(encoding="utf-8")


def test_reconfigure_drains_old_sink_and_does_not_duplicate(setup_logging):
    first, first_path = setup_logging()
    logger = logging.getLogger("src.manager")
    logger.info("first-controller-message")
    second, second_path = setup_logging()
    logger.info("second-controller-message")
    first.close()  # Closing an already replaced controller must be harmless.
    logger.info("second-controller-still-active")
    second.close()
    first_text = first_path.read_text(encoding="utf-8")
    second_text = second_path.read_text(encoding="utf-8")
    assert first_text.count("first-controller-message") == 1
    assert "second-controller" not in first_text
    assert second_text.count("second-controller-message") == 1
    assert second_text.count("second-controller-still-active") == 1


def test_redacts_rtsp_credentials_queries_and_tracebacks(setup_logging):
    controller, path = setup_logging()
    logger = logging.getLogger("src.stream_ouput")
    source = "rtsp://alice:s3cr3t@camera.local:554/live?token=secret-token&password=query-password"
    logger.warning("Failed opening %s", source)
    try:
        raise RuntimeError(f"Cannot publish to {source}")
    except RuntimeError:
        logger.exception("Output failed for %s", source)
    controller.close()
    text = path.read_text(encoding="utf-8")
    assert "Traceback" in text and "RuntimeError" in text
    assert "camera.local:554/live" in text
    for secret in ("alice", "s3cr3t", "secret-token", "query-password"):
        assert secret not in text


def test_redaction_does_not_mutate_record_for_other_handlers(setup_logging):
    controller, path = setup_logging()
    logger = logging.getLogger("src.stream_ouput")
    records = []

    class Observer(logging.Handler):
        def emit(self, record):
            records.append(record)

    observer = Observer()
    logger.addHandler(observer)
    source = "rtsp://test-user:test-password@camera.local/live?token=secret-token"
    try:
        try:
            raise RuntimeError(source)
        except RuntimeError:
            logger.exception("Failed %s", source)
        controller.close()
        assert len(records) == 1
        assert records[0].msg == "Failed %s"
        assert records[0].args == (source,)
        assert source in records[0].getMessage()
        assert str(records[0].exc_info[1]) == source
        assert "test-password" not in path.read_text(encoding="utf-8")
    finally:
        logger.removeHandler(observer)


def test_close_drains_queue_and_releases_file(setup_logging):
    controller, path = setup_logging()
    logger = logging.getLogger("src.manager")
    for index in range(1000):
        logger.info("shutdown-drain-%04d", index)
    controller.close()
    controller.close()
    text = path.read_text(encoding="utf-8")
    assert text.count("shutdown-drain-") == 1000
    assert "shutdown-drain-0999" in text
    # Windows refuses this deletion while a file handler still owns the file.
    path.unlink()
    assert not path.exists()


def test_queue_overflow_preserves_warning_and_reports_dropped_info(setup_logging):
    controller, path = setup_logging(overrides={"queue_size": 1})
    entered, release = threading.Event(), threading.Event()

    class StallFirstMessage(logging.Filter):
        def filter(self, record):
            if record.getMessage() == "listener-blocker":
                entered.set()
                release.wait(timeout=5)
            return True

    controller.sinks[0].addFilter(StallFirstMessage())
    logger = logging.getLogger("src.manager")
    try:
        logger.info("listener-blocker")
        assert entered.wait(timeout=2), "Log listener never received the first message"
        logger.info("queued-info")
        logger.info("dropped-info")
        logger.warning("preserved-overflow-warning")
        assert controller.handler.dropped == 1
    finally:
        release.set()
        controller.close()
    text = path.read_text(encoding="utf-8")
    assert "listener-blocker" in text and "queued-info" in text
    assert "dropped-info" not in text
    assert text.count("preserved-overflow-warning") == 1
    assert re.search(r"Dropped 1 .*log records", text)


def test_configuration_does_not_modify_root_logging(setup_logging):
    root = logging.getLogger()
    handlers, level = tuple(root.handlers), root.level
    controller, _ = setup_logging()
    assert tuple(root.handlers) == handlers
    assert root.level == level
    controller.close()
    assert tuple(root.handlers) == handlers
    assert root.level == level


def test_ffmpeg_pipe_severity_camera_context_and_redaction(setup_logging):
    from src.stream_ouput import _read_ffmpeg_log

    controller, path = setup_logging()
    process = SimpleNamespace(camera_id="stream-cam", stderr=io.BytesIO(
        b"[warning] output timestamps are discontinuous\n"
        b"[error] Failed publishing rtsp://ffmpeg-user:ffmpeg-password@camera.local/live?token=ffmpeg-token\n"
        b"Broken pipe\n"))
    _read_ffmpeg_log(process)
    controller.close()
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if "FFmpeg:" in line]
    assert len(lines) == 3
    assert "WARNING" in lines[0]
    assert "ERROR" in lines[1] and "ERROR" in lines[2]
    assert all("camera=stream-cam" in line for line in lines)
    assert "camera.local/live" in text
    for secret in ("ffmpeg-user", "ffmpeg-password", "ffmpeg-token"):
        assert secret not in text


def test_ffmpeg_repeated_warnings_are_summarized_at_end(setup_logging):
    from src.stream_ouput import _read_ffmpeg_log

    controller, path = setup_logging()
    process = SimpleNamespace(camera_id="noisy-cam", stderr=io.BytesIO(
        b"[warning] encoder falling behind\n" * 4))
    _read_ffmpeg_log(process)
    controller.close()
    text = path.read_text(encoding="utf-8")
    assert text.count("FFmpeg: [warning] encoder falling behind") == 1
    assert text.count("Suppressed 3 repeated FFmpeg messages") == 1
    assert "camera=noisy-cam" in text


def test_ffmpeg_warning_summary_is_emitted_before_next_message(setup_logging):
    from src.stream_ouput import _read_ffmpeg_log

    controller, path = setup_logging()
    process = SimpleNamespace(stderr=io.BytesIO(
        b"[warning] first message\n[warning] first message\n[error] stream failed\n"))
    _read_ffmpeg_log(process)
    controller.close()
    text = path.read_text(encoding="utf-8")
    assert text.index("Suppressed 1 repeated FFmpeg messages") < text.index("FFmpeg: [error] stream failed")
    assert text.count("FFmpeg: [warning] first message") == 1
    assert "camera=-" in text


def test_manager_logs_lifecycle_and_periodic_status(setup_logging, monkeypatch):
    import src.manager as module

    controller, path = setup_logging()
    monkeypatch.setattr(module, "load_config", lambda *_: ([], None, {"batch_size": 1}))
    manager = module.SystemManager(status_interval_seconds=.001)
    closed = []
    writer = SimpleNamespace(queue=Queue(), check=lambda: None,
                             close=lambda: closed.append(True))
    monkeypatch.setattr(manager, "_initialize", lambda: setattr(manager, "writer", writer))
    manager.run(duration=.015)
    controller.close()
    text = path.read_text(encoding="utf-8")
    for message in ("Starting pipeline:", "Pipeline status:",
                    "Stopping pipeline and draining event queue", "Pipeline stopped:"):
        assert message in text
    assert closed == [True]
    assert not manager.running


def test_manager_initialization_failure_is_logged_and_cleanup_runs(setup_logging, monkeypatch):
    import src.manager as module

    controller, path = setup_logging()
    monkeypatch.setattr(module, "load_config", lambda *_: ([], None, {"batch_size": 1}))
    manager = module.SystemManager()

    def fail_initialization():
        raise RuntimeError("model initialization failed")

    monkeypatch.setattr(manager, "_initialize", fail_initialization)
    with pytest.raises(RuntimeError, match="model initialization failed"):
        manager.run(duration=.01)
    controller.close()
    text = path.read_text(encoding="utf-8")
    assert "ERROR" in text and "Pipeline failed; stopping workers" in text
    assert "Traceback" in text and "model initialization failed" in text
    assert "Stopping pipeline and draining event queue" in text
    assert "Pipeline stopped:" in text
    assert not manager.running and manager.stop_event.is_set()


def test_cli_failure_closes_logging_after_recording_traceback(setup_logging, monkeypatch):
    import src.manager as module

    controller, path = setup_logging()
    monkeypatch.setattr(module, "configure_logging", lambda *_, **__: controller)
    monkeypatch.setattr(module.sys, "argv", ["manager", "--validate-config"])

    def invalid_config(*_):
        raise ValueError("invalid camera configuration")

    monkeypatch.setattr(module, "load_config", invalid_config)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    assert controller.closed
    text = path.read_text(encoding="utf-8")
    assert "Application failed" in text
    assert "Traceback" in text and "invalid camera configuration" in text


def test_application_source_has_no_print_calls():
    violations = []
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "yolo").glob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "Use module logging instead of print: " + ", ".join(violations)
