"""Application logging: bounded asynchronous dispatch, rotation and redaction."""
import atexit
import copy
from datetime import datetime
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import math
from pathlib import Path
from queue import Full, Queue
import re
import threading
from urllib.parse import urlsplit, urlunsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_CONFIG = ROOT / "configs/logging.yaml"
_active = None
_configure_lock = threading.RLock()
_url_pattern = re.compile(r"(?i)\b(?:rtsps?|https?)://[^\s<>\"']+")


def redact(text):
    """Hide URL credentials and queries in messages and exception tracebacks."""
    def replace(match):
        try:
            url = urlsplit(match.group())
            netloc = url.netloc.rsplit("@", 1)[-1]
            if "@" in url.netloc:
                netloc = "[REDACTED]@" + netloc
            return urlunsplit((url.scheme, netloc, url.path,
                               "[REDACTED]" if url.query else "", ""))
        except ValueError:
            return "[REDACTED_URL]"
    return _url_pattern.sub(replace, str(text))


class RedactingFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-8s %(name)s [%(threadName)s] "
                         "[camera=%(camera_id)s] %(message)s")

    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")

    def format(self, record):
        safe = copy.copy(record)
        safe.camera_id = getattr(record, "camera_id", "-")
        return redact(super().format(safe))


class _ApplicationQueueHandler(QueueHandler):
    def __init__(self, queue, sinks):
        super().__init__(queue)
        self.sinks = sinks
        self.dropped = 0

    def prepare(self, record):
        prepared = copy.copy(record)
        prepared.msg = record.getMessage()
        prepared.args = None
        return prepared

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except Full:
            if record.levelno >= logging.WARNING:
                # Preserve operational failures even if the disk cannot keep up.
                for sink in self.sinks:
                    if record.levelno >= sink.level:
                        sink.handle(record)
            else:
                self.dropped += 1


class _DrainingListener(QueueListener):
    def enqueue_sentinel(self):
        # A full bounded queue must drain before the shutdown sentinel is added.
        self.queue.put(self._sentinel)


class LoggingController:
    def __init__(self, logger, handler, listener, sinks, status_interval_seconds, previous):
        self.logger, self.handler, self.listener, self.sinks = logger, handler, listener, sinks
        self.status_interval_seconds = status_interval_seconds
        self.previous = previous
        self.closed = False

    def close(self):
        global _active
        with _configure_lock:
            if self.closed:
                return
            self.closed = True
            self.logger.removeHandler(self.handler)
            self.listener.stop()
            if self.handler.dropped:
                record = self.logger.makeRecord("src.logging_config", logging.WARNING, __file__, 0,
                    "Dropped %d DEBUG/INFO log records because the logging queue was full",
                    (self.handler.dropped,), None)
                for sink in self.sinks:
                    sink.handle(record)
            for sink in self.sinks:
                sink.flush()
                sink.close()
            self.handler.close()
            self.logger.setLevel(self.previous[0])
            self.logger.propagate = self.previous[1]
            if _active is self:
                _active = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def configure_logging(config_path=None, *, level=None, log_file=None):
    """Configure only the src namespace; the embedding app's root logger is untouched."""
    global _active
    path = Path(config_path) if config_path is not None else DEFAULT_LOG_CONFIG
    settings = {}
    if path.is_file():
        with path.open(encoding="utf-8-sig") as source:
            data = yaml.safe_load(source) or {}
        if not isinstance(data, dict) or not isinstance(data.get("logging", {}), dict):
            raise ValueError("Logging configuration must contain a logging mapping")
        settings = data.get("logging", {})
    elif config_path is not None:
        raise FileNotFoundError(f"Logging configuration not found: {path}")
    name = str(level if level is not None else settings.get("level", "INFO")).upper()
    if name not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"Invalid logging level: {name}")
    max_bytes = int(settings.get("max_bytes", 10 * 1024 * 1024))
    backup_count = int(settings.get("backup_count", 5))
    queue_size = int(settings.get("queue_size", 4096))
    interval = float(settings.get("status_interval_seconds", 30))
    if min(max_bytes, backup_count, queue_size) <= 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError("Log rotation sizes, queue_size and status_interval_seconds must be positive")
    console = settings.get("console", True)
    if not isinstance(console, bool):
        raise ValueError("logging.console must be true or false")
    output = log_file if log_file is not None else settings.get("file", "logs/system.log")
    if not console and not output:
        raise ValueError("Enable a logging console or file destination")
    formatter = RedactingFormatter()
    sinks = []
    try:
        if console:
            sinks.append(logging.StreamHandler())
        if output:
            file_path = Path(output).expanduser()
            if not file_path.is_absolute():
                file_path = ROOT / file_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            sinks.append(RotatingFileHandler(file_path, maxBytes=max_bytes,
                                            backupCount=backup_count, encoding="utf-8"))
        for sink in sinks:
            sink.setFormatter(formatter)
            sink.setLevel(name)
    except Exception:
        for sink in sinks:
            sink.close()
        raise
    with _configure_lock:
        if _active is not None:
            _active.close()
        logger = logging.getLogger("src")
        previous = logger.level, logger.propagate
        logger.setLevel(name)
        logger.propagate = False
        queue = Queue(maxsize=queue_size)
        handler = _ApplicationQueueHandler(queue, sinks)
        listener = _DrainingListener(queue, *sinks, respect_handler_level=True)
        listener.start()
        logger.addHandler(handler)
        _active = LoggingController(logger, handler, listener, sinks, interval, previous)
        return _active


def _shutdown():
    if _active is not None:
        _active.close()


atexit.register(_shutdown)
