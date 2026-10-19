import argparse
from contextlib import asynccontextmanager, closing
import logging
import math
from pathlib import Path
import sqlite3
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime, StrictBool, field_validator
from starlette.concurrency import run_in_threadpool

from .event_writer import EventWriter
from .logging_config import redact

logger = logging.getLogger("src.api")


class CameraState(BaseModel):
    camera_id: str
    enabled: bool
    status: str
    metrics: dict[str, Any]
    error: str | None = None

    @field_validator("error")
    @classmethod
    def hide_credentials(cls, value):
        return redact(value) if value is not None else None


class CameraUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


class PipelineState(BaseModel):
    running: bool
    processed_frames: int
    model_mode: str
    cameras: dict[str, str]
    event_queue: int
    stream_dropped: dict[str, int]
    camera_metrics: dict[str, dict[str, Any]]
    camera_enabled: dict[str, bool] = Field(default_factory=dict)
    camera_errors: dict[str, str | None] = Field(default_factory=dict)
    error: str | None = None

    @field_validator("error")
    @classmethod
    def hide_credentials(cls, value):
        return redact(value) if value is not None else None

    @field_validator("camera_errors")
    @classmethod
    def hide_camera_credentials(cls, value):
        return {cid: redact(error) if error is not None else None for cid, error in value.items()}


class EventItem(BaseModel):
    id: int
    timestamp: str
    camera_id: str
    track_id: int
    event_type: str


class EventPage(BaseModel):
    items: list[EventItem]
    total: int
    limit: int
    offset: int


def create_app(camera_config: str, model_config: str,
               manager_factory, manager=None, display=False, stream=True, model_mode=None,
               startup_timeout=120, shutdown_timeout=60, control_timeout=20,
               status_interval_seconds=30, on_pipeline_stopped=None):
    """Tạo ứng dụng; chỉ nạp model và mở camera khi lifespan bắt đầu."""
    for name, value in (("startup_timeout", startup_timeout),
                        ("shutdown_timeout", shutdown_timeout),
                        ("control_timeout", control_timeout)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} phải là số hữu hạn lớn hơn 0")

    @asynccontextmanager
    async def lifespan(application):
        active = manager
        if active is None:
            active = await run_in_threadpool(
                manager_factory, camera_config, model_config, display=display, stream=stream,
                model_mode=model_mode, status_interval_seconds=status_interval_seconds,
            )
        application.state.manager = active
        # Cùng lớp quản lý DB; API chỉ đọc bằng kết nối riêng cho mỗi truy vấn.
        application.state.database_path = Path(active.runtime["database"])
        application.state.run_error = None

        def run_pipeline():
            try:
                active.run()
            except Exception as exc:
                application.state.run_error = exc
                active.ready_event.set()
                logger.exception("Pipeline API dừng do lỗi")
            finally:
                if on_pipeline_stopped is not None:
                    on_pipeline_stopped()

        thread = threading.Thread(target=run_pipeline, name="SystemManager", daemon=True)
        application.state.pipeline_thread = thread
        thread.start()
        try:
            ready = await run_in_threadpool(active.ready_event.wait, startup_timeout)
            if not ready:
                raise RuntimeError("Hết thời gian chờ pipeline khởi động")
            error = application.state.run_error or active.error
            if error is not None:
                raise RuntimeError(f"Không khởi động được pipeline: {redact(error)}") from error
            if not active.running or not thread.is_alive():
                raise RuntimeError("Pipeline đã dừng trong lúc khởi động")
            logger.info("API và pipeline đã sẵn sàng")
            yield
        finally:
            active.stop()
            await run_in_threadpool(thread.join, shutdown_timeout)
            if thread.is_alive():
                logger.error("Pipeline chưa dừng sau %.1f giây", shutdown_timeout)
                raise RuntimeError("Pipeline chưa dừng hết khi đóng API")
            logger.info("Đã đóng API và pipeline")

    application = FastAPI(title="Multi-camera API", version="1.0.0", lifespan=lifespan)

    def get_manager():
        active = getattr(application.state, "manager", None)
        if active is None:
            raise HTTPException(503, "Pipeline chưa sẵn sàng")
        return active

    @application.get("/status", response_model=PipelineState, summary="Trạng thái pipeline và FPS")
    def get_status():
        active = get_manager()
        result = active.status()
        error = getattr(application.state, "run_error", None) or active.error
        if error is not None:
            result["error"] = redact(error)
        return result

    @application.get("/cameras", response_model=list[CameraState], summary="Danh sách camera")
    def get_cameras():
        return get_manager().list_cameras()

    @application.get("/cameras/{camera_id}", response_model=CameraState,
                     summary="Trạng thái và số liệu một camera")
    def get_camera(camera_id: str):
        try:
            return get_manager().get_camera_status(camera_id)
        except KeyError:
            raise HTTPException(404, "Không có camera này trong cấu hình") from None

    @application.patch("/cameras/{camera_id}", response_model=CameraState,
                       summary="Bật hoặc tắt camera đang chạy")
    def update_camera(camera_id: str, update: CameraUpdate):
        active = get_manager()
        try:
            # Kiểm tra ID trước để camera không tồn tại luôn trả về 404.
            active.get_camera_status(camera_id)
            if not active.running or active.error is not None or application.state.run_error is not None:
                raise HTTPException(503, "Pipeline không chạy; không thể điều khiển camera")
            return active.set_camera_enabled(camera_id, update.enabled, timeout=control_timeout)
        except KeyError:
            raise HTTPException(404, "Không có camera này trong cấu hình") from None
        except TimeoutError:
            raise HTTPException(504, "Hết thời gian chờ thay đổi trạng thái camera") from None
        except (RuntimeError, OSError) as exc:
            logger.warning("Không điều khiển được camera: %s", exc, extra={"camera_id": camera_id})
            raise HTTPException(503, "Không thực hiện được thay đổi trạng thái camera") from None
        except ValueError as exc:
            raise HTTPException(422, redact(exc)) from None

    @application.get("/events", response_model=EventPage,
                     summary="Sự kiện theo camera và khoảng thời gian")
    def get_events(camera_id: str | None = Query(default=None, min_length=1),
                   start_time: NaiveDatetime | None = None,
                   end_time: NaiveDatetime | None = None,
                   limit: int = Query(default=100, ge=1, le=500),
                   offset: int = Query(default=0, ge=0)):
        if start_time is not None and end_time is not None and start_time > end_time:
            raise HTTPException(422, "start_time phải nhỏ hơn hoặc bằng end_time")
        database_path = getattr(application.state, "database_path", None)
        if database_path is None:
            raise HTTPException(503, "Kho sự kiện chưa sẵn sàng")
        try:
            if not database_path.exists():
                return {"items": [], "total": 0, "limit": limit, "offset": offset}
            # Mở, truy vấn và đóng cùng thread; không dùng chung kết nối với writer.
            with closing(EventWriter(database_path)) as database:
                return database.list_events(camera_id=camera_id, start_time=start_time,
                                            end_time=end_time, limit=limit, offset=offset)
        except ValueError as exc:
            raise HTTPException(422, redact(exc)) from None
        except (sqlite3.DatabaseError, OSError, RuntimeError):
            logger.exception("Không truy vấn được kho sự kiện")
            raise HTTPException(503, "Kho sự kiện tạm thời không truy cập được") from None

    return application



