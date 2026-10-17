from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

from .schemas import CameraConfig, Event, ModelConfig
from .utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_MODES = ("shared", "per_camera")


@dataclass
class CameraSettings:
    capture: CameraConfig
    event: Event
    output: dict


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def rtsp_endpoint(value):
    """Identify a stream independently of credentials and local-host spelling."""
    parsed = urlsplit(str(value))
    if parsed.scheme.lower() != "rtsp":
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("RTSP URL must contain a host")
    try:
        address = ip_address(host)
    except ValueError:
        pass
    else:
        if address.is_loopback or (getattr(address, "ipv4_mapped", None) is not None and
                                   address.ipv4_mapped.is_loopback):
            host = "localhost"
    # Preserve queries: some camera servers select the channel with parameters.
    # Credentials authenticate access and do not identify the stream itself.
    return host, parsed.port or 554, unquote(parsed.path), parsed.query


def _event_geometry(block, field, points_key, camera_id, normalized):
    """Read and validate one optional event geometry independently."""
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{field} must be a mapping for {camera_id}")
    points = block.get(points_key)
    if points is None:
        return None

    name = "line" if points_key == "point" else "polygon"
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {name} for {camera_id}: expected numeric [x, y] points") from exc
    minimum = 2 if name == "line" else 3
    if (array.ndim != 2 or array.shape[1] != 2 or len(array) < minimum or
            (name == "line" and len(array) != 2) or not np.isfinite(array).all()):
        raise ValueError(f"Invalid {name} for {camera_id}")
    if normalized and ((array < 0).any() or (array > 1).any()):
        raise ValueError(f"Normalized {name} must be in [0, 1] for {camera_id}")
    if name == "line":
        if np.array_equal(array[0], array[1]):
            raise ValueError(f"Line must have distinct endpoints for {camera_id}")
    else:
        area = (np.dot(array[:, 0], np.roll(array[:, 1], 1)) -
                np.dot(array[:, 1], np.roll(array[:, 0], 1)))
        if not np.isfinite(area) or abs(area) < 1e-9:
            raise ValueError(f"Polygon must have nonzero finite area for {camera_id}")
    return array.tolist()


def load_config(camera_path, model_path):
    raw = load_yaml(camera_path)
    model_raw = load_yaml(model_path)
    cameras = []
    ids, urls = set(), set()
    for item in raw.get("cameras", []):
        cid = str(item["id"])
        if not cid or cid in ids:
            raise ValueError(f"Duplicate or empty camera ID: {cid}")
        ids.add(cid)
        source = item["source"]["url"]
        if source is None or str(source).strip() == "":
            raise ValueError(f"Missing source for {cid}")
        if not str(source).isdigit() and "://" not in str(source):
            source = str(local_path(source))
        coordinates = item.get("coordinates", "normalized")
        if coordinates not in ("normalized", "pixels"):
            raise ValueError(f"Invalid coordinates mode for {cid}")
        normalized = coordinates == "normalized"
        line = _event_geometry(item.get("lines", item.get("line")), "lines", "point", cid, normalized)
        polygon = _event_geometry(item.get("prohibit_area"), "prohibit_area", "polygon", cid, normalized)
        output = dict(item.get("output") or {})
        url = output.get("rtsp_url")
        options = {}
        if url:
            endpoint = rtsp_endpoint(url)
            if endpoint is None or endpoint in urls:
                raise ValueError(f"Output RTSP URL must be valid and unique for {cid}")
            urls.add(endpoint)
            options = {"output_url": url, "width": int(output.get("width", 800)),
                       "height": int(output.get("height", 450)), "fps": int(output.get("fps", 15))}
            if options["fps"] <= 0 or any(options[k] <= 0 or options[k] % 2 for k in ("width", "height")):
                raise ValueError(f"Output requires positive FPS and even dimensions for {cid}")
        cameras.append(CameraSettings(CameraConfig(cid, source), Event(line, polygon, coordinates == "normalized"), options))
    if not cameras:
        raise ValueError("At least one camera is required")
    sources = {endpoint for camera in cameras
               if (endpoint := rtsp_endpoint(camera.capture.source)) is not None}
    for camera in cameras:
        if camera.output and rtsp_endpoint(camera.output["output_url"]) in sources:
            raise ValueError(
                f"Output RTSP URL for {camera.capture.camera_id} conflicts with a camera source; "
                "use a separate output path (for example /output/cam_01)"
            )
    m = model_raw["model"]
    weight = local_path(m.get("weights", m.get("weight", "")))
    if not weight.is_file():
        raise FileNotFoundError(f"Model weights not found: {weight}")
    model = ModelConfig(str(weight), int(m.get("imgsz", 640)), str(m.get("device", "cpu")),
                        float(m.get("confidence", m.get("conf", 0.5))), m.get("classes"))
    if model.imgsz <= 0 or not 0 <= model.conf <= 1:
        raise ValueError("Invalid model image size or confidence")
    runtime = dict(model_raw.get("runtime", {}))
    runtime["model_mode"] = runtime.get("model_mode", "shared")
    if runtime["model_mode"] not in MODEL_MODES:
        raise ValueError(f"model_mode must be one of {MODEL_MODES}")
    for key, default in (("batch_size", 8), ("event_queue_size", 128), ("tracking_fps", 15)):
        runtime[key] = int(runtime.get(key, default))
        if runtime[key] <= 0:
            raise ValueError(f"{key} must be positive")
    runtime["reset_after_seconds"] = float(runtime.get("reset_after_seconds", 3))
    if not np.isfinite(runtime["reset_after_seconds"]) or runtime["reset_after_seconds"] <= 0:
        raise ValueError("reset_after_seconds must be finite and positive")
    runtime["database"] = local_path(runtime.get("database", "data/events.sqlite3"))
    return cameras, model, runtime
