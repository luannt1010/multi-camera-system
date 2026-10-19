import cv2
import numpy as np
import yaml

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        results = yaml.safe_load(f)
        return results

def resize_frame(frame, max_width=800, max_height=450):
    h, w = frame.shape[:2]
    scale = min(max_width / w,max_height / h, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def polygon_to_bbox(polygon_coords):
    pts = np.array(polygon_coords)
    xmin = np.min(pts[:, 0])
    ymin = np.min(pts[:, 1])
    xmax = np.max(pts[:, 0])
    ymax = np.max(pts[:, 1])
    return [xmin, ymin, xmax, ymax]

def make_status_frame(camera_id, status):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f"Camera {camera_id}: {status}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def draw_polygon(frame: np.ndarray, polygon, color=(0, 0, 255)) -> np.ndarray:
    if polygon is None:
        return frame
    polygon = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon], True, color, 2)
    return frame


def draw_results(packet, result, geometry, performance=None):
    """Draw on a copy so the asynchronous event snapshot stays immutable."""
    frame = packet.frame.copy()
    line, polygon = geometry
    draw_polygon(frame, polygon)
    if line is not None:
        cv2.line(frame, tuple(line[0].astype(int)), tuple(line[1].astype(int)), (255, 255, 0), 2)
    for box, tid, cls, score in zip(packet.boxes, packet.track_ids, packet.class_ids, packet.scores):
        x1, y1, x2, y2 = map(int, box)
        in_prohibit = result.events.get(int(tid), {}).get("in_prohibit", False) if result is not None else False
        color = (0, 0, 255) if in_prohibit else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid} class {cls} {score:.2f}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    label = packet.camera_id
    if result is not None and line is not None:
        label += f" | Left {result.left_count} Right {result.right_count}"
    # Keep the HUD legible when a high-resolution source is reduced for preview.
    hud_scale = max(1.0, frame.shape[1] / 800, frame.shape[0] / 450)
    font_scale = .6 * hud_scale
    thickness = max(1, round(2 * hud_scale))
    origin = (round(10 * hud_scale), min(round(25 * hud_scale), frame.shape[0] - 5))
    cv2.putText(frame, label,
                origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, label,
                origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    if performance is not None:
        fps_text = f"FPS: {performance['processing_fps']:.1f}"
        origin = (round(10 * hud_scale), min(round(50 * hud_scale), frame.shape[0] - 5))
        cv2.putText(frame, fps_text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, fps_text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), thickness)
    return frame

# if __name__ == "__main__":
#
#     cap = cv2.VideoCapture(r"D:\private\Videos\traffic_pl_1.mp4")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         cv2.imwrite("frame.png", frame)
#         cv2.imshow("", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()