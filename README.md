# Multi-camera system

Hệ thống phân tích video từ nhiều camera bằng **YOLO** và **ByteTrack**: phát hiện đối tượng, theo dõi ID trong từng camera, đếm vượt vạch và phát hiện đi vào vùng cấm. Sự kiện được lưu vào **SQLite** kèm ảnh JPEG; video đã vẽ kết quả có thể xem bằng preview hoặc phát qua **RTSP**.

Pipeline được **SystemManager** điều phối; **FastAPI** cung cấp API xem trạng thái, bật/tắt camera và truy vấn lịch sử sự kiện. Mỗi camera thu nhận frame ở luồng riêng; hệ thống ưu tiên frame mới nhất khi xử lý không theo kịp nguồn.

## Cấu hình YAML

| File | Mục đích |
| --- | --- |
| [configs/cameras.yaml](configs/cameras.yaml) | Nguồn camera, vạch đếm, vùng cấm và RTSP output khi chạy local. |
| [configs/cameras.docker.yaml](configs/cameras.docker.yaml) | Cấu hình camera cho Docker; các camera đang bật dùng `host.docker.internal` để truy cập RTSP server trên host. |
| [configs/model.yaml](configs/model.yaml) | Model YOLO, thiết bị suy luận và các tham số runtime. |
| [configs/logging.yaml](configs/logging.yaml) | Mức log, file log, rotation và chu kỳ ghi trạng thái. |
| [configs/yolo_config.yaml](configs/yolo_config.yaml) | Cấu hình huấn luyện/validation của `yolo/train.py`; không được nạp khi chạy pipeline. |

### Camera

Thêm hoặc sửa phần tử trong danh sách `cameras:`. Ví dụ một camera có cả vạch và vùng cấm:

```yaml
cameras:
  - id: cam_01
    coordinates: normalized
    source:
      url: "rtsp://localhost:8554/live1.sdp"
    lines:
      point: [[0.1, 0.5], [0.9, 0.5]]
    prohibit_area:
      polygon: [[0.1, 0.3], [0.9, 0.3], [0.9, 0.9], [0.1, 0.9]]
    output:
      rtsp_url: "rtsp://localhost:8554/output/cam_01"
      width: 800
      height: 450
      fps: 15
```

| Trường | Cách thiết lập |
| --- | --- |
| `id` | ID duy nhất cho mỗi camera. |
| `source.url` | URL RTSP, đường dẫn video như `data/video.mp4`, hoặc số webcam dạng chuỗi như `"0"`. Loại nguồn được tự nhận biết từ giá trị này. |
| `coordinates` | `normalized` (mặc định): tọa độ trong khoảng 0–1; `pixels`: tọa độ theo frame gốc. |
| `lines.point` | Hai điểm khác nhau tạo thành đoạn vạch đếm. Bỏ mục `lines` để tắt đếm vạch. |
| `prohibit_area.polygon` | Ít nhất ba điểm tạo thành vùng có diện tích khác 0. Bỏ mục này để tắt phát hiện vùng cấm. |
| `output.rtsp_url` | Đường dẫn publish kết quả; phải riêng biệt với mọi nguồn và output khác. Bỏ `output` hoặc đặt URL là `null` để tắt RTSP output của camera. |
| `output.width/height/fps` | Kích thước phải dương và chẵn; FPS phải dương. Mặc định 800 × 450, giới hạn gửi 15 FPS. |

Vạch và vùng cấm có thể bật độc lập; camera không có cả hai vẫn nhận diện, theo dõi và xuất video. Các URL hiện có trong YAML là cấu hình mẫu, cần đổi theo nguồn thực tế.

### Model và runtime

Nội dung hiện tại của `configs/model.yaml`:

```yaml
model:
  weights: "yolo/results/detect/runs/train/person_vehicle/weights/best.pt"
  device: "cuda:0"         # Đổi thành cpu nếu không dùng GPU CUDA.
  imgsz: 640              # Kích thước ảnh đầu vào khi suy luận.
  confidence: 0.5         # Ngưỡng confidence, từ 0 đến 1.
  classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # null: lấy mọi class của model.

runtime:
  model_mode: shared
  batch_size: 8
  tracking_fps: 15
  event_queue_size: 128
  database: "data/events.sqlite3"  # File SQLite lưu sự kiện.
  reset_after_seconds: 3
```

- `weights`: phải trỏ tới file model đã chuẩn bị. Đường dẫn tương đối của weights, video và database được tính từ thư mục gốc dự án.
- `shared`: một model xử lý batch nhiều camera. `per_camera`: mỗi camera một model, được gọi lần lượt trên luồng Manager.
- `batch_size`: số frame tối đa mỗi lượt, tối đa một frame/camera; không chờ đủ batch.
- `tracking_fps`: FPS dự kiến cho ByteTrack, không ép inference chạy ở FPS này.
- `event_queue_size`: số frame có sự kiện tối đa chờ ghi database.
- `reset_after_seconds`: ngưỡng bỏ frame cũ và reset trạng thái theo dõi khi gián đoạn dài.

**Khi chạy bằng CLI hiện tại, `--model_mode` mặc định là `shared` và ghi đè giá trị trong YAML.** Muốn dùng model riêng, truyền rõ `--model_mode per_camera`.

### Logging

Giữ cấu hình mặc định trong `configs/logging.yaml` hoặc sửa `logging.level`, `logging.file` và `logging.status_interval_seconds` theo nhu cầu. Mặc định log ở mức `INFO`, ghi ra console và `logs/system.log`, xoay file khoảng 10 MiB, giữ 5 bản cũ và ghi trạng thái mỗi 30 giây. Xem [hướng dẫn logging](docs/logging.md) để đọc FPS, độ trễ và số lần reconnect.

## Chạy local

Các lệnh dưới đây chạy bằng **PowerShell**, từ thư mục gốc dự án.

### 1. Chuẩn bị môi trường

Dùng Python 3.12 và FFmpeg có trong `PATH`. Nếu dùng `cuda:0`, môi trường PyTorch phải hỗ trợ CUDA và phù hợp với GPU/driver của máy.

Tạo môi trường Conda nếu chưa có; nếu đã có `free_env`, chỉ cần kích hoạt:

```powershell
conda create -n free_env python=3.12
conda activate free_env
python -m pip install -r requirements.txt
ffmpeg -version
```

Chuẩn bị file weights theo `model.weights`, sửa nguồn trong `configs/cameras.yaml`. Với nguồn/đầu ra RTSP, khởi động RTSP server riêng (ví dụ MediaMTX) và bảo đảm các stream nguồn đã hoạt động. Server RTSP không được khởi chạy bởi ứng dụng.

### 2. Khởi chạy

Chạy pipeline cùng API:

```powershell
python -m src.manager --cameras configs/cameras.yaml --model configs/model.yaml --model_mode shared
```

Các cách chạy khác, chọn theo nhu cầu:

```powershell
# API, pipeline và preview.
python -m src.manager --display

# API và preview, tắt RTSP output.
python -m src.manager --display --no_stream

# Dùng model riêng từng camera.
python -m src.manager --model_mode per_camera
```

Mở [Swagger UI](http://127.0.0.1:8000/docs) để thử API; [trạng thái hệ thống](http://127.0.0.1:8000/status) để xem camera và metrics. Mặc định không mở preview. Nhấn `Ctrl+C` để dừng; khi có preview, `Q`, `Esc` hoặc đóng cửa sổ cũng dừng hệ thống và chờ ghi hết sự kiện đã nhận.

Entry point hiện tại là **`python -m src.manager`**. Các cờ dùng dấu gạch dưới như `--model_mode`, `--no_stream`, `--log_file`; xem đầy đủ bằng `python -m src.manager --help`.

## Chạy Docker

Chạy lệnh trên CMD

### 1. Build image

```bash
cd multi-camera-system
set "PROJECT_DIR=multi-camera-system"
set "WEIGHTS_DIR=%PROJECT_DIR%\yolo\results\detect\runs\train\person_vehicle\weights"
cd /d "%PROJECT_DIR%"
if not exist "configs\cameras.yaml" copy "configs\cameras.docker.yaml" "configs\cameras.docker.yaml" #Copy file yaml vào env docker
notepad configs\cameras.docker.yaml # Kiểm tra lại cổng -> chuyển localhost,... -> host.docker.internal:8554, VD: rtsp://host.docker.internal:8554/output/cam_01
docker build -t multi-camera:latest .
```

### 2. Khởi chạy container

```bash
docker run -d --name multi-camera --gpus all -p 127.0.0.1:8000:8000 --stop-timeout 90 --mount "type=bind,source=%PROJECT_DIR%\configs,target=/app/configs,readonly" --mount "type=bind,source=%WEIGHTS_DIR%,target=/app/yolo/results/detect/runs/train/person_vehicle/weights,readonly" --mount "type=volume,source=multi-camera-data,target=/app/data" --mount "type=volume,source=multi-camera-logs,target=/app/logs" multi-camera:latest python -m src.manager --host 0.0.0.0 --port 8000 --cameras /app/configs/cameras.docker.yaml
```

Lệnh này khởi động pipeline và API, không mở preview. Truy cập [Swagger UI](http://127.0.0.1:8000/docs) từ host. Khi mở video bằng trình xem trên host, dùng địa chỉ server từ host, ví dụ `rtsp://localhost:8554/output/cam_01`; tên `host.docker.internal` trong YAML dùng cho kết nối từ container.

- **Model:** Dockerfile không copy weights và `.dockerignore` loại `yolo/`, `*.pt`; mount `yolo/` để đường dẫn weights mặc định tồn tại trong container.
- **Dữ liệu:** mount `data/` và `logs/` để giữ database/log sau khi container bị xóa. Trên Linux, hai thư mục cần cho phép UID/GID `10001` của container ghi.
- **Mạng RTSP:** `localhost` trong container trỏ về chính container. Dùng `host.docker.internal` nếu RTSP server ở host, hoặc địa chỉ camera/server thực tế. Docker Desktop cung cấp tên này; `--add-host` bổ sung ánh xạ cho Docker Engine. Xem [tài liệu Docker về host gateway](https://docs.docker.com/reference/cli/docker/container/run/#add-entries-to-container-hosts-file---add-host).
- **CPU:** đặt `model.device: cpu` và bỏ `--gpus all` khỏi lệnh chạy.
- **Lệnh khởi động:** ví dụ ghi đè `CMD` hiện tại của Dockerfile (vẫn gọi `src.api`) bằng entry point `src.manager`.

Dừng từ terminal khác bằng `docker stop --timeout 60 multi-camera-system`.

## Tài liệu chi tiết

| Tài liệu | Nội dung |
| --- | --- |
| [API](docs/api.md) | Endpoint, bật/tắt camera, lọc lịch sử và phân trang sự kiện. |
| [Logging](docs/logging.md) | Cấu hình log, FPS/độ trễ, reconnect và cách theo dõi lỗi. |
| [Benchmark model](docs/benchmark-models.md) | Phương pháp so sánh `shared` và `per_camera`, điều kiện đo và cách đọc kết quả. |
| [Kiểm thử sự kiện](docs/testing-events.md) | Quy ước đếm vạch/vùng cấm và cách chạy kiểm thử. |
