# Hướng dẫn logging

Ứng dụng dùng `logging` của Python. Các module trong `src/` và script Python của project dùng logger; không dùng `print()` để ghi thông tin vận hành.

## Chạy và xem log

Khi chạy manager, cấu hình được đọc từ `configs/logging.yaml`. Mặc định ghi đồng thời ra console (stderr) và file UTF-8 `logs/system.log`.

```powershell
conda activate free_env
python -m src.manager --display
```

Theo dõi file ở terminal khác, từ thư mục gốc project:

```powershell
Get-Content .\logs\system.log -Tail 100 -Wait
```

Bật chi tiết khi điều tra lỗi, hoặc chọn file khác:

```powershell
python -m src.manager --log-level DEBUG --log-file logs/debug.log
python -m src.manager --log-config configs/logging.yaml --validate-config
```

`--log-level` và `--log-file` ghi đè giá trị tương ứng trong YAML cho lần chạy đó. Đường dẫn file log tương đối luôn tính từ thư mục gốc project; đường dẫn truyền vào `--log-config` tính từ thư mục chạy lệnh.

## Cấu hình

```yaml
logging:
  level: INFO
  console: true
  file: logs/system.log
  max_bytes: 10485760
  backup_count: 5
  queue_size: 4096
  status_interval_seconds: 30
```

| Trường | Ý nghĩa |
|---|---|
| `level` | Ngưỡng tối thiểu: DEBUG, INFO, WARNING, ERROR hoặc CRITICAL |
| `console` | Ghi ra terminal; đặt false để chỉ ghi file |
| `file` | File log; đặt null để chỉ dùng console |
| `max_bytes` | Xoay file khi đạt xấp xỉ 10 MiB với giá trị mặc định |
| `backup_count` | Giữ tối đa 5 bản cũ: system.log.1 đến system.log.5 |
| `queue_size` | Số bản ghi log đang chờ xử lý, mặc định 4096 |
| `status_interval_seconds` | Chu kỳ ghi trạng thái toàn pipeline, mặc định 30 giây |

Ít nhất một đầu ra console hoặc file phải được bật. Các kích thước queue/file và chu kỳ trạng thái phải dương; backup_count ít nhất 1. File log và thư mục được tạo tự động. Mỗi lần chạy tiếp tục ghi nối tiếp file đang có; rotation quản lý file cũ.

Dùng một tiến trình manager để ghi vào một file. Nếu khởi chạy nhiều manager cùng lúc, chọn `--log-file` khác nhau cho từng tiến trình vì RotatingFileHandler không đồng bộ rotation giữa nhiều tiến trình.

## Đọc nội dung

Ví dụ định dạng:

```text
2026-10-17T19:30:00.123+07:00 INFO     src.camera_worker [capture-cam_01] [camera=cam_01] Camera connected (generation=1)
```

Các trường lần lượt là thời gian kèm múi giờ máy chạy, mức log, module, thread, camera và thông điệp. Log chung có `camera=-`.

| Mức | Dùng để theo dõi |
|---|---|
| DEBUG | Thời gian inference theo batch, reset tracker, sự kiện vừa phát hiện, chi tiết lỗi gửi FFmpeg |
| INFO | Khởi động/dừng, tải model, camera kết nối, worker sẵn sàng, sự kiện đã commit vào SQLite, trạng thái định kỳ |
| WARNING | Camera mất kết nối và thử lại, pipe FFmpeg bị treo, lỗi gửi có thể thử lại, log quá tải |
| ERROR | Worker/pipeline thất bại, queue sự kiện đầy, lỗi FFmpeg, lỗi shutdown |
| CRITICAL | Dành cho sự cố nghiêm trọng do code mở rộng phát ra; luồng hiện tại chưa có điểm chủ động phát mức này |

Log `Committed ... event records` chỉ xuất hiện sau khi transaction SQLite hoàn thành. `zone_entry` và crossing ở DEBUG là kết quả phát hiện trước khi ghi DB. `Pipeline status` cho biết camera còn kết nối, tổng frame đã xử lý, độ dài queue sự kiện và frame đầu ra bị thay thế.

## FPS, độ trễ và reconnect theo camera

Mỗi chu kỳ `status_interval_seconds`, manager ghi thêm một dòng INFO `Camera performance` cho từng camera; lúc dừng cũng ghi một lần cuối. Các số liệu có trong `SystemManager.status()["camera_metrics"][camera_id]`. Hình preview và frame RTSP hiển thị `FPS: ...` riêng cho từng camera, là FPS xử lý hoàn tất detection/tracking/event của camera đó.

Cỡ chữ camera/FPS tự tăng theo độ phân giải nguồn để vẫn đọc được sau khi thu nhỏ preview. Khi shutdown, frame đầu ra chưa gửi được bỏ khỏi queue và tính vào `dropped_frames`, nên `pending_frames` về 0 sau khi đóng worker.

| Trường | Ý nghĩa |
|---|---|
| `capture_fps` | FPS frame thực nhận được từ camera, không phải FPS khai báo của nguồn |
| `processing_fps` | FPS frame đã xử lý xong detection/tracking/event theo camera |
| `output_fps` | FPS frame gửi thành công sang pipe FFmpeg |
| `captured_frames` / `processed_frames` / `sent_frames` | Tổng số frame ở từng chặng trong phiên chạy |
| `frame_age_ms` | Từ lúc nhận frame đến khi batch inference bắt đầu |
| `inference_ms` | Thời gian inference của batch chứa camera đó; cùng batch có chung số này, không chia cho số camera |
| `processing_latency_ms` | Từ lúc nhận frame đến khi tracking/event và enqueue sự kiện hoàn tất, trước khi vẽ và gửi stream |
| `output_queue_ms` | Thời gian frame đã vẽ chờ trong queue đầu ra |
| `send_ms` | Thời gian resize/padding, tạo process nếu cần và ghi/flush frame sang FFmpeg |
| `e2e_ms` | Độ trễ nội bộ từ lúc nhận frame đến khi gửi thành công frame đó sang pipe FFmpeg; gồm cả xử lý, vẽ, chờ queue và gửi |
| `connection_attempts` | Tổng số lần thử mở nguồn, gồm kết nối ban đầu |
| `reconnect_attempts` | Số lần thử mở lại nguồn sau khi camera đã từng nhận ít nhất một frame |
| `reconnect_count` | Số phiên phục hồi thành công; chỉ tăng khi nhận frame đầu tiên sau reconnect |
| `send_failures` | Số lần gửi FFmpeg thất bại |
| `dropped_frames` / `pending_frames` | Frame output bị thay thế bởi frame mới / đang chờ gửi |

Các độ trễ là object `{last_ms, avg_ms, p95_ms}`: số gần nhất, trung bình và phân vị 95% trong cửa sổ 5 giây gần nhất. FPS cũng dùng cửa sổ 5 giây, tính theo các lần thực nhận/xử lý/gửi frame trên cùng một đồng hồ monotonic. Mẫu đầu tiên dùng làm mốc nên FPS ban đầu là 0. Bộ nhớ tối đa 4096 mẫu mỗi chỉ số; không giữ lịch sử tăng vô hạn. Khi không có mẫu mới quá 5 giây, FPS về 0 và độ trễ thành null. Capture FPS được reset ngay khi nguồn mất kết nối hoặc dừng; reconnect cũng reset mốc FPS xử lý.

Chỉ ghi độ trễ đầu ra khi `send()` báo thành công. Nếu tắt stream, camera không có output, hoặc gửi không kèm timestamp nguồn thì không có phép đo E2E tương ứng; không thay bằng độ trễ xử lý. `StreamWorker.submit(frame, captured_at=packet.timestamp)` truyền timestamp cùng đúng frame đó qua queue, kể cả khi queue thay thế frame cũ.

Mốc bắt đầu là timestamp Python tạo sau khi OpenCV đọc thành công frame. Vì vậy `e2e_ms` không đo độ trễ cảm biến, bộ đệm camera/decoder, mã hóa bên trong FFmpeg, đường truyền RTSP hoặc bộ đệm/hiển thị của trình xem. Đây là độ trễ đầu cuối trong phạm vi pipeline Python đến pipe FFmpeg, không phải phép đo camera tới màn hình người xem.

EOF của video và phát lại cũng tính là một phiên reconnect khi nhận được frame đầu tiên sau khi mở lại. Các lần thử mở thất bại trước frame đầu tiên chỉ tăng `connection_attempts`, không tăng reconnect. Bộ đếm được giữ trong phiên chạy và bắt đầu lại khi tạo manager/worker mới.

Camera mất kết nối sẽ được cảnh báo tối đa một lần mỗi 30 giây. Log kết nối thành công vẫn xuất hiện khi camera phục hồi. INFO không ghi cho từng frame; DEBUG có thể tạo nhiều log hơn khi hệ thống tải cao.

## FFmpeg và thông tin nhạy cảm

stderr của FFmpeg do `StreamOutput` khởi chạy được đọc bằng thread riêng và chuyển vào logger. File log nhận được cảnh báo/lỗi FFmpeg cùng ngữ cảnh camera; các dòng trùng liên tiếp trong một process được gom trong khoảng 30 giây. FFmpeg khởi chạy thủ công ở terminal khác có đầu ra riêng, không thuộc file log của manager. Chẩn đoán native từ OpenCV và giao diện tiến trình của thư viện ngoài có thể vẫn ghi trực tiếp ra terminal.

Formatter che phần username/password và query của URL RTSP/RTSPS/HTTP/HTTPS trong cả thông điệp lẫn traceback, ví dụ `rtsp://[REDACTED]@host/live?[REDACTED]`. Không ghi snapshot, bytes ảnh hay token riêng lẻ vào log. Cơ chế che này áp dụng cho URL; code mới vẫn phải tránh ghi mật khẩu/token dưới dạng chuỗi rời.

## Khi queue logging đầy và khi dừng

Logging dùng queue giới hạn và một listener thread để ghi file/console ngoài vòng inference. Khi queue đầy, DEBUG/INFO có thể bị bỏ; WARNING trở lên được ghi trực tiếp để giữ thông tin sự cố. Khi shutdown, hệ thống ghi số log DEBUG/INFO đã bỏ (nếu có), xử lý hết log đang chờ rồi đóng file. Trong tình huống quá tải này, ghi cảnh báo có thể chờ I/O và thứ tự giữa log chờ/ghi trực tiếp có thể khác thời điểm phát sinh.

## Gọi từ code và thêm log mới

Entry point CLI tự cấu hình logging. Nếu nhúng manager trong chương trình khác, chủ động mở/đóng logging ở điểm khởi chạy:

```python
from src.logging_config import configure_logging
from src.manager import SystemManager

with configure_logging() as logs:
    manager = SystemManager(status_interval_seconds=logs.status_interval_seconds)
    manager.run()
```

Trong module dùng logger riêng và truyền tham số qua placeholders:

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processed %d items", count, extra={"camera_id": camera_id})

try:
    perform_operation()
except Exception:
    logger.exception("Operation failed", extra={"camera_id": camera_id})
    raise
```

Với module ngoài package `src`, dùng tên thuộc cây logger ứng dụng, ví dụ `logging.getLogger("src.integration")`. Không gọi `basicConfig()` trong worker hay module thư viện. Không thay đổi handler trong vòng lặp frame. Chỉ dùng `logger.exception()` trong khối xử lý ngoại lệ; lỗi tạm thời có retry thường dùng WARNING và chi tiết traceback ở DEBUG.

## Kiểm thử

`tests/test_logging.py` kiểm tra console/file, UTF-8, mức log, rotation, camera mặc định, che URL và traceback, shutdown drain, xử lý queue đầy, cấu hình lại không trùng handler và cấm gọi print trong code ứng dụng.

```powershell
python -m pytest tests/test_logging.py -q
```
