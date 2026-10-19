# API quản lý camera

Điểm khởi chạy hệ thống nằm trong **`src.manager`**. Một lệnh khởi động cả pipeline xử lý camera và API quản lý, dùng chung một `SystemManager` và số liệu FPS. Cả hai chế độ model `shared` và `per_camera` đều được hỗ trợ.

## Khởi chạy từ Manager

Chạy từ thư mục gốc dự án, trong môi trường đã cài dependency và chuẩn bị cấu hình/model theo [README](../README.md):

```powershell
python -m src.manager --cameras configs/cameras.yaml --model configs/model.yaml --model_mode shared
```

**`src.api` không còn là module chạy trực tiếp.** File này cung cấp schema, endpoint và hàm `create_app()` để entry point trong `src.manager` tạo ứng dụng FastAPI. Các cách gọi cũ `python -m src.api` và `uvicorn src.api:app` không dùng được vì module không có CLI hoặc biến toàn cục `app`.

API mặc định phục vụ tại `http://127.0.0.1:8000`. Mở [Swagger UI](http://127.0.0.1:8000/docs) để xem schema, thử request và đọc response trực tiếp.

### Preview và RTSP output

Để chạy pipeline, API và cửa sổ preview của từng camera:

```powershell
python -m src.manager --display
```

Preview hiển thị kết quả và FPS từ cùng pipeline mà API cung cấp trạng thái. Thêm `--no_stream` để tắt xuất RTSP, vẫn giữ xử lý camera, API và preview:

```powershell
python -m src.manager --display --no_stream
```

`--display` dùng được với cả `--model_mode shared` và `--model_mode per_camera`. Nhấn Q, Esc hoặc đóng một cửa sổ preview để dừng cả pipeline và API. Ctrl+C cũng dừng cả hai và đợi ghi hết hàng đợi sự kiện.

### Chọn địa chỉ, model và logging

Ví dụ chạy model riêng từng camera, tắt RTSP output và ghi log vào file riêng:

```powershell
python -m src.manager --host 127.0.0.1 --port 8000 --model_mode per_camera --no_stream --log_file logs/api.log
```

Các tùy chọn dưới đây đều thuộc lệnh `python -m src.manager`:

| Tùy chọn | Ý nghĩa / mặc định |
| --- | --- |
| `--host`, `--port` | Địa chỉ phục vụ API; mặc định `127.0.0.1`, `8000`. |
| `--cameras` | YAML camera; mặc định `configs/cameras.yaml`. |
| `--model` | YAML model/runtime; mặc định `configs/model.yaml`. |
| `--model_mode` | `shared` hoặc `per_camera`; CLI mặc định `shared`, ghi đè giá trị trong YAML. |
| `--display` | Mở preview; mặc định tắt. |
| `--no_stream` | Tắt RTSP output. |
| `--log_config` | YAML logging; mặc định `configs/logging.yaml`. |
| `--log_level`, `--log_file` | Ghi đè mức log và file log cho lần chạy. |

Xem đầy đủ bằng `python -m src.manager --help`. API, Uvicorn và pipeline dùng chung cấu hình logging; xem [hướng dẫn logging](logging.md).

## Khởi tạo và dừng hệ thống

`src.manager.main()` đọc tham số CLI, cấu hình logging, gọi `src.api.create_app()` với `manager_factory=SystemManager`, rồi chạy Uvicorn với **một worker**. Đây là luồng khởi chạy chung cho pipeline và API.

Khi Uvicorn bắt đầu lifespan của ứng dụng, `SystemManager` được tạo qua factory đã truyền vào và pipeline chạy trong một thread. Ứng dụng chờ nạp model và khởi tạo worker xong rồi mới nhận request. Việc quản lý vòng đời bên trong `create_app()` không tạo ra một lệnh chạy riêng cho module `src.api`.

Khi dừng, hệ thống yêu cầu pipeline dừng, chờ các worker đóng và ghi hết hàng đợi sự kiện. Lỗi khởi tạo khiến API không bắt đầu phục vụ. Nếu chạy không có `--display`, lỗi pipeline sau startup xuất hiện trong `/status` và API vẫn phục vụ để kiểm tra; với `--display`, pipeline dừng thì API cũng dừng.

Chỉ chạy một tiến trình cho cùng các camera. Mỗi lần gọi entry point tạo một pipeline riêng, có thể mở trùng camera, model và RTSP đầu ra nếu chạy đồng thời.

## Các endpoint

| Phương thức | Đường dẫn | Công dụng |
| --- | --- | --- |
| GET | `/status` | Trạng thái pipeline, chế độ model, số frame đã xử lý, hàng đợi và số liệu từng camera |
| GET | `/cameras` | Danh sách camera, trạng thái bật/tắt và metrics |
| GET | `/cameras/{camera_id}` | Trạng thái và metrics của một camera |
| PATCH | `/cameras/{camera_id}` | Đặt trạng thái bật hoặc tắt bằng JSON `{"enabled": true}` hoặc `false` |
| GET | `/events` | Lọc và phân trang lịch sử sự kiện |

Ví dụ xem camera bằng PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/status
Invoke-RestMethod http://127.0.0.1:8000/cameras
Invoke-RestMethod http://127.0.0.1:8000/cameras/cam_01
```

Một camera có cấu trúc response như sau; `metrics` còn chứa số lần reconnect và các thống kê độ trễ pipeline:

```json
{
  "camera_id": "cam_01",
  "enabled": true,
  "status": "connected",
  "metrics": {
    "capture_fps": 25.0,
    "processing_fps": 24.2,
    "output_fps": 24.0
  },
  "error": null
}
```

`enabled` là trạng thái cho phép camera chạy. `status` phản ánh trạng thái worker thực tế, nên camera đã bật vẫn có thể đang kết nối lại. `capture_fps`, `processing_fps`, `output_fps` lần lượt đo tốc độ đọc, xử lý và gửi frame đầu ra; tốc độ đầu ra không phải FPS phía client RTSP nhận được.

## Bật và tắt camera khi hệ thống đang chạy

```powershell
Invoke-RestMethod -Method Patch -Uri http://127.0.0.1:8000/cameras/cam_01 -ContentType application/json -Body '{"enabled":false}'
Invoke-RestMethod -Method Patch -Uri http://127.0.0.1:8000/cameras/cam_01 -ContentType application/json -Body '{"enabled":true}'
```

Body bắt buộc dùng boolean JSON `true`/`false`; chuỗi `"false"` hoặc số `0` bị từ chối. Gửi lại cùng trạng thái không đảo ngược trạng thái camera. Request được chuyển tới vòng lặp manager và chỉ trả thành công khi lệnh hoàn tất. Các camera khác tiếp tục hoạt động. Việc bật/tắt chỉ áp dụng cho phiên đang chạy, không ghi lại file YAML. Endpoint này điều khiển camera đã có trong cấu hình, không thêm camera mới.

Khi tắt, manager đóng capture và stream của camera, xóa trạng thái tracker và event để lần bật sau không nối nhầm track cũ. Các bộ đếm capture được giữ qua những lần bật/tắt trong cùng phiên. Model đã nạp được tái sử dụng khi bật lại; với `per_camera`, tắt camera chưa giải phóng vùng nhớ GPU của model riêng đó. Vùng nhớ được giữ trong suốt vòng đời manager.

Khi dùng `--display`, tắt camera qua API sẽ đóng cửa sổ của camera đó; bật lại sẽ mở lại cửa sổ. Các camera khác và API tiếp tục chạy.

## Truy vấn sự kiện

Các tham số đều không bắt buộc:

| Tham số | Ý nghĩa |
| --- | --- |
| `camera_id` | ID camera cần lọc, có thể là camera cũ đã bỏ khỏi YAML |
| `start_time` | Mốc bắt đầu theo giờ hệ thống, ví dụ `2026-10-17T08:00:00` |
| `end_time` | Mốc kết thúc theo giờ hệ thống, lớn hơn hoặc bằng mốc bắt đầu |
| `limit` | Số bản ghi trong trang, mặc định 100, từ 1 đến 500 |
| `offset` | Số bản ghi bỏ qua, mặc định 0, không âm |

Khoảng thời gian bao gồm cả hai đầu mút. Sự kiện lấy thời gian bằng `datetime.now()`; truy vấn dùng cùng giờ hệ thống, dạng ISO không kèm timezone. API không chuyển đổi sang UTC hoặc thay đổi timestamp đã lưu. Không thêm hậu tố `Z` hoặc offset như `+07:00` vào bộ lọc.

Ví dụ bằng `curl.exe`:

```powershell
curl.exe -G http://127.0.0.1:8000/events --data-urlencode "camera_id=cam_01" --data-urlencode "start_time=2026-10-17T08:00:00" --data-urlencode "end_time=2026-10-17T10:00:00" --data-urlencode "limit=20" --data-urlencode "offset=0"
```

Response gồm `items`, `total`, `limit`, `offset`. Mỗi item có `id`, `timestamp`, `camera_id`, `track_id`, `event_type`; không gửi ảnh base64 trong danh sách để tránh response lớn. Sự kiện được sắp xếp từ mới đến cũ, cùng thời điểm thì ID lớn hơn đứng trước. `total` là tổng số sự kiện khớp bộ lọc, trước khi phân trang. Nếu DB chưa được tạo vì chưa bật event, API trả danh sách rỗng.

Phần đọc và ghi database nằm chung trong lớp `EventWriter` ở `src/event_writer.py`. Pipeline vẫn gửi sự kiện bằng `WriterWorker.submit()`; worker gọi `EventWriter.save()` để ghi. Mỗi request `/events` tạo `EventWriter(database_path)`, gọi `list_events()` rồi đóng ngay trong cùng thread, kể cả khi truy vấn lỗi. Mỗi đối tượng giữ một kết nối SQLite riêng, không dùng chung connection với thread ghi. API không tự tạo database khi chưa có sự kiện. Truy vấn lọc bằng SQL thông thường, không tạo index.

## Mã lỗi và kiểm thử

`404`: không có camera trong cấu hình. `422`: body hoặc bộ lọc không hợp lệ, thời gian có timezone, hoặc khoảng thời gian bị đảo ngược. `503`: pipeline không chạy hoặc kho sự kiện tạm thời không truy cập được. `504`: hết thời gian chờ lệnh điều khiển camera. Khi pipeline lỗi sau startup, `/status` vẫn truy cập được để xem `running=false` và `error`; lịch sử sự kiện vẫn đọc được nếu SQLite còn hoạt động.

```powershell
python -m pytest tests/test_api.py tests/test_camera_control.py tests/test_event_writer.py -q
```

Các test API dùng manager giả và `TestClient` có lifespan, không cần GPU, camera hoặc server RTSP. Thiết kế lifespan và kiểm thử tham khảo [tài liệu FastAPI](https://fastapi.tiangolo.com/advanced/events/) và [Testing Events](https://fastapi.tiangolo.com/advanced/testing-events/).
