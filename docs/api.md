# API quản lý camera

API chạy cùng một `SystemManager`, dùng chung số liệu FPS và pipeline hiện có. Chỉ cần chạy API, không chạy thêm `python -m src.manager` cho cùng các camera. Cả hai chế độ model `shared` và `per_camera` vẫn hoạt động.

```powershell
python -m src.api
```

Mặc định phục vụ tại `http://127.0.0.1:8000`. Mở `http://127.0.0.1:8000/docs` để xem schema, thử request và đọc response trực tiếp trên Swagger UI.

Để chạy API và cửa sổ preview của từng camera bằng một lệnh:

```powershell
python -m src.api --display
```

Preview hiển thị kết quả và FPS từ cùng manager mà API đang quản lý. Nếu chỉ cần API và preview, thêm `--no-stream` để tắt xuất RTSP:

```powershell
python -m src.api --display --no-stream
```

`--display` dùng được với cả `--model-mode shared` và `--model-mode per_camera`. Nhấn Q, Esc hoặc đóng một cửa sổ preview để dừng cả pipeline và API. Ctrl+C cũng dừng cả hai và đợi ghi hết hàng đợi sự kiện.

```powershell
python -m src.api --host 127.0.0.1 --port 8000 --model-mode per_camera --no-stream --log-file logs/api.log
```

Có thể chọn cấu hình bằng `--cameras configs/cameras.yaml`, `--model configs/model.yaml` và cấu hình logging bằng `--log-config`, `--log-level`, `--log-file` như manager. Log của API, Uvicorn và pipeline dùng chung cấu hình logging, không dùng `print`.

Khi khởi động, API tạo một thread chạy manager, chờ nạp model và khởi tạo worker xong rồi mới nhận request. Khi dừng, API yêu cầu manager dừng, chờ các worker đóng và ghi hết hàng đợi sự kiện. Lỗi khởi tạo khiến API dừng khởi động. Nếu chạy không có `--display`, lỗi pipeline xảy ra sau đó xuất hiện trong `/status` và API vẫn phục vụ để kiểm tra; với `--display`, pipeline dừng thì API cũng dừng.

Chạy đúng **một worker**, không bật `--reload` khi dùng camera thật. Mỗi worker tạo một manager riêng, dẫn đến mở trùng camera, model và RTSP đầu ra. `python -m src.api` đã đặt `workers=1`. Nếu dùng Uvicorn trực tiếp: `python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1`; dùng lệnh `src.api` ở trên để áp dụng logging YAML của project.

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
