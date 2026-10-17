# Multi-camera system

Pipeline: `SystemManager → CameraWorker (N) → FramePacket → InferenceWorker → DetectionPacket → Tracker → TrackingPacket → EventEngine → EventResults`.

Hai nhánh đầu ra: `EventWriter → SQLite + JPEG` và `Drawing Utils → FFmpeg → RTSP`.

Logging ghi ra console và file UTF-8 `logs/system.log`, tự xoay file và gắn module/thread/camera. Cấu hình tại `configs/logging.yaml`; xem [hướng dẫn logging](docs/logging.md) để bật DEBUG, theo dõi log và thêm logger trong code.

Preview và RTSP hiển thị FPS xử lý riêng cho từng camera. Log định kỳ có FPS capture/xử lý/output, số lần reconnect và độ trễ gần nhất/trung bình/P95. `e2e_ms` đo từ lúc Python nhận frame đến khi ghi/flush sang FFmpeg; xem định nghĩa và giới hạn phép đo trong hướng dẫn logging.

## Chạy

Project hiện dùng Python trong Conda `free_env`. Python 3.12 được dùng để kiểm thử. Cài dependency bằng `pip install -r requirements.txt` nếu dùng môi trường mới; FFmpeg phải có trong PATH. PyTorch/CUDA cần phù hợp với máy chạy.

Nguồn RTSP và đầu ra sau xử lý phải có đường dẫn riêng: ví dụ đọc `rtsp://localhost:8554/live1.sdp`, phát kết quả vào `rtsp://127.0.0.1:8554/output/cam_01`. `localhost` và `127.0.0.1` đều trỏ về máy này. Manager kiểm tra trùng stream trước khi khởi động, kể cả đầu ra camera này trùng nguồn camera khác.

```powershell
conda activate free_env
python -m src.manager --validate-config
python -m src.manager --display
```

Mặc định chạy không mở cửa sổ. `--display` hiển thị preview; Q, Esc hoặc đóng cửa sổ sẽ dừng toàn hệ thống. Ctrl+C cũng dừng và đợi ghi hết sự kiện đã nhận. Có thể chạy `python src/manager.py`.

```powershell
python -m src.manager --no-stream --duration 30
python -m src.manager --cameras configs/cameras.yaml --model configs/model.yaml
```

Để benchmark model dùng chung và model riêng từng camera, chọn `--model-mode shared` hoặc `--model-mode per_camera`; xem [hướng dẫn benchmark hai chế độ model](docs/benchmark-models.md).

Sửa nguồn thực tế trong `configs/cameras.yaml`: URL RTSP, đường dẫn video, hoặc số webcam. Các IP hiện có là cấu hình mẫu, cần thay theo hệ thống của bạn. Mỗi camera phải có ID và RTSP output riêng. Bỏ `output` hoặc đặt `rtsp_url: null` để tắt đầu ra camera đó. RTSP cần server nhận publish, ví dụ MediaMTX, được khởi động riêng.

`configs/model.yaml` trỏ tới weight đã có trong repository. Mặc định giữ GPU `cuda:0`; đặt `device: cpu` nếu không có CUDA. `classes: [0]` giữ bộ lọc class hiện tại; đặt `classes: null` để xử lý mọi class của model.

## Cấu hình và tọa độ

- `lines.point` (tùy chọn): 2 điểm khác nhau; `line.point` cũ vẫn được hỗ trợ.
- `prohibit_area.polygon` (tùy chọn): ít nhất 3 điểm, diện tích khác 0.
- `coordinates: normalized` (mặc định): các giá trị từ 0 đến 1, nhân với chiều rộng/cao frame nguồn. Dùng `coordinates: pixels` nếu cấu hình là pixel.
- `output.width`, `output.height`: số dương chẵn, mặc định 800 × 450; `output.fps`: mặc định 15. Output giữ tỷ lệ ảnh bằng padding đen.
- Đường dẫn tương đối của model, video và database tính từ thư mục gốc project, không phụ thuộc thư mục chạy lệnh.
- `runtime.batch_size`: giới hạn số frame thu thập mỗi lượt, tối đa một frame/camera; chọn lần lượt để camera cuối danh sách không bị bỏ đói. Model dùng chung xử lý cả batch; model riêng xử lý nhóm frame theo camera.
- `runtime.model_mode`: `shared` (mặc định) dùng một model cho batch nhiều camera; `per_camera` dùng model riêng từng camera, chia batch theo camera rồi inference lần lượt.
- `runtime.tracking_fps`: FPS xử lý dự kiến dùng cho ByteTrack; nên chỉnh gần tốc độ inference thực tế mỗi camera.
- `runtime.event_queue_size`: giới hạn số frame có sự kiện đang đợi ghi (mặc định 128). Frame độ phân giải cao sẽ cần nhiều RAM hơn; giảm thông số nếu RAM hạn chế.
- `runtime.reset_after_seconds`: bỏ frame quá cũ và reset danh tính tracker khi gián đoạn dài. Mỗi lần mở lại nguồn cũng reset danh tính; bộ đếm vẫn giữ trong phiên chạy.

Line và polygon cấu hình độc lập theo từng camera:

| Cấu hình | Xử lý sự kiện |
|---|---|
| Chỉ `lines.point` | Đếm vượt vạch, không kiểm tra vùng cấm |
| Chỉ `prohibit_area.polygon` | Kiểm tra vào vùng cấm, không đếm vượt vạch |
| Cả hai | Xử lý cả hai loại sự kiện |
| Không có cả hai | Chỉ detection/tracking/preview/RTSP, bỏ qua EventEngine và gửi sự kiện |

Ví dụ camera chỉ đếm vượt vạch:

```yaml
- id: cam_line
  source:
    url: "rtsp://localhost:8554/live1.sdp"
  lines:
    point: [[0.1, 0.5], [0.9, 0.5]]
```

Ví dụ camera chỉ kiểm tra vùng cấm:

```yaml
- id: cam_zone
  source:
    url: "rtsp://localhost:8554/live2.sdp"
  prohibit_area:
    polygon: [[0.1, 0.3], [0.9, 0.3], [0.9, 0.9], [0.1, 0.9]]
```

Ví dụ camera không có sự kiện:

```yaml
- id: cam_preview
  source:
    url: "0"
```

Các mục ví dụ được đặt trong danh sách `cameras:`. Có thể thêm `output.rtsp_url` riêng như cấu hình camera hiện tại. Bỏ mục line/polygon hoặc đặt giá trị `null` để tắt loại sự kiện đó. Nếu đã cung cấp danh sách điểm thì danh sách phải hợp lệ; `[]`, sai số chiều, điểm không hữu hạn, line trùng điểm và polygon không có diện tích đều bị báo lỗi trước khi khởi động. Khi mọi camera đều tắt sự kiện, manager không tạo writer hoặc mở database. Camera không có line cũng không hiển thị bộ đếm Left/Right.

## Hoạt động và giới hạn

Mỗi camera có capture thread và một slot frame mới nhất. Mặc định một model dùng chung xử lý batch; có thể chọn model riêng từng camera để benchmark. Tracker và trạng thái sự kiện tách theo camera. Thêm camera qua YAML, không cần sửa manager. Không có hàng đợi frame tăng vô hạn; khi quá tải, frame cũ được bỏ để giảm độ trễ. Đây là xử lý realtime nên không bảo đảm nhận mọi lần vượt vạch giữa hai frame bị bỏ.

Mỗi RTSP có worker và một slot riêng, giới hạn tốc độ gửi, thử kết nối lại sau lỗi. Watchdog kết thúc FFmpeg nếu ghi pipe bị treo quá 3 giây. FPS output là tốc độ gửi tối đa: khi inference chậm hơn, hệ thống không tự nội suy hay lặp frame để đạt FPS đó.

SQLite được tạo và sử dụng trên một writer thread, dùng WAL. JPEG snapshot lưu base64 ngay trong bảng `events`, giữ định dạng module cũ. Chỉ tạo bản ghi khi đi vào vùng cấm hoặc vượt đoạn vạch; không lưu lặp ở mọi frame còn trong vùng. Ảnh snapshot là ảnh nguồn chưa vẽ, timestamp là UTC lúc xử lý sự kiện. Các sự kiện cùng frame được ghi chung một transaction. Khi writer lỗi hoặc queue đầy, hệ thống báo lỗi và dừng thay vì âm thầm mất bản ghi; sự kiện chưa nhận do queue đầy không được lưu. Chưa có chính sách xóa dữ liệu cũ hay khôi phục bộ đếm sau restart.

Vượt vạch được xác định bằng tâm bbox, giao với đoạn vạch thực tế. Tên `left_to_right` tương ứng chuyển từ phía âm sang phía dương theo tích có hướng của hai điểm cấu hình; đảo thứ tự điểm sẽ đảo hướng. Vùng cấm bao gồm đường biên. Trạng thái ID không xuất hiện trong 60 giây được dọn.

`SystemManager.run()` / `start()` là hàm blocking, chỉ chạy một lần trên mỗi instance. `stop()` có thể gọi từ thread khác; `status()` trả về trạng thái camera, số frame xử lý, độ dài queue sự kiện và số frame RTSP bị thay thế. Khi dừng, manager ngừng capture, chờ worker, đóng FFmpeg và ghi hết queue sự kiện. Lỗi cleanup được báo rõ. Timeout capture RTSP là 3 giây; webcam phụ thuộc driver, manager báo lỗi nếu capture thread không dừng đúng hạn.

## Kiểm thử

Unit test riêng cho đếm vượt vạch, vùng cấm và state theo track ID: xem [hướng dẫn test sự kiện](docs/testing-events.md).

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

Nếu môi trường sandbox không ghi được thư mục tạm mặc định, chọn thư mục mới trong workspace:

```powershell
New-Item -ItemType Directory -Force .test-runs
python -m pytest --basetemp=.test-runs/manual-run
```

Các test bao phủ hình học và chiều vượt vạch, entry/re-entry, expiry và cô lập camera, snapshot SQLite, drain/error/overload writer, latest-frame queue, fairness batch, reset sau reconnect, cleanup khi lỗi, FFmpeg encode thật và watchdog với process treo. Test FFmpeg sẽ skip nếu binary không có trong PATH.

Đã chạy smoke test riêng với 2 video tổng hợp, model huấn luyện thật, ByteTrack thật và SQLite thật trên CPU: 30 frame trong cửa sổ chạy 4 giây, queue ghi rỗng và mọi worker đã dừng. Video tổng hợp không kiểm chứng độ chính xác nhận diện hay chất lượng sự kiện trên cảnh thực. Chưa kiểm chứng camera mạng và publish/playback RTSP xuyên suốt với server thật; cần kiểm thử tải trên phần cứng và nguồn triển khai trước khi chốt số camera/FPS.
