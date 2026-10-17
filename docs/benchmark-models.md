# Benchmark model dùng chung và model riêng từng camera

Manager hỗ trợ hai chế độ để so sánh trên cùng pipeline capture, tracking, sự kiện và output:

| Chế độ | Số model | Cách inference |
|---|---|---|
| `shared` | Một model cho toàn bộ camera | Gửi cả batch frame vào một lần `predict()` |
| `per_camera` | Một model riêng cho mỗi camera | Nhóm frame trong batch theo camera rồi gọi model tương ứng lần lượt |

Hai chế độ dùng chung logic thu thập: mỗi vòng lấy tối đa một frame mới nhất từ mỗi camera, tổng số frame không vượt `runtime.batch_size`. Không chờ đủ batch; camera được luân phiên khi số camera lớn hơn giới hạn batch. Frame quá cũ và frame sau reconnect vẫn áp dụng cùng chính sách bỏ/reset.

Ở `per_camera`, batch thu thập vẫn chứa frame của nhiều camera, nhưng mỗi model chỉ nhận nhóm frame của camera đó. Với collector hiện tại, mỗi nhóm có một frame nên batch thực tế của mỗi model là 1. Đây không phải một tensor batch đa camera chạy qua nhiều model trong một lệnh. Các lần inference chạy tuần tự trên thread manager; chế độ này không tạo thêm thread hoặc process inference.

## Chọn chế độ

Cấu hình trong `configs/model.yaml`:

```yaml
runtime:
  model_mode: shared
  batch_size: 8
```

Nếu bỏ `model_mode`, mặc định là `shared`. Hai chế độ cùng dùng cấu hình weights, device, imgsz và bộ lọc class trong mục `model`.

Có thể ghi đè chế độ bằng CLI mà không sửa YAML:

```powershell
python -m src.manager --model-mode shared
python -m src.manager --model-mode per_camera
```

Khi gọi từ Python, `SystemManager(model_mode="per_camera")` ghi đè giá trị trong YAML. Mặc định constructor là `model_mode=None`, nghĩa là lấy từ cấu hình.

## Chạy hai lượt benchmark

Chạy từ thư mục gốc project và dùng cùng môi trường Python:

```powershell
conda activate free_env
python -m src.manager --model-mode shared --duration 300 --log-file logs/benchmark-shared.log
python -m src.manager --model-mode per_camera --duration 300 --log-file logs/benchmark-per-camera.log
```

Các lệnh trên giữ output theo cấu hình camera để đo cả pipeline. Nếu chỉ muốn đo pipeline không xuất RTSP, thêm `--no-stream` vào **cả hai** lệnh. Giữ cùng lựa chọn preview ở hai lượt; mở preview cũng tốn tài nguyên.

Để so sánh công bằng:

- Giữ nguyên model, `imgsz`, device, số camera, `batch_size`, cấu hình tracker/sự kiện và output.
- Dùng cùng nội dung video, độ phân giải, FPS nguồn và tốc độ phát. Với nguồn phát ngoài, khởi động lại cùng đoạn video trước mỗi lượt; cảnh RTSP trực tiếp thay đổi có thể làm kết quả khác nhau.
- Giữ cùng chính sách frame mới nhất và bỏ frame; không đổi queue hoặc tốc độ output giữa hai lượt.
- Chạy từng chế độ riêng, tránh để hai manager hoặc ứng dụng GPU nặng chạy cùng lúc.
- Lặp lại ít nhất 3 lần; đổi thứ tự hai chế độ giữa các lần để giảm ảnh hưởng nhiệt độ và tải nền.

`--duration 300` giới hạn vòng xử lý khoảng 300 giây sau khi khởi tạo. Thời gian load model nằm ngoài cửa sổ này; các lần predict đầu có thể vẫn chứa warmup. Chưa có chức năng tự loại warmup khỏi số liệu. Khi tổng hợp log, chọn các mốc ổn định sau khoảng warmup đã quy định giống nhau, ví dụ 30 giây đầu. Các latency/FPS là thống kê cửa sổ gần đây, còn `processed_frames` là bộ đếm tích lũy; không dùng riêng số cuối cùng để suy ra throughput sau khi đã loại warmup. Có thể tính throughput trong khoảng ổn định từ chênh lệch bộ đếm giữa hai mốc chia cho thời gian giữa hai mốc đó.

## Đọc kết quả

Log `Camera performance` chứa thống kê từng camera; `Pipeline status` có tổng số frame đã xử lý. Giữ cùng `status_interval_seconds` ở hai lượt để có cùng nhịp lấy mẫu.

| Chỉ số | Ý nghĩa khi so sánh |
|---|---|
| `processing_fps` | Tốc độ frame hoàn tất tracking/sự kiện của từng camera; kiểm tra cả camera chậm nhất |
| `processed_frames` | Số frame đã xử lý; lấy chênh lệch theo thời gian để tính throughput của khoảng đo |
| `inference_ms` | `shared`: thời gian gọi model cho toàn batch, được ghi cho từng camera trong batch; không chia cho số frame. `per_camera`: thời gian gọi model của riêng camera đó |
| `frame_age_ms` | Từ lúc Python nhận frame đến khi bắt đầu gọi model tương ứng; ở `per_camera` bao gồm thời gian chờ model của các camera được xử lý trước |
| `processing_latency_ms` | Từ lúc Python nhận frame đến khi hoàn tất tracking/sự kiện; xem cả trung bình và P95 |
| `capture_fps`, reconnect | Kiểm tra nguồn đầu vào có ổn định và tương đương giữa hai lượt không |
| `output_fps`, số frame output bị thay thế | Kiểm tra nhánh xuất stream có theo kịp xử lý không |
| `e2e_latency_ms` | Từ lúc Python nhận frame đến khi ghi/flush frame sang FFmpeg; chưa bao gồm encode, mạng, buffer RTSP và màn hình người xem |

`inference_ms` khác cách nhóm công việc giữa hai chế độ, nên không dùng riêng chỉ số này để kết luận chế độ nào nhanh hơn. So sánh thêm tổng throughput, FPS từng camera, độ trễ xử lý và độ chính xác đếm vạch/vùng cấm khi hệ thống phải bỏ frame.

Manager hoàn tất inference của toàn bộ batch rồi mới chạy tracker/sự kiện. Vì vậy ở `per_camera`, `processing_latency_ms` của camera được infer trước cũng bao gồm thời gian chờ các model phía sau hoàn tất; `inference_ms` vẫn chỉ ghi thời gian gọi model tương ứng.

Ghi thêm CPU, GPU, RAM và VRAM bằng công cụ theo dõi của máy vì manager chưa tự thu thập các số liệu này. Model riêng từng camera tạo nhiều instance; mức tăng bộ nhớ và tốc độ thực tế cần đo trên phần cứng triển khai. Thử tăng số camera lần lượt 1, 2, 4, 8 và đánh giá chế độ nào vẫn đáp ứng ngưỡng FPS/độ trễ của ứng dụng.
