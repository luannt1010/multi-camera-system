# Báo cáo hệ thống phân tích video đa camera

`main.tex` chứa toàn bộ nội dung, sơ đồ TikZ và bốn biểu đồ PGFPlots, không cần ảnh hoặc dữ liệu ngoài để biên dịch.

## Biên dịch

Chọn **XeLaTeX** trong Overleaf, hoặc chạy hai lần từ thư mục này:

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Font chính ưu tiên TeX Gyre Termes, dự phòng Times New Roman. Cần các gói phổ biến: fontspec, geometry, amsmath, booktabs, tabularx, enumitem, xcolor, tikz, pgfplots, caption, fancyhdr, xurl và hyperref. Không cần shell-escape, BibTeX hoặc Python để biên dịch.

## Bố cục theo yêu cầu

1. Tổng quan hệ thống và workflow.
2. Phương thức vận hành: camera worker; inference → tracker; sự kiện; event writer và API; stream output.
3. Benchmark shared với per_camera tại 1, 4 và 8 camera, phương pháp tổng hợp, bảng và biểu đồ.
4. Nhận xét, đánh giá và lựa chọn.

## Nguồn và cách tổng hợp

Nội dung vận hành đối chiếu mã nguồn hiện tại trong `src/`. Benchmark lấy từ sáu log ngày 17/10/2026, không chạy benchmark mới. File `benchmark-evidence.csv` ghi nguồn, dòng status, khoảng thời gian, chênh lệch frame và số liệu trước khi làm tròn.

- Dùng snapshot định kỳ thứ 3 → thứ 9 của phiên thành công: khoảng 180 giây, bỏ phần đầu khởi động. `8cam/per.log` có một lượt khởi động lỗi trước phiên thành công, đã loại.
- FPS tổng = chênh lệch `processed_frames` / thời gian giữa hai mốc. FPS/camera = FPS tổng / số camera.
- FPS xuất trung bình/camera = tổng chênh lệch `sent_frames` / thời gian / số camera.
- Latency = trung bình `avg_ms` ở snapshot thứ 4 → thứ 9, tất cả camera và snapshot đồng trọng số. Đây là trung bình cửa sổ 5 giây tại các mốc lấy mẫu, không phải trung bình theo từng frame của cả khoảng đo.
- `pipe_latency_ms` tương ứng `e2e_ms`, chỉ kết thúc khi Python write/flush xong sang FFmpeg; không đo thời gian đến màn hình người xem.
- Mỗi cấu hình chỉ một phiên; không có khoảng tin cậy hoặc kiểm chứng độ chính xác sự kiện. Log chưa lưu đầy đủ phiên bản, weights và tham số model. Không coi file YAML hiện tại là bằng chứng cấu hình lịch sử.
- Tên CPU/GPU cụ thể và dung lượng RAM trong bản báo cáo cũ không có bằng chứng thô đi kèm nên không đưa vào kết luận.

File `report.tex` sẵn có tại thư mục gốc dự án được giữ nguyên.
