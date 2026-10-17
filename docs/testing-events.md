# Unit test đếm vạch và vùng cấm

Các test gọi trực tiếp `EventEngine.process()` với `TrackingPacket` giả, bbox và thời gian được chỉ định rõ. Frame là mảng NumPy; phép kiểm tra hình học chạy thật bằng CPU. Bộ test không khởi động model, camera, FFmpeg hoặc database.

## Chạy test

Từ thư mục gốc project:

```powershell
conda activate free_env
python -m pytest tests/test_line_crossing.py tests/test_prohibited_area.py tests/test_event_state.py -v
```

Chạy riêng phần đếm vạch hoặc vùng cấm:

```powershell
python -m pytest tests/test_line_crossing.py -v
python -m pytest tests/test_prohibited_area.py -v
```

Chạy riêng ca chạm vạch rồi đi qua/quay lại:

```powershell
python -m pytest tests/test_line_crossing.py::test_points_on_line_preserve_the_last_nonzero_side -v
```

Chạy toàn bộ kiểm thử project:

```powershell
python -m pytest -q
```

Nếu sandbox không ghi được thư mục tạm mặc định, dùng `--basetemp` trỏ vào một thư mục mới bên trong `.test-runs/` như hướng dẫn trong README.

## Các hành vi được kiểm tra

| File | Các tình huống chính |
|---|---|
| `tests/test_line_crossing.py` | Hai chiều vượt vạch, ngang/dọc/chéo, đảo điểm đầu/cuối, đi cùng phía, dừng trên vạch, chạm rồi quay lại, bắt đầu ngay trên vạch, giao đúng đầu mút, đi ngoài đoạn vạch, nhiều ID đổi thứ tự, qua lại nhiều lần, tâm bbox, normalize/pixel |
| `tests/test_prohibited_area.py` | Bắt đầu trong/ngoài vùng, cạnh/đỉnh thuộc vùng, ở trong lâu chỉ tạo entry lần đầu, ra rồi vào lại, nhiều ID, tâm bbox thay vì phần bbox giao vùng, polygon lõm, đảo thứ tự đỉnh, normalize ở nhiều kích thước, thiếu polygon, bảo toàn ảnh/bbox |
| `tests/test_event_state.py` | Frame rỗng, TTL đúng ngưỡng và quá ngưỡng, dọn tất cả state khi ID hết hạn, ID quay lại, reset giữ số đếm, state riêng từng camera, vượt vạch và vào vùng cùng frame, tắt event, bảo toàn metadata/frame |

Các test chốt quy ước hiện tại:

- Tâm bbox quyết định vị trí đối tượng.
- Chỉ giao với đoạn vạch thực tế mới tăng bộ đếm; giao tại hai đầu mút được tính.
- `left_to_right` là đổi từ phía âm sang phía dương của vạch có hướng. Đảo hai điểm vạch sẽ đảo tên hướng và bộ đếm tương ứng.
- Frame đầu tiên chỉ thiết lập phía của track, không được tính là một lần vượt vạch.
- Đứng trên vạch không làm mất phía khác 0 trước đó.
- Cạnh và đỉnh polygon được tính là nằm trong vùng cấm.
- `zone_entry` chỉ bật khi track bắt đầu ở trong vùng hoặc chuyển từ ngoài vào trong; ở trong liên tục không tạo entry mới.
- State chỉ hết hạn khi khoảng cách thời gian lớn hơn `state_ttl`; đúng bằng TTL vẫn giữ.
- `reset()` xóa danh tính/vị trí cũ nhưng giữ bộ đếm tích lũy.

## Cách thêm một ca mới

Fixture `tracking_packet_factory` trong `tests/conftest.py` tạo packet từ các bộ `(track_id, center_x, center_y)`. Bbox mặc định có nửa chiều rộng/cao bằng 2 pixel. Hai packet có thể dùng cùng ID để mô phỏng một đối tượng di chuyển:

```python
from src.event_engine import EventEngine
from src.schemas import Event


def test_example_crossing(tracking_packet_factory):
    engine = EventEngine(Event(line=[[10, 50], [90, 50]]))

    before = tracking_packet_factory([(1, 50, 40)], timestamp=1)
    after = tracking_packet_factory([(1, 50, 60)], timestamp=2)

    engine.process(before)
    result = engine.process(after)

    assert result.events[1]["crossed"] is True
    assert result.events[1]["direction"] == "left_to_right"
    assert result.right_count == 1
    assert result.left_count == 0
```

`timestamp` dùng giây mô phỏng, không cần `sleep()`. Dùng nhiều ID trong một packet để kiểm tra nhiều đối tượng; dùng `objects=()` để mô phỏng frame không còn đối tượng.
