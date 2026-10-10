# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Tuấn Khanh
- MSSV: 2A202602819
- Email: tuankhan1245@gmail.com
- Link repo (fork): https://github.com/Ataraxiza/K4-L2L3-DAY23-NguyenTuanKhanh-2A202602819-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): ef61ede14458c8dcfba84de7f46a73a1ac892dcb

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `fusion_mode = "compare"`, `frames = [0, 20]`, `segment = "training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"`, `seed = 0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: `precision = 0.9545454545454546`, `recall = 1.0`, `tp = 42`, `fp = 2`, `fn = 0`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `rmse = 0.11370319750154775`, `matches = 34`, `sum_sq_err = 0.4395661821505832`, `ghost_track_frames = 0`, `missed_gt_frames = 8`, `mean_confirmed_tracks = 1.619047619047619`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `rmse = 0.15438964346912915`, `matches = 34`, `sum_sq_err = 0.8104295083578437`, `ghost_track_frames = 0`, `missed_gt_frames = 8`, `mean_confirmed_tracks = 1.619047619047619`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: cả hai mode có cùng số matches và cùng số ghost/miss, nghĩa là camera không làm tăng độ phủ hay tạo ghost trong run hiện tại; fused có RMSE lớn hơn LiDAR chỉ khoảng 0.0407 m, vẫn nằm trong ngưỡng chênh lệch không quá 0.05 m cho chuẩn nhất quán. Một lần update camera chỉ tinh chỉnh trạng thái và không làm thay đổi lifecycle vì camera không cộng/trừ score hay tạo/xoá track.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - LiDAR đo trực tiếp không gian 3D `(x, y, z)` nên innovation là sai lệch 3 chiều giữa state và measurement, với `R` tùy thuộc vào độ tin cậy của detector và tương ứng với vector đo 3D.
   - Camera đo 2D trên ảnh, nên `z` gồm 2 tọa độ hình chiếu từ thế giới về không gian cảm biến/ảnh, còn `R` chỉ là ma trận hiển thị cho nhiễu đo 2D. Việc này đòi hỏi mô hình `h(x)` và Jacobian `H` khác với LiDAR, và chỉ có thể cập nhật khi xe nằm trong FOV và có độ sâu hợp lệ.

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Gating loại các cặp đo-track quá xa về mặt thống kê, tránh gán nhầm giữa track và measurement không khả thi dù khoảng cách Euclidean nhìn có vẻ nhỏ.
   - Mahalanobis dùng ma trận hiệp phương sai `S` nên tính đến độ bất định của state và measurement; các cặp có khoảng cách chuẩn hóa `d²` vượt ngưỡng χ² sẽ bị loại, giảm nhiễu và bảo vệ track khỏi việc ghép sai khi `P` lớn hoặc đo không tương thích.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Đây là `track-then-fuse`: mỗi frame predict một lần, thực hiện association LiDAR và update EKF, sau đó mới gán camera và update tiếp trên cùng state. Điều này phù hợp với thứ tự trong `fusion-run-lab` và log: predict → AssocL → AssocC, không tạo ra một measurement hợp nhất trước khi tracking.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Residual giữa measurement và dự đoán camera sẽ tăng bất thường, và nhiều cặp đo-track sẽ fall outside gate hoặc bị loại bởi `is_in_field_of_view`/`chi2_gate`.
   - Khi calibration sai, hệ số projection và Jacobian không còn khớp với dữ liệu thực, nên innovation trung bình lớn hơn kỳ vọng, khiến updates camera tác động lệch hoặc khiến fused trở nên xấu hơn LiDAR.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   - Vì hàm association phải luôn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` dù không có measurement nào; logic vòng đời phải biết đây là LiDAR hay camera để xác định chính xác score/init/delete và không gây xung đột kiểu dữ liệu.
   - LiDAR quyết định khởi tạo, xác nhận, score, và xoá track; camera chỉ tinh chỉnh EKF trên các track đã có mà không thay đổi lifecycle. Nếu không có `sensor` rõ ràng, hệ thống không biết nên xử lý như thế nào khi frame rỗng hoặc khi chỉ làm camera update.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - Một track được xác nhận khi score vượt ngưỡng xác nhận theo LiDAR; miss trong FOV chỉ trừ điểm theo một cửa sổ cho đến khi score giảm quá ngưỡng xóa, còn track confirmed không bị hạ ngay lập tức vì một miss.
   - Track bị xoá khi `P[0,0]` hoặc `P[1,1]` vượt `max_P`, hoặc track confirmed có `score < delete_threshold`, hoặc track chưa confirmed có `score <= 0`.
   - Camera không thay đổi score, không khởi tạo hoặc xóa track; nó chỉ cập nhật state EKF nếu mắt nhìn được và phát hiện xe nằm trong FOV.

## Bonus (không bắt buộc)

- Không.

## Khai báo sử dụng AI (bắt buộc)

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): GitHub Copilot.
- Dùng cho phần nào (hàm, câu hỏi, debug): hỗ trợ đọc logic E–H, gỡ lỗi association/track lifecycle và kiểm tra các điều kiện báo cáo trước khi chạy `pytest` và `check_submission.py`.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): chạy `pytest student/tests -q`, kiểm tra `student/artifacts/*.json` và `grade_run*.log` đối chiếu với `metrics.json`, và chạy `python tools/check_submission.py` để xác nhận các ràng buộc nộp.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
