# Checkpoints

Bài lab làm **cá nhân**, 2 giờ trên lớp. Mỗi checkpoint ghi rõ **cần làm**, **sản phẩm**,
**cần hiểu** và **tự kiểm tra**. Lab coach đi vòng quanh lớp cuối mỗi checkpoint;
chưa qua checkpoint nào thì báo ngay, đừng im lặng làm tiếp.

> **Commit sau mỗi checkpoint** với message `CPx: <việc vừa làm>`, ví dụ
> `CP1: EKF predict and update`. Lịch sử commit là bằng chứng bạn tự làm theo tiến
> độ; bài chỉ có 1–2 commit dồn cuối giờ sẽ bị gọi vấn đáp (xem [RUBRIC.md](RUBRIC.md) mục 4).

Mọi lệnh chạy từ gốc repo, sau khi đã nạp biến môi trường: `set -a; source .env; set +a`.

---

## CP0 — Chuẩn bị (làm ở nhà, trước buổi học)

**Cần làm**
1. Fork repo, đặt tên `K4-L2L3-DAY23-<HoVaTen>-<MSSV>-SensorFusion`, clone và thêm remote `upstream` — [README.md](README.md) mục 5, bước 1.
2. Cài môi trường, tạo `.env` từ `.env.example` và `paths.yaml` — README mục 5, bước 2–3.
3. Tải segment mặc định và weights từ [Google Drive Day23-Fusion](https://drive.google.com/drive/folders/1XTXA60c9gV6rkK6fWohMyHw8qpsfYBlE?usp=sharing) theo [data/README.md](data/README.md).
4. Đọc §2 của [docs/HUONG_DAN_KY_THUAT.md](docs/HUONG_DAN_KY_THUAT.md) và lướt Part A–D trong `student/workspace/`.
5. Điền phần "Thông tin học viên" trong `student/SUBMISSION.md`.

**Sản phẩm**
- Bản fork trên GitHub, đã clone về máy.
- `student/config/paths.yaml` trỏ đúng dữ liệu (file này không commit).

**Cần hiểu**
- Detector LiDAR trả hộp 3D mỗi frame; tracker lấy **tâm hộp** làm đo `z = (x, y, z)`.
- Lab là **track-then-fuse**: mỗi frame predict một lần, update LiDAR (AssocL), rồi update camera (AssocC).

**Tự kiểm tra**
```bash
pytest student/tests -q
fusion-run-lab --help
ls data/Waymo/*.tfrecord data/weights/pretrained_fpn-resnet/*.pth
```
- [X] `pytest` không có dòng `failed`; test E–H báo `xfailed`.
- [X] `fusion-run-lab --help` in hướng dẫn, không lỗi import.
- [X] Thấy file `.tfrecord` và `.pth`.

---

## CP1 — Part E: EKF (0:00 – 0:25)

**Cần làm** — `student/workspace/kalman.py`: `build_F`, `build_Q`, `ekf_predict`,
`innovation`, `innovation_covariance`, `ekf_update` theo các dòng `# vi: TODO Part E`.

**Sản phẩm** — `test_kalman.py` pass.

**Cần hiểu**
- `F` mô hình vận tốc không đổi với `dt`; `Q` lớn lên theo `dt` và hệ số nhiễu `q`.
- `γ = z − h(x)`, `S = H P Hᵀ + R`, `K = P Hᵀ S⁻¹`. LiDAR có `H` tuyến tính 3×6; camera có `H` là Jacobian.
- `dt` và `q` đọc từ `get_tracking_params()`, không viết cứng.

**Tự kiểm tra**
```bash
pytest student/tests/test_kalman.py -v
```
- [ ] Tất cả `passed`. Commit `CP1: ...`.

---

## CP2 — Part G: mô hình đo camera (0:25 – 0:50)

**Cần làm** — `student/workspace/camera_fusion.py`: `is_in_field_of_view`,
`camera_measurement_prediction` (pinhole `h(x)`), `build_camera_measurement` (`z`, `R`) theo `# vi: TODO Part G`.

**Sản phẩm** — test camera pass.

**Cần hiểu**
- Đổi điểm sang hệ toạ độ cảm biến `p_s = R p + t` trước khi chiếu.
- Toạ độ không hữu hạn hoặc độ sâu `x_s ≤ 1e-6` không chiếu được: `is_in_field_of_view` trả `False`, hàm chiếu `raise ValueError` có ngữ cảnh.
- FOV camera suy ra từ intrinsics và bề rộng ảnh; kiểm tra FOV **trước** khi chiếu.

**Tự kiểm tra**
```bash
pytest student/tests -v -k "camera_invalid or camera_front or camera_jacobian or camera_fov or student_camera"
```
- [ ] Tất cả `passed`. Commit `CP2: ...`.

---

## CP3 — Part F: association (0:50 – 1:15)

**Cần làm** — `student/workspace/association.py`: `mahalanobis_distance`, `chi2_gate`,
`association_cost_matrix`, `pick_next_pair` (greedy) và `associate_and_update` theo `# vi: TODO Part F`.
Lấy module Kalman bằng `load_workspace_module("kalman")`, không `import kalman`.

**Sản phẩm** — test association pass.

**Cần hiểu**
- `d² = γᵀ S⁻¹ γ`; cặp có `d²` vượt ngưỡng χ² (theo số chiều đo) bị loại.
- Cặp ngoài FOV bị loại **trước** khi tính Mahalanobis.
- `associate_and_update` luôn gọi `manager.manage_tracks(...)`, kể cả khi không có đo.

**Tự kiểm tra**
```bash
pytest student/tests -v -k "mahalanobis or greedy or out_of_fov or association"
```
- [ ] Tất cả `passed`. Commit `CP3: ...`.

---

## CP4 — Part H: vòng đời track (1:15 – 1:35)

**Cần làm** — `student/workspace/track_management.py`: `init_track_state_from_meas`,
`update_track_score`, `should_delete_track` theo `# vi: TODO Part H`.

**Sản phẩm** — toàn bộ `student/tests` pass.

**Cần hiểu**
- Chỉ lượt **LiDAR** quyết định tồn tại: hit `+1/window` (tối đa 1), miss trong FOV `−1/window`.
- Track confirmed không bị hạ trạng thái chỉ vì một miss.
- Camera chỉ update trạng thái EKF: không cộng/trừ score, không tạo, không xoá track.

**Tự kiểm tra**
```bash
pytest student/tests -q
```
- [ ] Không còn `failed` hay `xfailed`. Commit `CP4: ...`.

---

## CP5 — Part I: chạy trên Waymo (1:35 – 1:50)

**Cần làm**
1. Trong `student/config/paths.yaml` đặt `frame_start: 0`, `frame_end: 198` (chạy cả segment). Chạy thử nhanh với `frame_end: 20` trước nếu muốn.
2. Chạy:
   ```bash
   fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
   ```

**Sản phẩm** — `student/artifacts/metrics.json`, `grade_run.log` và các file
`metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`, `grade_run_fused.log`.

**Cần hiểu**
- RMSE chỉ tính trên confirmed tracks ghép một-một với xe thật trong cổng XY 2 m; đọc cùng `matches`, `ghost_track_frames`, `missed_gt_frames`.
- Fused không bắt buộc tốt hơn LiDAR; nó **không được xấu hơn** quá 0.05 m (xem [RUBRIC.md](RUBRIC.md)).

**Tự kiểm tra**
- [ ] `metrics.json` có `"fusion_mode": "compare"`, `tracking.lidar` và `tracking.fused` đều khác `null`.
- [ ] `tracking.lidar.rmse` ≤ 0.45 m là mức đầy đủ điểm; lớn hơn nhiều thì xem lại E–H.
- [ ] Không sửa tay file trong `artifacts/`. Commit `CP5: ...` (gồm cả artifacts).

---

## CP6 — Báo cáo và nộp (1:50 – 2:00)

**Cần làm**
1. Điền `student/SUBMISSION.md`: số liệu từ `metrics.json`, 6 câu giải thích E–H, khai báo AI.
2. Chạy `python tools/check_submission.py` và sửa đến khi báo `KẾT QUẢ: SẴN SÀNG NỘP`.
3. Commit `CP6: ...`, `git push`, nộp trên LMS theo [SUBMISSION.md](SUBMISSION.md).

**Sản phẩm** — fork đã push; link repo và commit hash đã nộp trên LMS.

**Tự kiểm tra**
- [ ] `git status` sạch; `git log origin/main -1` trùng commit hash bạn nộp.
