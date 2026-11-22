# BẢNG PHÂN CÔNG CÔNG VIỆC

**Đề tài:** Phân tích và dự đoán giá bất động sản tại Hà Nội kết hợp Machine Learning
**Nhóm:** Nhóm 01 - Lập trình Python
**GVHD:** Lương Ngọc Sơn
**Thời gian thực hiện:** 11/2026

## 1. Thành viên

| STT | Họ và tên | MSSV | Vai trò |
|---:|---|---|---|
| 1 | Tô Thành Công | B25DCTC024 | Nhóm trưởng |
| 2 | Bùi Quốc Anh | B25DCTC004 | Thành viên |
| 3 | Đặng Đức Anh | B25DCTC013 | Thành viên |
| 4 | Trần Hoàng Dũng | B25DCTC030 | Thành viên |
| 5 | Dương Minh Tuấn | B25DCTC132 | Thành viên |

## 2. Phân công theo hạng mục

| Hạng mục | Nội dung | File phụ trách | Người thực hiện | Tỷ lệ |
|---|---|---|---|---:|
| 1 | Thu thập, khảo sát schema, làm sạch dữ liệu, loại trùng | `task1_reports.py`, `fix_clean.py`, `verify_clean.py` | Tô Thành Công | 20% |
| 2 | EDA E01–E06: phân phối, log-scale, boxplot, scatter, heatmap | `run_notebook_eda.py` | Bùi Quốc Anh | 20% |
| 3 | Phân tích M01–M08: phân khúc, quận/huyện, vành đai, missing | `run_part3_analysis.py` | Đặng Đức Anh | 20% |
| 4 | Pipeline ML, so sánh 6 mô hình, tuning LightGBM | `ml_pipeline.py`, `tune_ml.py` | Trần Hoàng Dũng | 20% |
| 5 | Đánh giá mô hình, Time Split, SHAP, viết báo cáo LaTeX | `final_eval.py`, `enhance_eval.py`, `paper/main.tex` | Dương Minh Tuấn | 20% |

## 3. Phân công theo giai đoạn

| Giai đoạn | Thời gian | Công việc | Người thực hiện |
|---|---|---|---|
| Khảo sát | 01/11 – 06/11 | Đọc dataset, xác minh schema, viết README | Tô Thành Công |
| Làm sạch | 07/11 – 16/11 | Lọc Hà Nội, xử lý giá/diện tích, loại trùng, audit missing | Tô Thành Công, Bùi Quốc Anh |
| EDA | 17/11 – 21/11 | Sinh 6 biểu đồ E01–E06, nhận xét từng hình | Bùi Quốc Anh |
| Phân tích sâu | 17/11 – 21/11 | Thống kê mô tả theo loại hình, quận, vành đai, nhóm diện tích | Đặng Đức Anh |
| Mô hình hóa | 17/11 – 21/11 | Baseline, Ridge, RF, HGB, XGBoost, LightGBM; Group Split; tuning | Trần Hoàng Dũng |
| Đánh giá | 21/11 | Metrics, phân khúc, Time Split, Tree SHAP, kiểm tra leakage | Dương Minh Tuấn |
| Báo cáo | 21/11 – 22/11 | Viết LaTeX, vẽ hình, rà soát số liệu, đóng gói | Dương Minh Tuấn, Tô Thành Công |

## 4. Ghi chú

- Tỷ lệ đóng góp tính theo khối lượng công việc và mức độ tham gia thảo luận.
- Mọi thành viên đều tham gia kiểm tra chéo kết quả trước khi nộp.
- Nhóm trưởng chịu trách nhiệm tổng hợp, kiểm tra tính nhất quán số liệu giữa báo cáo và file kết quả.

| Họ và tên | Chữ ký |
|---|---|
| Tô Thành Công | |
| Bùi Quốc Anh | |
| Đặng Đức Anh | |
| Trần Hoàng Dũng | |
| Dương Minh Tuấn | |
