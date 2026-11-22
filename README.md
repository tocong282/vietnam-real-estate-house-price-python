# Phân tích và dự đoán giá bất động sản tại Hà Nội kết hợp Machine Learning

**Nhóm 01 - Lập trình Python** · GVHD: Lương Ngọc Sơn · 2026

Dự án xây dựng pipeline hoàn chỉnh từ dữ liệu tin rao bán thô đến mô hình dự đoán **giá rao bán** bất động sản Hà Nội, có kiểm soát rò rỉ dữ liệu và diễn giải mô hình bằng Tree SHAP.

> Giá dự đoán là **giá rao bán**, không phải giá giao dịch và không thay thế thẩm định giá.

---

## 1. Dữ liệu

| Mục | Giá trị |
|---|---|
| Nguồn | Kaggle `locngodev/vietnam-real-estate` |
| File | `raw_data_all.csv`, 19 cột, ~3,2 GB |
| Dữ liệu thô | 3.500.744 dòng |
| Sau lọc Hà Nội | 1.140.542 dòng |
| Giá > 0 và diện tích > 0 | 1.035.822 dòng |
| Sau loại trùng nghiệp vụ | **783.600 dòng** |
| Dùng để fit mô hình | 780.353 dòng |

**Không kèm dữ liệu trong gói nộp** vì dung lượng quá lớn. Chạy trực tiếp trên Kaggle Notebook:

1. Tạo Notebook mới trên Kaggle.
2. Panel phải → **Add Input** → chọn dataset `locngodev/vietnam-real-estate`.
3. Mở notebook `notebooks/kaggle_hanoi_all_property_types.ipynb`, chạy toàn bộ.

Notebook tự dò đường dẫn `/kaggle/input/**/raw_data_all.csv`, không cần cấu hình thêm.

Chạy local (nếu đã tải CSV):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python prepare.py --raw data/raw_data_all.csv --out data/clean.parquet
```

## 2. Cấu trúc gói nộp

```
.
├── README.md                  # file này
├── requirements.txt           # phiên bản thư viện đã chạy
├── BANG_PHAN_CONG.pdf         # bảng phân công công việc
├── notebooks/
│   ├── kaggle_hanoi_all_property_types.ipynb   # notebook chính chạy trên Kaggle
│   └── kaggle_vietnam_house_price.ipynb        # notebook khảo sát ban đầu
├── src/
│   ├── prepare.py             # lọc phạm vi, chuẩn hóa, gán group key
│   ├── task1_reports.py       # data quality, funnel, missing, ngoại lai
│   ├── fix_clean.py           # loại trùng nghiệp vụ
│   ├── run_notebook_eda.py    # EDA E01–E06
│   ├── run_part3_analysis.py  # phân tích M01–M08
│   ├── ml_pipeline.py         # so sánh 6 mô hình, Group Split
│   ├── tune_ml.py             # tuning LightGBM
│   ├── final_eval.py          # đánh giá cuối, phân khúc, Time Split
│   ├── enhance_eval.py        # SHAP, hình đánh giá
│   ├── train.py               # baseline tối thiểu
│   └── check.py               # kiểm tra rò rỉ và target
├── results/                   # bảng kết quả CSV
├── BAO_CAO_NHOM01_5_8_TRANG.pdf  # báo cáo 7 trang
└── report/
    ├── main.tex               # nguồn LaTeX
    ├── ptit-logo.jpg          # logo bìa
    └── figures/               # biểu đồ trong báo cáo
```

## 3. Quy trình

```
raw_data_all.csv
  → lọc province_name == "Hà Nội"
  → giữ price > 0 và area > 0
  → loại trùng theo khóa nghiệp vụ
  → 783.600 dòng sạch
  → lọc cực đoan đơn vị (chỉ cho tập fit) → 780.353 dòng
  → GroupShuffleSplit 80/20 theo project_name/name
  → 6 mô hình so sánh → LightGBM tuned
```

### Chống rò rỉ dữ liệu

- Không dùng `price_per_m2` trong tập đặc trưng.
- Không dùng `name`, `description` làm đặc trưng (có thể chứa giá trong tiêu đề).
- `name` chỉ dùng tạo group key.
- Chuẩn hóa, mã hóa học tham số **chỉ trên train**.
- Tuning bằng GroupKFold trên train; test giữ nguyên đến lần đánh giá cuối.

## 4. Kết quả

### Group Split (test n = 157.427)

| Mô hình | MAE (triệu) | RMSE (triệu) | MedAE (triệu) | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| **LightGBM** | **4.010,9** | **11.678,6** | **1.346,4** | **0,800** | **22,5%** |
| HistGradientBoosting | 4.171 | 12.217 | 1.397 | 0,781 | 23,2% |
| XGBoost | 4.199 | 12.150 | 1.468 | 0,783 | 23,2% |
| Random Forest | 4.883 | 14.773 | 1.651 | 0,680 | 26,9% |
| Ridge Regression | 6.857 | 22.315 | 2.489 | 0,269 | 38,5% |
| Baseline median | 9.563 | 25.466 | 4.000 | 0,048 | 55,9% |

Đơn vị: triệu VNĐ. `4.010,9` ≈ 4,01 tỷ; `1.346,4` ≈ 1,35 tỷ.

### Time Split (train 2025 = 630.403, test 2026 = 149.950)

MAE 4.769 · RMSE 13.556 · MedAE 1.705 · R² 0,790 · MAPE 23,5%

### Yếu tố ảnh hưởng (Tree SHAP)

| Đặc trưng | Tỷ lệ |
|---|---:|
| `area` | 30,6% |
| `property_type_name` | 27,3% |
| `district_name` | 16,0% |
| `ward_name` | 6,6% |

Bốn đặc trưng đầu chiếm 80,5% tổng ảnh hưởng.

## 5. Cấu hình mô hình bàn giao

```text
objective        = regression_l1
n_estimators     = 1200
learning_rate    = 0.025
num_leaves       = 255
min_child_samples = 80
feature_fraction = 0.85
bagging_fraction = 0.90
lambda_l1        = 1
lambda_l2        = 10
```

Target huấn luyện `log1p(price)`, dự báo ngược bằng `expm1()`.

## 6. Giới hạn

- Giá là giá rao bán, không phải giá chốt giao dịch.
- Không có ID bất động sản; nhóm theo tiêu đề chuẩn hóa chỉ là proxy chống trùng.
- Tập test 2026 chỉ phủ quý I/2026, chưa đại diện toàn năm.
- Sai số lệch phải: MedAE thấp hơn MAE nhiều lần cho thấy một số ít tin giá trị lớn có sai số rất cao.
- Mô hình không nắm được yếu tố pháp lý, hướng view, chất lượng nội thất.

## 7. Môi trường

CPU không hỗ trợ AVX/AVX2 nên bộ thư viện được pin về bản tương thích:

```text
numpy==1.26.4
pandas==2.2.3
scipy==1.13.1
scikit-learn==1.5.2
lightgbm==4.7.0
xgboost==3.4.1
pyarrow==17.0.0
matplotlib==3.9.2
seaborn==0.13.2
joblib==1.4.2
```

Không cần GPU. Toàn bộ pipeline chạy được trên Kaggle CPU.
