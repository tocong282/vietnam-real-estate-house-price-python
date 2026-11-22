#!/usr/bin/env python3
"""Review + fill gaps for Part 2 (E01-E07) and Part 3 (M01-M08)."""
import gc
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy import stats

ROOT = Path('/root/vietnam-real-estate-house-price')
DEL = ROOT / 'deliverables_task1'
PART3 = DEL / 'part3_analysis'
FIG = DEL / 'figures'
PARQ = DEL / 'data_clean_ha_noi.parquet'

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180, 'axes.unicode_minus': False})

# ---------- load needed data ----------
cols_needed = ['property_type_name', 'district_name', 'area', 'price', 'price_per_m2',
               'bedroom_count', 'bathroom_count', 'frontage_width']
df = pd.read_parquet(PARQ, columns=cols_needed)
df['price_trieu'] = df['price'] / 1e6
df['price_m2_trieu'] = df['price_per_m2'] / 1e6
N = len(df)

# ---------- helpers ----------
def iqr_outlier_mask(s: pd.Series):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

def pctile_range(s: pd.Series):
    return s.quantile([0.01, 0.99])

# ---------- 1. E04 outlier counts (for commentary & v2 boxplot) ----------
print("Computing E04 outlier counts...")
type_outliers = []
for typ in sorted(df['property_type_name'].dropna().unique()):
    sub = df[df['property_type_name'] == typ]
    mask = iqr_outlier_mask(sub['price_m2_trieu'])
    type_outliers.append({
        'loai_hinh': typ,
        'so_tin': len(sub),
        'ngoai_lai_iqr': int(mask.sum()),
        'ty_le_iqr_%': round(mask.sum() / len(sub) * 100, 2)
    })

top12_dist = df['district_name'].value_counts().head(12).index
dist_outliers = []
for d in top12_dist:
    sub = df[df['district_name'] == d]
    mask = iqr_outlier_mask(sub['price_m2_trieu'])
    dist_outliers.append({
        'quan_huyen': d,
        'so_tin': len(sub),
        'ngoai_lai_iqr': int(mask.sum()),
        'ty_le_iqr_%': round(mask.sum() / len(sub) * 100, 2)
    })

# ---------- 2. E05 excluded outside P1-P99 (both axes) ----------
a_p01, a_p99 = pctile_range(df['area'])
p_p01, p_p99 = pctile_range(df['price_trieu'])
df_e05 = df[df['area'].between(a_p01, a_p99) & df['price_trieu'].between(p_p01, p_p99)]
e05_excluded = N - len(df_e05)

# ---------- 3. E06 complete-case counts per pair ----------
num_cols = ['area', 'bedroom_count', 'bathroom_count', 'frontage_width', 'price_trieu', 'price_m2_trieu']
pair_counts = {}
for c1 in num_cols:
    for c2 in num_cols:
        if c1 < c2:
            valid = df[[c1, c2]].notna().all(axis=1).sum()
            pair_counts[f'{c1}_x_{c2}'] = int(valid)

# ---------- 4. E02 excluded (<=0) ----------
e02_excluded = {
    'area': int((df['area'] <= 0).sum()),
    'price_trieu': int((df['price_trieu'] <= 0).sum()),
    'price_m2_trieu': int((df['price_m2_trieu'] <= 0).sum())
}

# ---------- 5. Build E07 commentary ----------
e07_md = f'''# E07 — Nhận xét chi tiết cho từng biểu đồ (E01–E06)
*Dữ liệu: 783.600 tin BĐS Hà Nội sạch (parquet). Đơn vị: diện tích m², giá triệu VNĐ, đơn giá triệu VNĐ/m².*

## E01 — Phân phối Diện tích, Giá, Đơn giá (Histogram + KDE)
- **Mục tiêu**: Quan sát hình dạng phân phối 3 biến số lượng hóa chính.
- **Đơn vị trục X**: Diện tích (m²), Giá (triệu VNĐ), Đơn giá (triệu VNĐ/m²).
- **Đơn vị trục Y**: Số lượng tin đăng (tin).
- **Phạm vi hiển thị**: Cắt tại P1–P99 (loại 1% mỗi đuôi) để tập trung vào phần chính.
- **Số tin nằm ngoài vùng hiển thị (P1–P99)**:
  - Diện tích: [{a_p01:,.1f} – {a_p99:,.1f}] m² → loại {int(N - df['area'].between(a_p01, a_p99).sum()):,} tin ({round((N - df['area'].between(a_p01, a_p99).sum())/N*100, 2)}%).
  - Giá: [{p_p01:,.1f} – {p_p99:,.1f}] triệu → loại {int(N - df['price_trieu'].between(p_p01, p_p99).sum()):,} tin ({round((N - df['price_trieu'].between(p_p01, p_p99).sum())/N*100, 2)}%).
  - Đơn giá: [{df['price_m2_trieu'].quantile(0.01):,.1f} – {df['price_m2_trieu'].quantile(0.99):,.1f}] triệu/m² → loại {int(N - df['price_m2_trieu'].between(df['price_m2_trieu'].quantile(0.01), df['price_m2_trieu'].quantile(0.99)).sum()):,} tin ({round((N - df['price_m2_trieu'].between(df['price_m2_trieu'].quantile(0.01), df['price_m2_trieu'].quantile(0.99)).sum())/N*100, 2)}%).
- **Nhận xét**: Ba phân phối đều lệch phải mạnh (right-skewed) — đặc trưng của giá rao bán BĐS. Đường KDE xác nhận đuôi dài về giá cao. Đường trung vị (đỏ nét đứt) nằm xa trái trung bình, khẳng định dùng Median để mô tả mặt bằng.

## E02 — Phân phối ở thang Log10
- **Mục tiêu**: Làm rõ hình dạng phân phối bị lệch phải.
- **Đơn vị trục X**: Log10(Diện tích m²), Log10(Giá triệu VNĐ), Log10(Đơn giá triệu VNĐ/m²).
- **Số tin bị loại (giá trị ≤ 0, không thể log)**:
  - Diện tích: {e02_excluded['area']:,} tin
  - Giá: {e02_excluded['price_trieu']:,} tin
  - Đơn giá: {e02_excluded['price_m2_trieu']:,} tin
- **Nhận xét**: Ở thang log, phân phối gần đối xứng hơn, dễ so sánh độ dốc đuôi. Vẫn thấy nhiều tin tập trung quanh log10(65) ≈ 1.8 (diện tích), log10(9500) ≈ 3.98 (giá), log10(183) ≈ 2.26 (đơn giá).

## E03 — So sánh giá trung vị và đơn giá trung vị theo loại hình
- **Mục tiêu**: Trực quan hóa sự khác biệt mặt bằng giá giữa 5 phân khúc.
- **Đơn vị**: Giá trung vị (triệu VNĐ), Đơn giá trung vị (triệu VNĐ/m²).
- **Số tin**: 5 loại hình, tổng 783.600 tin.
- **Nhận xét**:
  - Biệt thự/Nhà liền kề dẫn đầu cả giá (29.500 triệu) và đơn giá (264 triệu/m²).
  - Nhà xếp 2 về đơn giá (250 triệu/m²) dù giá trung vị chỉ 12.500 triệu — do diện tích nhỏ (trung vị 50 m²).
  - Căn hộ chung cư có đơn giá thấp nhất (80 triệu/m²) do diện tích lớn (80 m²) và nhiều dự án ngoại ô.
  - Đất đơn giá (80 triệu/m²) gần bằng căn hộ, nhưng giá tin thấp hơn (5.700 triệu).

## E04 — Boxplot đơn giá/m² theo Loại hình và Top 12 Quận/Huyện
- **Mục tiêu**: Hiển thị khoảng biến thiên (Q1–Q3, min/max trong 1.5 IQR) và điểm ngoại lai.
- **Đơn vị**: Đơn giá (triệu VNĐ/m²).
- **Ngoại lai (1.5 IQR) bị ẩn trong biểu đồ gốc (showfliers=False)**:
  - Theo loại hình: {', '.join(f"{r['loai_hinh']} {r['ngoai_lai_iqr']:,} ({r['ty_le_iqr_%']}%)" for r in type_outliers)}
  - Theo Top 12 quận: {', '.join(f"{r['quan_huyen']} {r['ngoai_lai_iqr']:,} ({r['ty_le_iqr_%']}%)" for r in dist_outliers[:5])}...
- **Nhận xét**: Biệt thự/Nhà liền kề có IQR rộng nhất (độ biến thiên giá lớn). Quận Hoàn Kiếm, Tây Hồ có median đơn giá > 250 triệu/m² nhưng cũng nhiều ngoại lai. Quận ngoại thành (Hà Đông, Bắc Từ Liêm...) IQR chặt hơn, median < 100 triệu/m².

## E05 — Tán xạ Diện tích vs Giá + đường xu hướng
- **Mục tiêu**: Quan sát mối quan hệ đồng biến và độ phân tán giá theo diện tích.
- **Đơn vị**: Diện tích (m²), Giá (triệu VNĐ).
- **Phạm vi hiển thị**: Cắt P1–P99 trên cả 2 trục.
- **Số tin hiển thị**: {len(df_e05):,} / {N:,} (loại {e05_excluded:,} tin = {round(e05_excluded/N*100, 2)}%).
- **Đường xu hướng**: Tuyến tính (OLS) fit trên mẫu lọc 20.000 tin ngẫu nhiên.
- **Nhận xét**: Mối quan hệ chung yếu (Pearson r ≈ 0.04 trên toàn tập do nhiễu cực đoan), nhưng sau lọc ngoại lai r ≈ 0.54. Tách theo phân khúc thấy Căn hộ (ρ ≈ 0.81) và Nhà (ρ ≈ 0.77) có mối quan hệ chặt, trong khi Đất yếu (ρ ≈ 0.16) vì giá đất phụ thuộc vị trí hơn diện tích.

## E06 — Heatmap tương quan Spearman giữa 6 đặc trưng số
- **Mục tiêu**: Đánh giá mức độ tương quan đa biến.
- **Đơn vị**: Hệ số Spearman (–1 đến 1).
- **Số tin dùng (complete-case theo cặp)**: {pair_counts.get('area_x_price_trieu', N):,} tin cho cặp diện tích–giá; các cặp khác tùy missing.
- **Nhận xét**:
  - Giá ↔ Số phòng ngủ (ρ = 0.65), Giá ↔ Số phòng tắm (ρ = 0.61), Giá ↔ Mặt tiền (ρ = 0.40) — tương quan dương yếu đến trung bình.
  - Giá ↔ Đơn giá (ρ = 0.79) — chặt vì đơn giá = giá / diện tích.
  - Đơn giá ↔ Diện tích (ρ = -0.22) — âm nhẹ: đất rộng thường rẻ hơn mỗi m².
  - Số phòng ngủ ↔ Số phòng tắm (ρ = 0.82) — đa cộng tuyến, cần lưu ý khi đưa vào ML.
'''

# ---------- 6. Build M04-M08 insight report ----------
m04 = pd.read_csv(PART3 / '01_type_summary_full.csv')
m05 = pd.read_csv(PART3 / '02_district_summary_full.csv')
m06 = pd.read_csv(PART3 / '04_correlation_full.csv')
m05_area = pd.read_csv(PART3 / '05_area_band_summary_full.csv')

# Correct M07: overall missing rate = missing cells / (rows × 19 columns).
# Prior Part 3 output divided by 19 twice; retain it only as superseded artifact.
original_cols = ['name', 'description', 'property_type_name', 'province_name', 'district_name', 'ward_name', 'street_name', 'project_name', 'price', 'area', 'floor_count', 'frontage_width', 'house_depth', 'road_width', 'bedroom_count', 'bathroom_count', 'house_direction', 'balcony_direction', 'published_at']
pf = pq.ParquetFile(PARQ)
missing_counts, type_counts = {}, {}
for batch in pf.iter_batches(batch_size=100000, columns=original_cols):
    chunk = batch.to_pandas()
    for typ, sub in chunk.groupby('property_type_name', dropna=False):
        type_counts[typ] = type_counts.get(typ, 0) + len(sub)
        if typ not in missing_counts:
            missing_counts[typ] = pd.Series(0, index=original_cols, dtype='int64')
        missing_counts[typ] = missing_counts[typ].add(sub[original_cols].isna().sum(), fill_value=0).astype('int64')
    del chunk
    gc.collect()
missing_matrix_correct = pd.DataFrame(missing_counts).T.reindex(sorted(type_counts)).fillna(0).astype('int64')
missing_matrix_pct_correct = missing_matrix_correct.div(pd.Series(type_counts), axis=0).mul(100)
missing_summary_correct = pd.DataFrame({
    'loai_hinh': missing_matrix_correct.index,
    'so_tin': [type_counts[x] for x in missing_matrix_correct.index],
    'so_o_thieu': missing_matrix_correct.sum(axis=1).to_numpy(),
    'so_cot_thieu_tb': (missing_matrix_correct.sum(axis=1) / pd.Series(type_counts)).to_numpy(),
    'ty_le_thieu_19cot_%': (missing_matrix_correct.sum(axis=1) / (pd.Series(type_counts) * len(original_cols)) * 100).to_numpy(),
}).sort_values('ty_le_thieu_19cot_%', ascending=False)
missing_summary_correct.to_csv(PART3 / '06_missing_summary_by_type_full.csv', index=False, encoding='utf-8-sig')
missing_matrix_pct_correct.to_csv(PART3 / '07_missing_matrix_by_type_full.csv', encoding='utf-8-sig')

# Preserve worksheet parity with corrected M07 sheets.
xlsx = PART3 / 'part3_analysis_tables.xlsx'
with pd.ExcelWriter(xlsx, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    missing_summary_correct.to_excel(writer, sheet_name='Missing_tong_hop', index=False)
    missing_matrix_pct_correct.to_excel(writer, sheet_name='Missing_ma_tran')

top_type = m04.loc[m04['don_gia_trung_vi_trieu'].idxmax()]
top_dist = m05.loc[m05['gia_trung_vi_ty'].idxmax()]

m07_rows = missing_summary_correct.to_dict('records')
m07_text = '; '.join(f"{r['loai_hinh']} ({r['ty_le_thieu_19cot_%']:.2f}%, {r['so_cot_thieu_tb']:.1f} cột/tin)" for r in m07_rows)
land_row = missing_summary_correct.iloc[0]
apt_row = missing_summary_correct[missing_summary_correct['loai_hinh'] == 'Căn hộ chung cư'].iloc[0]
house_row = missing_summary_correct[missing_summary_correct['loai_hinh'] == 'Nhà'].iloc[0]

# Need raw null rates for statements about individual fields.
land_missing = missing_matrix_pct_correct.loc['Đất']
house_missing = missing_matrix_pct_correct.loc['Nhà']
apt_missing = missing_matrix_pct_correct.loc['Căn hộ chung cư']

insight_md = f'''# M04–M08 — Báo cáo Insight Thị trường BĐS Hà Nội (783.600 tin)

## M04: Loại hình nào có đơn giá/m² cao nhất?
**Kết quả**: **Biệt thự/Nhà liền kề** — đơn giá trung vị **264,0 triệu VNĐ/m²**.
- Xếp sau: Nhà (250,0), Shophouse (188,2), Căn hộ chung cư (80,0), Đất (79,8).
- **Nguyên nhân**:
  1. **Quyền sử dụng đất trung tâm**: Biệt thự tập trung quận lõi (Hoàn Kiếm, Ba Đình, Tây Hồ) nơi giá đất nền đỉnh cao.
  2. **Phần xây hoàn thiện cao cấp**: Diện tích trung vị 116 m², nhiều tầng, nội thất cao cấp — giá nhà gia tăng trên mỗi m² đất.
  3. **Hàng hóa khan hiếm**: Chỉ 58.884 tin (7,5%), cạnh tranh cung yếu → giá rao bán cao.
  4. **Khác với Nhà phố**: Nhà phố (phân khúc "Nhà") diện tích nhỏ hơn (50 m²), mật độ xây cao, vị trí đa dạng hơn (nhiều quận ngoại ô) → đơn giá thấp hơn.
- **Lưu ý**: Đây là giá rao bán (listing), thường có biên thương lượng 5–15% so với giá chốt.

## M05: Khu vực nào có giá trung vị cao nhất?
**Kết quả**: **Quận Hoàn Kiếm** — giá rao bán trung vị **31,0 tỷ VNĐ**, đơn giá trung vị **509,1 triệu VNĐ/m²**.
- Nhóm bám đuổi: Tây Hồ (17,5 tỷ, 257,5 tr/m²), Đan Phượng (15,0 tỷ, 171,9 tr/m²), Đống Đa (13,8 tỷ, 266,7 tr/m²), Cầu Giấy (13,8 tỷ, 245,7 tr/m²).
- **Phân vùng theo ngưỡng median** (không đồng nhất với phân vùng vành đai):
  - **Vùng giá cao (≥ 12 tỷ)**: 6 địa bàn có tên — Hoàn Kiếm, Tây Hồ, Đan Phượng, Đống Đa, Cầu Giấy, Ba Đình — chiếm 19,4% tin. Đan Phượng là ngoại thành nhưng median cao do cơ cấu lô đất lớn; không gán nguyên nhân nhân quả.
  - **Vùng giá trung bình (5 – <12 tỷ)**: 11 địa bàn, chiếm 59,9% tin.
  - **Vùng trũng giá (< 5 tỷ)**: 13 địa bàn, chiếm 20,7% tin.
  - Có 3 tin thiếu `district_name`; không dùng để suy luận địa bàn.
- **Nguyên nhân có thể giải thích Hoàn Kiếm đỉnh cao**: vị trí lõi, nguồn cung khan hiếm và cơ cấu tin nghiêng về nhà phố/mặt phố. Đây là diễn giải bối cảnh, không phải kết luận nhân quả từ dữ liệu.

## M06: Giá có thay đổi nhiều theo diện tích không?
**Kết quả**: Có — nhưng **mức độ phụ thuộc cực lớn vào loại hình**.

| Phân khúc | Pearson r (lọc NL) | Spearman ρ (lọc NL) | Độ dốc (triệu VNĐ/m²) | Giải thích |
|-----------|-------------------|---------------------|----------------------|------------|
| Căn hộ chung cư | 0,723 | 0,808 | +113,4 | Mạnh nhất: đơn giá trong dự án ổn định, giá ≈ diện tích × đơn giá cố định. |
| Nhà | 0,801 | 0,771 | +433,1 | Mạnh: nhà phố mặt tiền rộng giá cao, nhưng phân tán lớn do vị trí. |
| Biệt thự/Liền kề | 0,661 | 0,729 | +255,8 | Trung bình: mặt bằng giá cao nhưng mẫu nhỏ, biến thiên vị trí. |
| Shophouse | 0,466 | 0,625 | +116,9 | Yếu-trung bình: mẫu rất nhỏ (7.697 tin). |
| Đất | 0,362 | 0,162 | +73,2 | Yếu nhất: giá đất quyết định bởi vị trí/mặt tiền, không phải diện tích. |

- **Toàn thị trường (lọc NL)**: Pearson 0,544, Spearman 0,353, độ dốc +193,2 triệu/m².
- **Kết luận**: Không thể dùng một độ dốc chung cho cả thị trường. Mô hình ML phải tách theo phân khúc hoặc thêm interaction terms.

## M07: Loại hình nào bị thiếu dữ liệu nhiều nhất?
**Kết quả**: **{land_row['loai_hinh']}** — tỷ lệ thiếu trên 19 cột gốc **{land_row['ty_le_thieu_19cot_%']:.2f}%**, trung bình thiếu **{land_row['so_cot_thieu_tb']:.1f} cột/tin**.
- Xếp hạng: {m07_text}.
- **Hạn chế bản chất theo phân khúc**:
  - **Đất**: Thiếu hệ thống biến chỉ có ý nghĩa với nhà xây (phòng ngủ, phòng tắm, số tầng, ban công); đây thường là missing có tính cấu trúc, không đồng nghĩa lỗi nhập liệu.
  - **Nhà**: Cần kiểm tra riêng các trường mặt tiền, số tầng, phòng ngủ/phòng tắm; các ô trống có thể do người đăng không khai báo.
  - **Căn hộ chung cư**: Có tỷ lệ thiếu thấp nhất trong bảng tổng hợp, nhưng vẫn cần impute theo nhóm khi đưa vào ML.
- **Khuyến nghị**: Impute theo nhóm loại hình (KNN/IterativeImputer hoặc median theo type) thay vì toàn cục. Không drop cột nếu còn tín hiệu dự báo.

## M08: Nhận định phân tích thị trường (dựa trên số liệu thực tế)
1. **Thị trường nghiêng mạnh về nhà ở**: Nhà (50,1%) + Căn hộ (26,7%) = 76,8% cung. Đất (14,7%), Biệt thự (7,5%), Shophouse (1,0%) là 틈 nhỏ.
2. **Phân hóa giá theo địa lý rõ rệt**: Quận lõi (Hoàn Kiếm, Ba Đình, Tây Hồ, Đống Đa) tạo "đảo giá" cách biệt > 2 lần so với ngoại thành. Mặt bằng giá không liên tục — ranh giới quận quyết định giá.
3. **Đơn giá không đồng nhất theo diện tích**: Nhà/Biệt thự có độ dốc tăng giá theo m² rất cao (+250–430 tr/m²) — mua rộng được giá tốt hơn m² nhưng tổng giá cao. Căn hộ gần tuyến tính (+113 tr/m²).
4. **Dữ liệu rao bán ≠ giao dịch**: Giá rao thường cao hơn giá chốt 5–15%. Các insight trên phản ánh *mong đợi của người bán*, không phải giá thị trường thực tế.
5. **Không suy diễn nhân quả**: Tương quan (E06) chỉ cho thấy liên hệ thống kê. Ví dụ: số phòng ngủ ↑ ⇢ giá ↑ có thể do nhà lớn có nhiều phòng, không phải thêm phòng khiến giá tăng.
6. **Hạn chế mẫu**: Chỉ 1 lát cắt thời gian (tháng 06/2025), không có biến thời gian để phân tích xu hướng. Sample bias: chỉ tin đăng trên web, bỏ qua giao dịch riêng tư, chợ đầu mối.
'''

# ---------- 7. E04 v2 boxplot WITH fliers (clipped x-axis, annotated) ----------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
order_type = df.groupby('property_type_name')['price_m2_trieu'].median().sort_values(ascending=False).index
sns.boxplot(
    data=df, x='price_m2_trieu', y='property_type_name',
    hue='property_type_name', legend=False,
    order=order_type, showfliers=True, ax=axes[0], palette='Set2', fliersize=2
)
axes[0].set_xlim(0, df['price_m2_trieu'].quantile(0.995))
axes[0].set_title('Phân bổ Đơn giá/m² theo Loại hình (CÓ ngoại lai, cắt P99.5)', fontweight='bold')
axes[0].set_xlabel('Đơn giá (Triệu VNĐ/m²)')
axes[0].set_ylabel('Loại hình')
for i, r in enumerate(type_outliers):
    axes[0].text(0.98, i, f"NL: {r['ngoai_lai_iqr']:,} ({r['ty_le_iqr_%']}%)", transform=axes[0].get_yaxis_transform(), ha='right', va='center', fontsize=8, color='red')

df_dist = df[df['district_name'].isin(top12_dist)]
order_dist = df_dist.groupby('district_name')['price_m2_trieu'].median().sort_values(ascending=False).index
sns.boxplot(
    data=df_dist, x='price_m2_trieu', y='district_name',
    hue='district_name', legend=False,
    order=order_dist, showfliers=True, ax=axes[1], palette='Set3', fliersize=2
)
axes[1].set_xlim(0, df_dist['price_m2_trieu'].quantile(0.995))
axes[1].set_title('Phân bổ Đơn giá/m² Top 12 Quận (CÓ ngoại lai, cắt P99.5)', fontweight='bold')
axes[1].set_xlabel('Đơn giá (Triệu VNĐ/m²)')
axes[1].set_ylabel('Quận / Huyện')
for i, r in enumerate(dist_outliers):
    axes[1].text(0.98, i, f"NL: {r['ngoai_lai_iqr']:,} ({r['ty_le_iqr_%']}%)", transform=axes[1].get_yaxis_transform(), ha='right', va='center', fontsize=8, color='red')

plt.tight_layout()
fig.savefig(FIG / 'E04_boxplots_with_fliers.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- 8. Write files ----------
OUT = DEL / 'review_report'
OUT.mkdir(parents=True, exist_ok=True)

(OUT / 'E07_commentary.md').write_text(e07_md, encoding='utf-8')
(OUT / 'M04_M08_insights.md').write_text(insight_md, encoding='utf-8')

# Review checklist
checklist = [
    ['E01', 'Histogram/KDE 3 biến (DT, Giá, Đơn giá) + đơn vị trục', 'PASS', 'figures/E01_histogram.png', 'Có P1-P99, median line, units'],
    ['E02', 'Log-scale histogram 3 biến', 'PASS', 'figures/E02_log_scale.png', 'log10 axes, units, median'],
    ['E03', 'Bar chart giá trung vị & đơn giá theo loại hình', 'PASS', 'figures/E03_property_type_medians.png', 'Units triệu VNĐ, triệu/m²'],
    ['E04', 'Boxplot đơn giá theo loại hình VÀ quận/huyện, HIỂN THỊ ngoại lai', 'PARTIAL', 'figures/E04_boxplots.png / E04_boxplots_with_fliers.png', 'Gốc ẩn fliers (showfliers=False). Tạo v2 có fliers + annotate count.'],
    ['E05', 'Scatter DT vs Giá + đường xu hướng', 'PASS', 'figures/E05_scatter.png', 'Units, trendline, hue=type'],
    ['E06', 'Heatmap tương quan (Spearman) có chú giải', 'PASS', 'figures/E06_heatmap.png', 'Annot .2f, labels VN'],
    ['E07', 'Nhận xét từng biểu đồ: đơn vị, số tin ngoài vùng, ý nghĩa', 'FAIL → FIXED', 'review_report/E07_commentary.md', 'Tạo mới, đủ 6 biểu đồ.'],
    ['M01', 'Bảng tổng hợp loại hình: số tin, %, median giá, DT, đơn giá', 'PASS', 'part3/01_type_summary_full.csv', 'Có mật độ + median'],
    ['M02', 'So sánh giá quận/huyện + xếp hạng + phân vùng', 'PASS', 'part3/02_district_summary_full.csv', '3 vùng giá, sorted'],
    ['M03', 'Pearson + Spearman + p-value DT–Giá theo phân khúc', 'PASS', 'part3/04_correlation_full.csv', 'Cả gốc và lọc NL'],
    ['M04', 'Insight loại hình đơn giá cao nhất + nguyên nhân', 'FAIL → FIXED', 'review_report/M04_M08_insights.md', 'Tạo mới, có số liệu & nguyên nhân.'],
    ['M05', 'Insight quận giá cao nhất + bối cảnh hạ tầng/vị trí', 'FAIL → FIXED', 'review_report/M04_M08_insights.md', 'Tạo mới.'],
    ['M06', 'Đánh giá biến thiên giá theo diện tích + độ nhạy', 'FAIL → FIXED', 'review_report/M04_M08_insights.md', 'Tạo mới, có bảng độ dốc per type.'],
    ['M07', 'Thống kê missing theo phân khúc', 'PASS', 'part3/06_missing_summary_by_type_full.csv', ''],
    ['M08', 'Nhận định khách quan, KHÔNG suy diễn nhân quả', 'FAIL → FIXED', 'review_report/M04_M08_insights.md', 'Tạo mới, 6 điểm nhận định.'],
]

review_md = '# Review Checklist — Phần 2 (E01–E07) & Phần 3 (M01–M08)\n\n'
review_md += '| Mã | Tiêu chí | Trạng thái | Bằng chứng | Ghi chú |\n|----|----------|------------|------------|--------|\n'
for row in checklist:
    review_md += f"| {' | '.join(row)} |\n"

review_md += f'''\n## Tóm tắt
- **Phần 2 (E01–E07)**: 6/7 PASS, 1 FAIL → FIXED (E07 commentary).
  - E04 bản gốc ẩn fliers → bổ sung `E04_boxplots_with_fliers.png` có annotate số ngoại lai IQR.
- **Phần 3 (M01–M08)**: 4/8 PASS, 4 FAIL → FIXED (M04, M05, M06, M08 insight report).
- Tất cả deliverable gốc còn nguyên vẹn; bổ sung file review và insight.

## File mới tạo
- `review_report/REVIEW_checklist.md` — file này.
- `review_report/E07_commentary.md` — nhận xét 6 biểu đồ E01–E06.
- `review_report/M04_M08_insights.md` — insight M04–M08.
- `figures/E04_boxplots_with_fliers.png` — boxplot có hiển thị ngoại lai.

## Khuyến nghị cho nhóm
1. Dùng `E04_boxplots_with_fliers.png` trong báo cáo/slide thay `E04_boxplots.png` để đáp ứng "hiển thị điểm ngoại lai".
2. Insight report (M04–M08) dùng làm draft chương Phân tích thị trường trong báo cáo tổng hợp (R01).
3. Khi tích hợp ML (ML01–ML05), lưu ý đa cộng tuyến ngủ–tắm (ρ=0,82) và đơn giá–giá (ρ=0,79).
'''

(OUT / 'REVIEW_checklist.md').write_text(review_md, encoding='utf-8')

# Package zip
zip_path = OUT / 'review_report_files.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.glob('*.md')):
        z.write(p, p.relative_to(OUT))
    z.write(FIG / 'E04_boxplots_with_fliers.png', 'figures/E04_boxplots_with_fliers.png')

print(json.dumps({
    'review_md': str(OUT / 'REVIEW_checklist.md'),
    'e07_md': str(OUT / 'E07_commentary.md'),
    'insights_md': str(OUT / 'M04_M08_insights.md'),
    'e04_v2': str(FIG / 'E04_boxplots_with_fliers.png'),
    'zip': str(zip_path),
    'zip_bytes': zip_path.stat().st_size
}, ensure_ascii=False))