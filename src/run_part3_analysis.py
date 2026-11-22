from pathlib import Path
import gc
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
import seaborn as sns

ROOT = Path('/root/vietnam-real-estate-house-price')
OUT = ROOT / 'deliverables_task1' / 'part3_analysis'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
PARQ = ROOT / 'deliverables_task1' / 'data_clean_ha_noi.parquet'

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180, 'axes.unicode_minus': False})

# Numeric/category view for descriptive statistics and plots.
plot_cols = ['property_type_name', 'district_name', 'area', 'price', 'price_per_m2',
             'bedroom_count', 'bathroom_count', 'frontage_width']
df = pd.read_parquet(PARQ, columns=plot_cols)
df['price_ty'] = df['price'] / 1e9
df['ppm_trieu'] = df['price_per_m2'] / 1e6
N = len(df)

# ---------- A. Type summary ----------
type_summary = df.groupby('property_type_name', dropna=False).agg(
    so_tin=('price', 'count'),
    gia_trung_vi_ty=('price_ty', 'median'),
    gia_q1_ty=('price_ty', lambda x: x.quantile(.25)),
    gia_q3_ty=('price_ty', lambda x: x.quantile(.75)),
    gia_trung_binh_ty=('price_ty', 'mean'),
    don_gia_trung_vi_trieu=('ppm_trieu', 'median'),
    don_gia_trung_binh_trieu=('ppm_trieu', 'mean'),
    dien_tich_trung_vi_m2=('area', 'median'),
).sort_values('don_gia_trung_vi_trieu', ascending=False).reset_index()
type_summary['ty_le_%'] = type_summary['so_tin'] / N * 100
type_summary.to_csv(OUT / '01_type_summary_full.csv', index=False, encoding='utf-8-sig')

# Chart A: counts, total price median, unit price median.
plot_type = type_summary.sort_values('so_tin', ascending=True)
fig, ax = plt.subplots(1, 3, figsize=(17, 6))
for axis, col, title, xlabel, color in [
    (ax[0], 'so_tin', 'Số lượng tin đăng', 'Số tin', '#2e75b6'),
    (ax[1], 'gia_trung_vi_ty', 'Giá trung vị', 'Tỷ VND', '#4472c4'),
    (ax[2], 'don_gia_trung_vi_trieu', 'Đơn giá trung vị', 'Triệu VND/m²', '#1f4e79'),
]:
    axis.barh(plot_type['property_type_name'], plot_type[col], color=color)
    axis.set_title(title, fontweight='bold')
    axis.set_xlabel(xlabel)
    axis.grid(axis='x', alpha=.25)
    for y, value in enumerate(plot_type[col]):
        axis.text(value, y, f' {value:,.1f}', va='center', fontsize=8)
fig.suptitle('Phân khúc bất động sản Hà Nội — toàn bộ dữ liệu sạch', fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / '01_type_comparison.png', bbox_inches='tight')
plt.close(fig)

# ---------- B. District and belt summary ----------
district_summary = df.groupby('district_name', dropna=False).agg(
    so_tin=('price', 'count'),
    gia_trung_vi_ty=('price_ty', 'median'),
    don_gia_trung_vi_trieu=('ppm_trieu', 'median'),
    dien_tich_trung_vi_m2=('area', 'median'),
).sort_values('gia_trung_vi_ty', ascending=False).reset_index()
district_summary['ty_le_%'] = district_summary['so_tin'] / N * 100
district_summary['phan_vung'] = np.select(
    [district_summary['gia_trung_vi_ty'] >= 12,
     district_summary['gia_trung_vi_ty'] >= 5],
    ['Vùng giá cao (>= 12 tỷ)', 'Vùng giá trung bình (5 - <12 tỷ)'],
    default='Vùng trũng giá (< 5 tỷ)')
district_summary.to_csv(OUT / '02_district_summary_full.csv', index=False, encoding='utf-8-sig')

core = {'Hoàn Kiếm', 'Ba Đình', 'Đống Đa', 'Hai Bà Trưng'}
inner = core | {'Tây Hồ', 'Cầu Giấy', 'Thanh Xuân', 'Hoàng Mai', 'Long Biên', 'Nam Từ Liêm', 'Bắc Từ Liêm', 'Hà Đông'}
def belt(d):
    if d in core:
        return '1. 4 quận lõi trung tâm'
    if d in inner:
        return '2. 8 quận nội thành mở rộng'
    return '3. Huyện/thị xã ngoại thành'

df['vanh_dai'] = df['district_name'].map(belt)
belt_summary = df.groupby('vanh_dai').agg(
    so_tin=('price', 'count'),
    gia_trung_vi_ty=('price_ty', 'median'),
    don_gia_trung_vi_trieu=('ppm_trieu', 'median'),
    dien_tich_trung_vi_m2=('area', 'median'),
).reset_index()
belt_summary['ty_le_%'] = belt_summary['so_tin'] / N * 100
belt_summary.to_csv(OUT / '03_belt_summary_full.csv', index=False, encoding='utf-8-sig')

dplot = district_summary[district_summary['so_tin'] >= 20].sort_values('gia_trung_vi_ty', ascending=True)
fig, ax = plt.subplots(1, 2, figsize=(17, 12))
colors = {'Vùng giá cao (>= 12 tỷ)': '#c0392b', 'Vùng giá trung bình (5 - <12 tỷ)': '#2980b9', 'Vùng trũng giá (< 5 tỷ)': '#27ae60'}
for y, (_, row) in enumerate(dplot.iterrows()):
    ax[0].barh(row['district_name'], row['gia_trung_vi_ty'], color=colors[row['phan_vung']])
    ax[1].barh(row['district_name'], row['don_gia_trung_vi_trieu'], color=colors[row['phan_vung']])
    ax[0].text(row['gia_trung_vi_ty'], y, f" {row['gia_trung_vi_ty']:.2f}", va='center', fontsize=7)
    ax[1].text(row['don_gia_trung_vi_trieu'], y, f" {row['don_gia_trung_vi_trieu']:.1f}", va='center', fontsize=7)
ax[0].set_title('Giá trung vị theo quận/huyện (tỷ VND)', fontweight='bold')
ax[1].set_title('Đơn giá trung vị theo quận/huyện (triệu VND/m²)', fontweight='bold')
ax[0].set_xlabel('Tỷ VND'); ax[1].set_xlabel('Triệu VND/m²')
fig.suptitle('Phân hóa giá theo địa bàn — chỉ hiển thị địa bàn có n >= 20', fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / '02_district_comparison.png', bbox_inches='tight')
plt.close(fig)

# ---------- C. Correlation and area bands ----------
def correlation_row(data, group, stage):
    data = data[['area', 'price_ty']].replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data['area'] > 0) & (data['price_ty'] > 0)]
    if len(data) < 3 or data['area'].nunique() < 2 or data['price_ty'].nunique() < 2:
        return {'giai_doan': stage, 'nhom': group, 'so_tin_n': len(data), 'pearson_r': np.nan, 'p_pearson': np.nan, 'spearman_rho': np.nan, 'p_spearman': np.nan, 'do_doc_trieu_m2': np.nan}
    r, pr = stats.pearsonr(data['area'], data['price_ty'])
    rho, ps = stats.spearmanr(data['area'], data['price_ty'])
    slope = stats.linregress(data['area'], data['price_ty']).slope * 1000
    return {'giai_doan': stage, 'nhom': group, 'so_tin_n': len(data), 'pearson_r': r, 'p_pearson': pr, 'spearman_rho': rho, 'p_spearman': ps, 'do_doc_trieu_m2': slope}

valid = df[(df['area'] > 0) & (df['price_ty'] > 0)].copy()
filtered = valid[valid['area'].between(10, 1000) & valid['ppm_trieu'].between(5, 1000)].copy()
rows = [correlation_row(valid, 'Toàn thị trường', 'Gốc'), correlation_row(filtered, 'Toàn thị trường', 'Lọc cực đoan')]
for kind in sorted(df['property_type_name'].dropna().unique()):
    rows += [correlation_row(valid[valid['property_type_name'] == kind], kind, 'Gốc'), correlation_row(filtered[filtered['property_type_name'] == kind], kind, 'Lọc cực đoan')]
corr = pd.DataFrame(rows)
corr.to_csv(OUT / '04_correlation_full.csv', index=False, encoding='utf-8-sig')

bins = [0, 40, 60, 80, 100, 150, 300, np.inf]
labels = ['<40 m²', '40-60 m²', '60-80 m²', '80-100 m²', '100-150 m²', '150-300 m²', '>300 m²']
valid['nhom_dt'] = pd.cut(valid['area'], bins=bins, labels=labels)
area_summary = valid.groupby('nhom_dt', observed=False).agg(so_tin=('price_ty', 'count'), gia_trung_vi_ty=('price_ty', 'median'), don_gia_trung_vi_trieu=('ppm_trieu', 'median')).reset_index()
area_summary.to_csv(OUT / '05_area_band_summary_full.csv', index=False, encoding='utf-8-sig')

# Chart C: sampled points, regression line computed from filtered full data.
plot_df = filtered.sample(min(25000, len(filtered)), random_state=42)
groups = [('Toàn thị trường', filtered, '#34495e')] + [(x, filtered[filtered['property_type_name'] == x], c) for x, c in zip(sorted(filtered['property_type_name'].dropna().unique()), ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e'])]
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for axis, (name, sub, color) in zip(axes.flat, groups):
    sample = sub.sample(min(3000, len(sub)), random_state=42) if len(sub) else sub
    axis.scatter(sample['area'], sample['price_ty'], s=8, alpha=.25, color=color, rasterized=True)
    if len(sub) >= 3 and sub['area'].nunique() > 1:
        x = np.linspace(sub['area'].min(), sub['area'].max(), 100)
        slope, intercept = np.polyfit(sub['area'].to_numpy(), sub['price_ty'].to_numpy(), 1)
        axis.plot(x, intercept + slope * x, color='black', linewidth=2)
        r = correlation_row(sub, name, 'Lọc cực đoan')
        axis.text(.04, .94, f"n={len(sub):,}\nr={r['pearson_r']:.3f}\nρ={r['spearman_rho']:.3f}", transform=axis.transAxes, va='top', fontsize=9, bbox={'facecolor':'white','alpha':.8})
    axis.set_title(name, fontweight='bold')
    axis.set_xlabel('Diện tích (m²)'); axis.set_ylabel('Giá (tỷ VND)')
axes.flat[-1].axis('off')
fig.suptitle('Tương quan diện tích và giá theo phân khúc (đã lọc giá/m² cực đoan)', fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / '03_area_price_correlation.png', bbox_inches='tight')
plt.close(fig)

# ---------- D. Missing by type, streamed to avoid RAM peak ----------
original_cols = ['name', 'description', 'property_type_name', 'province_name', 'district_name', 'ward_name', 'street_name', 'project_name', 'price', 'area', 'floor_count', 'frontage_width', 'house_depth', 'road_width', 'bedroom_count', 'bathroom_count', 'house_direction', 'balcony_direction', 'published_at']
pf = pq.ParquetFile(PARQ)
missing_sum = None
row_count = {}
for batch in pf.iter_batches(batch_size=100000, columns=original_cols):
    chunk = batch.to_pandas()
    g = chunk.groupby('property_type_name', dropna=False)
    counts = g.size()
    miss = g[original_cols].apply(lambda x: x.isna().sum())
    if missing_sum is None:
        missing_sum = miss
    else:
        missing_sum = missing_sum.add(miss, fill_value=0)
    for key, value in counts.items():
        row_count[key] = row_count.get(key, 0) + int(value)
    del chunk, g, counts, miss
    gc.collect()
missing_pct = missing_sum.div(pd.Series(row_count), axis=0).div(len(original_cols)).mul(100)
missing_pct = missing_pct.sort_index()
overall = pd.DataFrame({'so_tin': pd.Series(row_count), 'ty_le_thieu_19cot_%': missing_pct.mean(axis=1), 'so_cot_thieu_tb': missing_pct.sum(axis=1) / 100}).sort_values('ty_le_thieu_19cot_%', ascending=False)
overall.to_csv(OUT / '06_missing_summary_by_type_full.csv', encoding='utf-8-sig')
missing_pct.to_csv(OUT / '07_missing_matrix_by_type_full.csv', encoding='utf-8-sig')

fig, ax = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 2.3]})
bar = overall.sort_values('ty_le_thieu_19cot_%')
ax[0].barh(bar.index.astype(str), bar['ty_le_thieu_19cot_%'], color='#c0392b')
for y, value in enumerate(bar['ty_le_thieu_19cot_%']): ax[0].text(value, y, f' {value:.1f}%', va='center')
ax[0].set_title('Tỷ lệ thiếu tổng thể theo loại hình', fontweight='bold'); ax[0].set_xlabel('% missing trên 19 cột')
# Heatmap only columns with any missing, retain original order.
heat_cols = [c for c in original_cols if missing_pct[c].sum() > 0]
sns.heatmap(missing_pct[heat_cols], annot=True, fmt='.1f', cmap='YlOrRd', linewidths=.3, ax=ax[1], cbar_kws={'label':'% missing'})
ax[1].set_title('Ma trận missing theo cột và loại hình', fontweight='bold'); ax[1].set_xlabel('Cột'); ax[1].set_ylabel('')
ax[1].tick_params(axis='x', rotation=45)
fig.suptitle('Chất lượng dữ liệu theo phân khúc — toàn bộ dữ liệu sạch', fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / '04_missing_by_type.png', bbox_inches='tight')
plt.close(fig)

# ---------- Excel + report + package ----------
with pd.ExcelWriter(OUT / 'part3_analysis_tables.xlsx', engine='openpyxl') as writer:
    type_summary.to_excel(writer, sheet_name='Loai_hinh', index=False)
    district_summary.to_excel(writer, sheet_name='Quan_huyen', index=False)
    belt_summary.to_excel(writer, sheet_name='Vanh_dai', index=False)
    corr.to_excel(writer, sheet_name='Tuong_quan', index=False)
    area_summary.to_excel(writer, sheet_name='Nhom_dien_tich', index=False)
    overall.to_excel(writer, sheet_name='Missing_tong_hop')
    missing_pct.to_excel(writer, sheet_name='Missing_ma_tran')

report = f'''# Phần tiếp theo — phân tích phân khúc, địa bàn, tương quan và missing

## Phạm vi
- Nguồn: `data_clean_ha_noi.parquet`
- Số tin phân tích: **{N:,}**
- Đơn vị: giá tỷ VND; đơn giá triệu VND/m²; diện tích m².
- Notebook gốc trong `bds 2.zip` chỉ đọc `08_preview_clean_5000.csv` (5.000 dòng, tháng 06/2025). Bộ này chạy lại trên toàn bộ dữ liệu sạch để tránh kết luận từ mẫu xem trước.

## Quy tắc xử lý
- Mô tả phân khúc dùng Median, Q1, Q3; Mean chỉ để đối chiếu.
- Tương quan báo cáo cả mẫu gốc và mẫu lọc cực đoan: diện tích 10–1.000 m², đơn giá 5–1.000 triệu VND/m².
- Không xóa bản ghi vì ngoại lai; chỉ lọc khi tính bảng tương quan nhạy cảm với điểm cực đoan.
- Missing tính trên 19 cột gốc, theo từng loại hình, bằng xử lý theo batch.

## File chính
- `part3_analysis_tables.xlsx`: 7 bảng tổng hợp.
- `01_type_summary_full.csv`: loại hình.
- `02_district_summary_full.csv`: quận/huyện.
- `03_belt_summary_full.csv`: 3 vành đai.
- `04_correlation_full.csv`: Pearson, Spearman, độ dốc.
- `05_area_band_summary_full.csv`: nhóm diện tích.
- `06_missing_summary_by_type_full.csv`, `07_missing_matrix_by_type_full.csv`: missing.
- `figures/`: 4 biểu đồ PNG.
'''
(OUT / 'report_part3.md').write_text(report, encoding='utf-8')

zip_path = OUT / 'part3_analysis_files.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.glob('*.csv')) + sorted(OUT.glob('*.xlsx')) + sorted(OUT.glob('*.md')) + sorted(FIG.glob('*.png')):
        z.write(p, p.relative_to(OUT))

print(json.dumps({'rows': N, 'filtered_correlation_rows': len(filtered), 'zip': str(zip_path), 'zip_bytes': zip_path.stat().st_size}, ensure_ascii=False))
print(type_summary[['property_type_name','so_tin','ty_le_%','gia_trung_vi_ty','dien_tich_trung_vi_m2','don_gia_trung_vi_trieu']].to_string(index=False))
print(belt_summary.to_string(index=False))
