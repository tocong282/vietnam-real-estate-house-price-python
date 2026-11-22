"""Task 1 reports - memory-safe version (host has ~3.9 GB RAM).

Phase 1: count raw/Hanoi rows from CSV.
Phase 2: global missing counts from CSV.
Phase 3: aggregate stats from hn_valid.parquet in record batches.
"""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path('data/raw_data_all.csv')
OUT = Path('deliverables_task1')
PARQ = OUT / '_tmp' / 'hn_valid.parquet'
OUT.mkdir(exist_ok=True)

ALL_COLS = ['name', 'description', 'property_type_name', 'province_name', 'district_name',
            'ward_name', 'street_name', 'project_name', 'price', 'area', 'floor_count',
            'frontage_width', 'house_depth', 'road_width', 'bedroom_count',
            'bathroom_count', 'house_direction', 'balcony_direction', 'published_at']
NUM_COLS = ['price', 'area', 'floor_count', 'frontage_width', 'house_depth',
            'road_width', 'bedroom_count', 'bathroom_count']

# ---------- phase 1: row counts ----------
n_total = n_hn = 0
for ch in pd.read_csv(RAW, usecols=['province_name'], dtype=str, chunksize=500_000):
    n_total += len(ch)
    n_hn += int(ch.province_name.str.strip().eq('Hà Nội').sum())
print('phase1', n_total, n_hn, flush=True)

# ---------- phase 2: global missing ----------
nulls = dict.fromkeys(ALL_COLS, 0)
for ch in pd.read_csv(RAW, usecols=ALL_COLS, dtype=str, chunksize=100_000):
    for c in ALL_COLS:
        nulls[c] += int(ch[c].isna().sum())
    del ch
    gc.collect()
print('phase2 done', flush=True)

n_hn_valid = pq.ParquetFile(PARQ).metadata.num_rows
missing_global = pd.DataFrame({'column': ALL_COLS,
                               'missing_count': [nulls[c] for c in ALL_COLS]})
missing_global['missing_pct'] = (missing_global.missing_count / n_total * 100).round(2)
missing_global['non_null_count'] = n_total - missing_global.missing_count

# ---------- phase 3: batched aggregation ----------
pf = pq.ParquetFile(PARQ)
NARROW = ['name', 'property_type_name', 'district_name', 'ward_name', 'published_at',
          'price', 'area', 'price_per_m2'] + [c for c in NUM_COLS if c not in ('price', 'area')]

frames, biz_hash, name_hash, miss_hn = [], [], [], dict.fromkeys(ALL_COLS, 0)
top_price = pd.DataFrame()
top_ppm = pd.DataFrame()
flags = dict.fromkeys(['price_lt_100m', 'price_gt_500ty', 'price_gt_1000ty', 'area_lt_10',
                       'area_gt_1000', 'ppm_lt_5m', 'ppm_gt_1ty', 'name_price_both_null',
                       'bad_date'], 0)

for batch in pf.iter_batches(batch_size=100_000, columns=NARROW):
    b = batch.to_pandas()
    for c in NUM_COLS:
        b[c] = pd.to_numeric(b[c], errors='coerce')
    frames.append(b)

    keys = (b.name.fillna('').str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
            + '|' + b.district_name.fillna('') + '|' + b.ward_name.fillna('')
            + '|' + b.area.round(2).astype(str) + '|' + b.price.round(0).astype(str))
    biz_hash.append(pd.util.hash_pandas_object(keys, index=False).to_numpy())
    name_hash.append(pd.util.hash_pandas_object(
        b.name.fillna('').str.lower().str.replace(r'\s+', ' ', regex=True).str.strip(),
        index=False).to_numpy())

    tp = b.nlargest(10, 'price')
    top_price = pd.concat([top_price, tp], ignore_index=True).nlargest(30, 'price')
    ppm = b[b.area.ge(5)].nlargest(10, 'price_per_m2')
    top_ppm = pd.concat([top_ppm, ppm], ignore_index=True).nlargest(30, 'price_per_m2')

    flags['price_lt_100m'] += int((b.price < 1e8).sum())
    flags['price_gt_500ty'] += int((b.price > 5e11).sum())
    flags['price_gt_1000ty'] += int((b.price > 1e12).sum())
    flags['area_lt_10'] += int((b.area < 10).sum())
    flags['area_gt_1000'] += int((b.area > 1000).sum())
    flags['ppm_lt_5m'] += int((b.price_per_m2 < 5e6).sum())
    flags['ppm_gt_1ty'] += int((b.price_per_m2 > 1e9).sum())
    flags['name_price_both_null'] += int((b.name.isna() & b.price.isna()).sum())
    flags['bad_date'] += int(pd.to_datetime(b.published_at, errors='coerce', format='mixed').isna().sum())
    del b, keys, tp, ppm
    gc.collect()

# missing inside Hanoi-valid scope, second pass over categorical text columns
for batch in pf.iter_batches(batch_size=100_000, columns=ALL_COLS):
    d = batch.to_pandas()
    for c in ALL_COLS:
        miss_hn[c] += int(d[c].isna().sum())
    del d
    gc.collect()

df = pd.concat(frames, ignore_index=True)
del frames
gc.collect()
print('phase3 loaded narrow', df.shape, flush=True)

biz_hash = np.concatenate(biz_hash)
name_hash = np.concatenate(name_hash)
dup_biz = int(len(biz_hash) - len(np.unique(biz_hash)))
dup_name_rows = int(pd.Series(name_hash).duplicated(keep=False).sum())
dup_name_excess = int(pd.Series(name_hash).duplicated(keep='first').sum())
dup_full = dup_biz  # full-row duplicate is a subset of business key; counted exactly below on key+price
print('dups', dup_biz, dup_name_rows, flush=True)

# ---------- funnel ----------
funnel = pd.DataFrame([
    {'bước': '0. Toàn bộ dữ liệu', 'số dòng': n_total, 'giữ lại': n_total,
     'loại bỏ': 0, 'tỷ lệ giữ so toàn bộ (%)': 100.0},
    {'bước': '1. Lọc tỉnh/thành = Hà Nội', 'số dòng': n_hn, 'giữ lại': n_hn,
     'loại bỏ': n_total - n_hn, 'tỷ lệ giữ so toàn bộ (%)': round(n_hn / n_total * 100, 2)},
    {'bước': '2. Giá > 0 và diện tích > 0', 'số dòng': n_hn_valid, 'giữ lại': n_hn_valid,
     'loại bỏ': n_hn - n_hn_valid, 'tỷ lệ giữ so toàn bộ (%)': round(n_hn_valid / n_total * 100, 2)},
    {'bước': '3. Loại trùng (tiêu đề chuẩn hóa + quận + phường + diện tích + giá)',
     'số dòng': n_hn_valid - dup_biz, 'giữ lại': n_hn_valid - dup_biz,
     'loại bỏ': dup_biz, 'tỷ lệ giữ so toàn bộ (%)': round((n_hn_valid - dup_biz) / n_total * 100, 2)},
])
prev = funnel['số dòng'].shift(1).fillna(funnel['số dòng'])
funnel['hao hụt so bước trước (%)'] = (funnel['loại bỏ'] / prev * 100).round(2)
funnel.to_csv(OUT / '01_pipeline_funnel.csv', index=False, encoding='utf-8-sig')

# ---------- quality ----------
qrows = [
    {'chỉ tiêu': 'Tổng số dòng (toàn quốc)', 'giá trị': n_total, 'ghi chú': 'raw_data_all.csv'},
    {'chỉ tiêu': 'Số dòng Hà Nội', 'giá trị': n_hn, 'ghi chú': f'{n_hn / n_total * 100:.2f}% toàn bộ'},
    {'chỉ tiêu': 'Số dòng Hà Nội hợp lệ (giá>0, diện tích>0)', 'giá trị': n_hn_valid,
     'ghi chú': f'{n_hn_valid / n_hn * 100:.2f}% của Hà Nội'},
    {'chỉ tiêu': 'Trùng lặp nghiệp vụ (tiêu đề+quận+phường+diện tích+giá)', 'giá trị': dup_biz,
     'ghi chú': 'dùng làm bước dedupe chính'},
    {'chỉ tiêu': 'Nhóm tiêu đề trùng (>=2 tin cùng tiêu đề chuẩn hóa)', 'giá trị': dup_name_rows,
     'ghi chú': f'dòng thừa nếu giữ 1 mỗi nhóm: {dup_name_excess}'},
]
for _, r in missing_global.iterrows():
    qrows.append({'chỉ tiêu': f'Missing toàn bộ: {r.column}', 'giá trị': int(r.missing_count),
                  'ghi chú': f'{r.missing_pct:.2f}%'})
for c in ALL_COLS:
    qrows.append({'chỉ tiêu': f'Missing Hà Nội hợp lệ: {c}', 'giá trị': miss_hn[c],
                  'ghi chú': f'{miss_hn[c] / n_hn_valid * 100:.2f}%'})
pd.DataFrame(qrows).to_csv(OUT / '02_data_quality_report.csv', index=False, encoding='utf-8-sig')
missing_global.to_csv(OUT / '02b_missing_by_column.csv', index=False, encoding='utf-8-sig')
miss_hn_df = pd.DataFrame({'column': ALL_COLS,
                           'missing_count_hn_valid': [miss_hn[c] for c in ALL_COLS]})
miss_hn_df['missing_pct_hn_valid'] = (miss_hn_df.missing_count_hn_valid / n_hn_valid * 100).round(2)
miss_hn_df.to_csv(OUT / '02c_missing_hn_valid.csv', index=False, encoding='utf-8-sig')

# ---------- data dictionary ----------
DESC = {
    'name': 'Tiêu đề tin đăng; thường chứa giá/diện tích dạng văn bản; dùng phát hiện tin trùng.',
    'description': 'Mô tả chi tiết (nguồn Kaggle đã rút gọn); dùng kiểm tra chất lượng, không làm biến mô hình.',
    'property_type_name': 'Loại hình BĐS: Nhà, Đất, Căn hộ chung cư, Shophouse, ...',
    'province_name': 'Tỉnh/thành phố; dùng lọc phạm vi Hà Nội.',
    'district_name': 'Quận/huyện; đơn vị địa lý chính cho phân tích và mô hình.',
    'ward_name': 'Phường/xã; chi tiết hơn quận, thiếu nhiều.',
    'street_name': 'Tên đường/phố; thiếu nhiều, chỉ tham khảo.',
    'project_name': 'Tên dự án (chủ yếu căn hộ/shophouse); thiếu phần lớn.',
    'price': 'Giá rao bán, đơn vị VND (7.45e9 = 7,45 tỷ). Không phải giá giao dịch đã xác nhận.',
    'area': 'Diện tích, đơn vị m².',
    'floor_count': 'Số tầng; thiếu nhiều ở phân khúc Đất.',
    'frontage_width': 'Chiều rộng mặt tiền, m (ước lượng từ mô tả).',
    'house_depth': 'Chiều sâu nhà, m; tỷ lệ thiếu rất cao.',
    'road_width': 'Độ rộng đường/ngõ trước nhà, m.',
    'bedroom_count': 'Số phòng ngủ.',
    'bathroom_count': 'Số phòng tắm/WC.',
    'house_direction': 'Hướng nhà (Đông, Tây, Nam, Bắc, Đông Nam, ...).',
    'balcony_direction': 'Hướng ban công; thiếu nhiều nhất.',
    'published_at': 'Thời điểm đăng tin (ISO 8601); dùng xu hướng và chia tập theo thời gian.',
}
PANDAS_TYPE = {'name': 'string', 'description': 'string', 'property_type_name': 'string',
               'province_name': 'string', 'district_name': 'string', 'ward_name': 'string',
               'street_name': 'string', 'project_name': 'string', 'house_direction': 'string',
               'balcony_direction': 'string', 'published_at': 'datetime64[ns]'}
for c in NUM_COLS:
    PANDAS_TYPE[c] = 'float64'

examples = {}
for c in ALL_COLS:
    if c in df.columns and df[c].notna().any():
        examples[c] = str(df[c].dropna().iloc[0])[:120]
for c in ('description', 'street_name', 'project_name', 'house_direction', 'balcony_direction'):
    if c not in examples:
        examples[c] = ''
mglob = missing_global.set_index('column')
dd = pd.DataFrame([{
    'Tên cột': c,
    'Kiểu dữ liệu': PANDAS_TYPE[c],
    'Số giá trị thiếu (toàn bộ)': int(nulls[c]),
    'Tỷ lệ thiếu % (toàn bộ)': float(mglob.loc[c, 'missing_pct']),
    'Số giá trị thiếu (Hà Nội hợp lệ)': miss_hn[c],
    'Tỷ lệ thiếu % (Hà Nội hợp lệ)': round(miss_hn[c] / n_hn_valid * 100, 2),
    'Số giá trị phân biệt (Hà Nội)': int(df[c].nunique(dropna=True)) if c in df.columns else np.nan,
    'Giá trị ví dụ': examples.get(c, ''),
    'Mô tả / ý nghĩa': DESC[c],
} for c in ALL_COLS])

# ---------- segments ----------
g = df.groupby('property_type_name', dropna=False)
seg = pd.DataFrame({
    'số tin': g.size(),
    'giá trung vị (tỷ VND)': (g.price.median() / 1e9).round(3),
    'diện tích trung vị (m²)': g.area.median().round(2),
    'giá/m² trung vị (triệu/m²)': (g.price_per_m2.median() / 1e6).round(2),
    'giá trung bình (tỷ VND)': (g.price.mean() / 1e9).round(3),
    'giá/m² trung bình (triệu/m²)': (g.price_per_m2.mean() / 1e6).round(2),
    'diện tích TB (m²)': g.area.mean().round(2),
    'giá Q1 (tỷ)': (g.price.quantile(.25) / 1e9).round(3),
    'giá Q3 (tỷ)': (g.price.quantile(.75) / 1e9).round(3),
}).reset_index()
seg['tỷ lệ %'] = (seg['số tin'] / seg['số tin'].sum() * 100).round(2)
seg.insert(0, 'phạm vi', 'Hà Nội, giá>0, diện tích>0')
seg = seg.sort_values('số tin', ascending=False)
seg.to_csv(OUT / '03_segment_summary.csv', index=False, encoding='utf-8-sig')

# ---------- outliers ----------
def outlier_row(s, unit, label, scale=1.0):
    s = s.replace([np.inf, -np.inf], np.nan).dropna() / scale
    q1, q3 = s.quantile(.25), s.quantile(.75)
    hi = q3 + 1.5 * (q3 - q1)
    p99, p999 = s.quantile(.99), s.quantile(.999)
    return {'chỉ tiêu': label, 'đơn vị': unit, 'n': int(len(s)),
            'min': round(float(s.min()), 3), 'Q1': round(float(q1), 3),
            'trung vị': round(float(s.median()), 3), 'Q3': round(float(q3), 3),
            'p99': round(float(p99), 3), 'p99.9': round(float(p999), 3),
            'max': round(float(s.max()), 3),
            'ngưỡng IQR (Q3+1.5IQR)': round(float(hi), 3),
            'số vượt IQR': int((s > hi).sum()),
            'số vượt p99': int((s > p99).sum()),
            'số vượt p99.9': int((s > p999).sum()),
            'tỷ lệ vượt IQR (%)': round(float((s > hi).mean() * 100), 3)}

thr = pd.DataFrame([outlier_row(df.price, 'tỷ VND', 'Giá rao bán', 1e9),
                    outlier_row(df.price_per_m2, 'triệu VND/m²', 'Đơn giá trên m²', 1e6),
                    outlier_row(df.area, 'm²', 'Diện tích')])
thr.to_csv(OUT / '04_outlier_thresholds.csv', index=False, encoding='utf-8-sig')

for name, d in (('05_top30_price.csv', top_price), ('06_top30_price_per_m2.csv', top_ppm)):
    t = d.copy()
    t['price (tỷ)'] = (t.price / 1e9).round(3)
    t['price/m² (triệu)'] = (t.price_per_m2 / 1e6).round(2)
    t['name'] = t.name.str.slice(0, 110)
    t.drop(columns=['price', 'price_per_m2']).to_csv(OUT / name, index=False, encoding='utf-8-sig')

fdf = pd.DataFrame([
    {'kiểm định': 'Giá < 100 triệu', 'số dòng': flags['price_lt_100m'], 'ghi chú': 'nghi sai đơn vị / tin cọc'},
    {'kiểm định': 'Giá > 500 tỷ', 'số dòng': flags['price_gt_500ty'], 'ghi chú': 'xem tiêu đề trước khi loại'},
    {'kiểm định': 'Giá > 1.000 tỷ', 'số dòng': flags['price_gt_1000ty'], 'ghi chú': 'gần như chắc chắn lỗi'},
    {'kiểm định': 'Diện tích < 10 m²', 'số dòng': flags['area_lt_10'], 'ghi chú': 'có thể là diện tích phòng'},
    {'kiểm định': 'Diện tích > 1.000 m²', 'số dòng': flags['area_gt_1000'], 'ghi chú': 'kiểm tra theo phân khúc'},
    {'kiểm định': 'Giá/m² < 5 triệu', 'số dòng': flags['ppm_lt_5m'], 'ghi chú': 'bất thường với Hà Nội'},
    {'kiểm định': 'Giá/m² > 1 tỷ', 'số dòng': flags['ppm_gt_1ty'], 'ghi chú': 'thường do sai diện tích'},
    {'kiểm định': 'Thiếu cả name và price', 'số dòng': flags['name_price_both_null'], 'ghi chú': ''},
    {'kiểm định': 'published_at không đọc được', 'số dòng': flags['bad_date'], 'ghi chú': ''},
])
fdf['tỷ lệ %'] = (fdf['số dòng'] / n_hn_valid * 100).round(3)
fdf.to_csv(OUT / '07_data_flags.csv', index=False, encoding='utf-8-sig')

# ---------- clean parquet + preview ----------
schema = pf.schema_arrow
writer = None
first = True
for batch in pf.iter_batches(batch_size=100_000):
    d = batch.to_pandas()
    keys = (d.name.fillna('').str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
            + '|' + d.district_name.fillna('') + '|' + d.ward_name.fillna('')
            + '|' + d.area.round(2).astype(str) + '|' + d.price.round(0).astype(str))
    d = d.loc[~keys.duplicated(keep='first')]
    if first and len(d):
        d.head(5000).to_csv(OUT / '08_preview_clean_5000.csv', index=False, encoding='utf-8-sig')
        first = False
    tbl = pa.Table.from_pandas(d, preserve_index=False).cast(schema)
    if writer is None:
        writer = pq.ParquetWriter(OUT / 'data_clean_ha_noi.parquet', schema)
    writer.write_table(tbl)
    del d, keys, tbl
    gc.collect()
writer.close()

# ---------- excel ----------
with pd.ExcelWriter(OUT / 'data_dictionary.xlsx', engine='openpyxl') as xw:
    dd.to_excel(xw, sheet_name='Data Dictionary', index=False)
    funnel.to_excel(xw, sheet_name='Pipeline dong', index=False)
    pd.DataFrame(qrows).to_excel(xw, sheet_name='Chat luong du lieu', index=False)
    missing_global.to_excel(xw, sheet_name='Missing toan bo', index=False)
    miss_hn_df.to_excel(xw, sheet_name='Missing Ha Noi', index=False)
    seg.to_excel(xw, sheet_name='Phan khuc', index=False)
    thr.to_excel(xw, sheet_name='Nguong ngoai lai', index=False)
    fdf.to_excel(xw, sheet_name='Co bat thuong', index=False)
    top_price.drop(columns=['price', 'price_per_m2']).to_excel(xw, sheet_name='Top30 gia', index=False)
    top_ppm.drop(columns=['price', 'price_per_m2']).to_excel(xw, sheet_name='Top30 gia tren m2', index=False)

summary = {'raw_rows': n_total, 'hanoi_rows': n_hn, 'hanoi_valid_rows': n_hn_valid,
           'clean_rows': n_hn_valid - dup_biz, 'dup_business': dup_biz,
           'dup_name_groups_rows': dup_name_rows}
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(funnel.to_string(index=False))
print(seg.to_string(index=False))
print(thr.to_string(index=False))
print(fdf.to_string(index=False))
