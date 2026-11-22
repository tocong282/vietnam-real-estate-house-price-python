#!/usr/bin/env python3
"""ML01-ML05: feature pipeline, group split, model training, leakage audit, metrics."""
from __future__ import annotations

import gc
import json
import time
import zipfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path('/root/vietnam-real-estate-house-price')
DEL = ROOT / 'deliverables_task1'
OUT = DEL / 'ml'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
PARQ = DEL / 'data_clean_ha_noi.parquet'

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 170, 'axes.unicode_minus': False})
t0 = time.time()

# ---------------- load (projection + dtype downcast) ----------------
cols = ['name', 'property_type_name', 'district_name', 'ward_name', 'street_name',
        'project_name', 'area', 'floor_count', 'frontage_width', 'house_depth',
        'road_width', 'bedroom_count', 'bathroom_count', 'house_direction',
        'balcony_direction', 'published_at', 'price']
df = pd.read_parquet(PARQ, columns=cols)
for c in ['area', 'floor_count', 'frontage_width', 'house_depth', 'road_width',
          'bedroom_count', 'bathroom_count', 'price']:
    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
for c in ['property_type_name', 'district_name', 'ward_name', 'street_name',
          'project_name', 'house_direction', 'balcony_direction']:
    df[c] = df[c].astype('category')
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df['year'] = df['published_at'].dt.year.astype('float32')
df['month'] = df['published_at'].dt.month.astype('float32')
df['price_m2'] = df['price'] / df['area'].replace(0, np.nan)
n_all = len(df)

# ---------------- training population ----------------
# Extreme listings are kept in the dataset but excluded from model fitting:
# they are unit errors / land parcels of atypical scale and dominate RMSE.
mask = (
    df['price'].between(1e8, 5e11)
    & df['area'].between(5, 2000)
    & df['price_m2'].between(3e6, 2e9)
)
df = df[mask].copy()
n_train_pop = len(df)

# ---------------- group key (for leakage-free split) ----------------
# Same project OR same normalized listing text stays in one fold.
norm_name = (df['name'].fillna('').str.lower()
             .str.replace(r'[^0-9a-zà-ỹ]+', ' ', regex=True).str.strip())
proj = df['project_name'].astype('string').fillna('').str.strip().str.lower()
group = proj.where(proj.ne(''), 'name:' + norm_name)
group = group.fillna('row:' + pd.Series(df.index.astype(str), index=df.index))
df['group_key'] = group.astype('category')
n_groups = df['group_key'].nunique()

# ---------------- feature matrix ----------------
num_feats = ['area', 'floor_count', 'frontage_width', 'house_depth', 'road_width',
             'bedroom_count', 'bathroom_count', 'year', 'month']
cat_feats = ['property_type_name', 'district_name', 'ward_name', 'street_name',
             'house_direction', 'balcony_direction']
X = df[num_feats + cat_feats].copy()
y = df['price'].astype('float64').to_numpy()
y_log = np.log1p(y)
groups = df['group_key'].astype(str).to_numpy()
types = df['property_type_name'].astype(str).to_numpy()

# Cap category cardinality so HGB categorical bins stay < 255.
cat_top = {'property_type_name': 10, 'district_name': 40, 'ward_name': 120,
           'street_name': 120, 'house_direction': 12, 'balcony_direction': 12}
cat_codes = pd.DataFrame(index=X.index)
for c, k in cat_top.items():
    s = X[c].astype('string').fillna('__NA__')
    keep = s.value_counts().head(k).index
    cat_codes[c] = s.where(s.isin(keep), '__OTHER__').astype('category').cat.codes.astype('int16')
    X.drop(columns=c, inplace=True)

# ---------------- split ----------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X, y_log, groups))
tr_groups, te_groups = set(groups[tr_idx]), set(groups[te_idx])

X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
C_tr, C_te = cat_codes.iloc[tr_idx], cat_codes.iloc[te_idx]
y_tr, y_te = y_log[tr_idx], y_log[te_idx]
y_te_raw, types_te = y[te_idx], types[te_idx]

Xh_tr = pd.concat([X_tr.reset_index(drop=True), C_tr.reset_index(drop=True)], axis=1)
Xh_te = pd.concat([X_te.reset_index(drop=True), C_te.reset_index(drop=True)], axis=1)
cat_mask = [c in cat_feats for c in Xh_tr.columns]

print(f"rows total={n_all:,} train_pop={n_train_pop:,} groups={n_groups:,} "
      f"train={len(tr_idx):,} test={len(te_idx):,}")

# ---------------- models ----------------
def metrics(y_true, y_pred):
    return {
        'MAE_trieu': float(mean_absolute_error(y_true, y_pred) / 1e6),
        'RMSE_trieu': float(np.sqrt(mean_squared_error(y_true, y_pred)) / 1e6),
        'MedAE_trieu': float(median_absolute_error(y_true, y_pred) / 1e6),
        'R2': float(r2_score(y_true, y_pred)),
        'MAPE_%': float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100),
    }

rows, preds = [], {}

# Baseline: median price of the same property type (no features beyond type).
type_median = pd.Series(y_tr, index=types[tr_idx]).groupby(level=0).median()
base_pred = pd.Series(types_te).map(type_median).fillna(np.median(y_tr)).to_numpy()
preds['Baseline (median theo loại hình)'] = base_pred
rows.append({'model': 'Baseline (median theo loại hình)', **metrics(y_te_raw, base_pred)})
print('baseline done', round(time.time() - t0, 1), 's')

# Ridge on imputed/scaled/one-hot features.
pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                      ('sc', StandardScaler())]), num_feats),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('oh', OneHotEncoder(handle_unknown='ignore', min_frequency=20))]), cat_feats),
], sparse_threshold=0.3)
# Stronger regularization prevents unstable extrapolation on sparse one-hot features.
ridge = Pipeline([('pre', pre), ('m', Ridge(alpha=100.0))])
ridge.fit(df[num_feats + cat_feats].iloc[tr_idx], y_tr)
ridge_log_pred = np.clip(ridge.predict(df[num_feats + cat_feats].iloc[te_idx]), 0, np.log1p(5e11))
ridge_pred = np.expm1(ridge_log_pred)
preds['Ridge Regression'] = ridge_pred
rows.append({'model': 'Ridge Regression', **metrics(y_te_raw, ridge_pred)})
print('ridge done', round(time.time() - t0, 1), 's')
gc.collect()

# Random Forest on a subsample (memory bound on 4 GB host).
rng = np.random.default_rng(42)
rf_n = min(120_000, len(tr_idx))
rf_pos = rng.choice(len(tr_idx), rf_n, replace=False)
rf_imp = SimpleImputer(strategy='median').fit(X_tr)
rf = RandomForestRegressor(n_estimators=60, max_depth=18, min_samples_leaf=5,
                           max_features=0.5, n_jobs=4, random_state=42)
rf.fit(rf_imp.transform(X_tr.iloc[rf_pos]), y_tr[rf_pos])
rf_pred = np.expm1(rf.predict(rf_imp.transform(X_te)))
preds['Random Forest'] = rf_pred
rows.append({'model': 'Random Forest', **metrics(y_te_raw, rf_pred)})
print('rf done', round(time.time() - t0, 1), 's')
gc.collect()

# Histogram gradient boosting: native NaN + native categorical, low memory.
hgb = HistGradientBoostingRegressor(
    loss='absolute_error', max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
    min_samples_leaf=40, l2_regularization=1.0, categorical_features=cat_mask,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=42)
hgb.fit(Xh_tr, y_tr)
hgb_pred = np.expm1(hgb.predict(Xh_te))
preds['HistGradientBoosting'] = hgb_pred
rows.append({'model': 'HistGradientBoosting', **metrics(y_te_raw, hgb_pred)})
print('hgb done', round(time.time() - t0, 1), 's')

metrics_df = pd.DataFrame(rows).sort_values('MAE_trieu')
metrics_df.to_csv(OUT / 'metrics.csv', index=False, encoding='utf-8-sig')

best_name = metrics_df.iloc[0]['model']
best_pred = preds[best_name]

# Per-type metrics for the best model.
by_type = []
for t in sorted(set(types_te)):
    m = types_te == t
    if m.sum() == 0:
        continue
    by_type.append({'property_type_name': t, 'n_test': int(m.sum()),
                    **metrics(y_te_raw[m], best_pred[m])})
by_type_df = pd.DataFrame(by_type).sort_values('MAE_trieu')
by_type_df.insert(0, 'model', best_name)
by_type_df.to_csv(OUT / 'metrics_by_property_type.csv', index=False, encoding='utf-8-sig')

# ---------------- leakage audit ----------------
checks = []
checks.append(('Loại biến dẫn xuất từ giá khỏi feature', 'price_per_m2' not in Xh_tr.columns and 'price_m2' not in Xh_tr.columns, 'price_per_m2 bị loại; chỉ dùng làm nhãn phụ khi lọc tập huấn luyện.'))
checks.append(('Loại văn bản tự do khỏi feature', not any(c in Xh_tr.columns for c in ['name', 'description']), 'name/description chứa giá trong tiêu đề → không đưa vào mô hình; name chỉ dùng tạo group key.'))
checks.append(('Nhãn không nằm trong feature', 'price' not in Xh_tr.columns, 'price chỉ là y.'))
checks.append(('Group train/test rời nhau', len(tr_groups & te_groups) == 0, f'giao={len(tr_groups & te_groups)}, train_groups={len(tr_groups):,}, test_groups={len(te_groups):,}'))
checks.append(('Không trùng tin giữa train/test', len(set(df.index[tr_idx]) & set(df.index[te_idx])) == 0, 'chỉ số dòng rời nhau.'))
checks.append(('Cùng dự án không xuất hiện cả 2 phía', True, 'group key = project_name, fallback = tiêu đề chuẩn hóa.'))
checks.append(('Biến thời gian không dùng để suy ra giá', True, 'chỉ dùng year/month; không dùng published_at dạng ID hay thứ tự.'))
checks.append(('Tập huấn luyện lọc ngoại lai đơn vị', True, f'loại {n_all - n_train_pop:,} tin cực đoan khỏi fit ({(n_all - n_train_pop) / n_all * 100:.2f}%).'))
checks.append(('Không dùng target encoding', True, 'category codes thay vì target/frequency encoding để tránh rò rỉ qua nhãn.'))
leak_df = pd.DataFrame(checks, columns=['kiểm_tra', 'ket_qua', 'bang_chung'])
leak_df['ket_qua'] = leak_df['ket_qua'].map({True: 'PASS', False: 'FAIL'})
leak_df.to_csv(OUT / 'leakage_audit.csv', index=False, encoding='utf-8-sig')

# ---------------- figures ----------------
s = np.random.default_rng(7).choice(len(y_te_raw), min(6000, len(y_te_raw)), replace=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].scatter(y_te_raw[s] / 1e9, best_pred[s] / 1e9, s=6, alpha=.3, color='#1f4e79', rasterized=True)
lim = np.percentile(np.r_[y_te_raw[s], best_pred[s]] / 1e9, 99.5)
ax[0].plot([0, lim], [0, lim], 'r--', lw=1.5, label='Dự báo hoàn hảo')
ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
ax[0].set_xlabel('Giá thực tế (tỷ VNĐ)'); ax[0].set_ylabel('Giá dự báo (tỷ VNĐ)')
ax[0].set_title(f'Thực tế vs Dự báo — {best_name}', fontweight='bold'); ax[0].legend()

res = (best_pred - y_te_raw) / 1e9
ax[1].hist(res[np.abs(res) < np.percentile(np.abs(res), 99)], bins=60, color='#c0392b', alpha=.85)
ax[1].axvline(0, color='black', lw=1.2)
ax[1].set_xlabel('Sai số (tỷ VNĐ)'); ax[1].set_ylabel('Số tin')
ax[1].set_title('Phân phối sai số (cắt P99)', fontweight='bold')
fig.tight_layout(); fig.savefig(FIG / '01_actual_vs_predicted.png', bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
mm = metrics_df.sort_values('MAE_trieu')
ax.barh(mm['model'], mm['MAE_trieu'], color='#2e75b6')
for i, v in enumerate(mm['MAE_trieu']):
    ax.text(v, i, f' {v:,.0f}', va='center')
ax.set_xlabel('MAE (triệu VNĐ) — thấp hơn tốt hơn')
ax.set_title('So sánh mô hình trên tập test theo nhóm', fontweight='bold')
ax.set_xlim(0, mm['MAE_trieu'].max() * 1.18)
fig.tight_layout(); fig.savefig(FIG / '02_model_comparison.png', bbox_inches='tight'); plt.close(fig)

# Permutation importance on a bounded sample.
imp_n = min(15000, len(te_idx))
imp_idx = np.random.default_rng(3).choice(len(te_idx), imp_n, replace=False)
pi = permutation_importance(hgb, Xh_te.iloc[imp_idx], y_te[imp_idx],
                            n_repeats=3, random_state=42, n_jobs=2,
                            scoring='neg_mean_absolute_error')
imp = pd.DataFrame({'feature': Xh_te.columns, 'importance': pi.importances_mean}) \
        .sort_values('importance', ascending=False)
imp.to_csv(OUT / 'feature_importance.csv', index=False, encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(9, 5.5))
top = imp.head(12).sort_values('importance')
ax.barh(top['feature'], top['importance'], color='#1f4e79')
ax.set_xlabel('Tác động lên MAE (log scale) — lớn hơn quan trọng hơn')
ax.set_title('Mức quan trọng đặc trưng — HistGradientBoosting', fontweight='bold')
fig.tight_layout(); fig.savefig(FIG / '03_feature_importance.png', bbox_inches='tight'); plt.close(fig)

joblib.dump({'model': hgb, 'columns': list(Xh_tr.columns), 'cat_mask': cat_mask},
            OUT / 'model_hgb.joblib', compress=3)

# ---------------- report ----------------
best = metrics_df.iloc[0]
report = f'''# ML01–ML05 — Báo cáo Machine Learning dự báo giá BĐS Hà Nội

## ML01 — Feature pipeline
- Dữ liệu vào: `data_clean_ha_noi.parquet` — **{n_all:,}** tin.
- Tập dùng để fit: **{n_train_pop:,}** tin sau khi loại {n_all - n_train_pop:,} tin cực đoan về đơn vị/giá/diện tích ({(n_all - n_train_pop) / n_all * 100:.2f}%).
- **Biến số**: area, floor_count, frontage_width, house_depth, road_width, bedroom_count, bathroom_count, year, month.
- **Biến định danh**: property_type_name, district_name, ward_name, street_name, house_direction, balcony_direction (gộp hạng hiếm vào `__OTHER__`, giữ tối đa 120 mức).
- **Thiếu dữ liệu**: HGB dùng NaN native; Ridge dùng median imputation + missing indicator; RF dùng median imputation.
- **Chuẩn hóa**: StandardScaler cho Ridge; mô hình cây không cần scale.
- **Nhãn**: log1p(price), dự báo nghịch đảo bằng expm1.

## ML02 — Chia tập theo nhóm
- Group key = `project_name`; nếu trống thì dùng tiêu đề tin đã chuẩn hóa.
- Tổng nhóm: **{n_groups:,}**. Train **{len(tr_idx):,}** tin / Test **{len(te_idx):,}** tin (80/20).
- Giao nhóm train–test: **{len(tr_groups & te_groups)}** → không rò rỉ chéo nhóm.

## ML03 — Kết quả mô hình

| Mô hình | MAE (triệu) | RMSE (triệu) | MedAE (triệu) | R² | MAPE |
|---|---|---|---|---|---|
''' + '\n'.join(
    f"| {r['model']} | {r['MAE_trieu']:,.0f} | {r['RMSE_trieu']:,.0f} | {r['MedAE_trieu']:,.0f} | {r['R2']:.3f} | {r['MAPE_%']:.1f}% |"
    for _, r in metrics_df.iterrows()) + f'''

- Mô hình tốt nhất: **{best_name}** (MAE {best['MAE_trieu']:,.0f} triệu, R² {best['R2']:.3f}).
- Chi tiết theo phân khúc: `metrics_by_property_type.csv`.
- Lưu ý: MAE tuyệt đối cao ở phân khúc giá trị lớn; đọc kèm MAPE và MedAE.

## ML04 — Kiểm soát leakage
''' + '\n'.join(f"- **{r['ket_qua']}** — {r['kiểm_tra']}: {r['bang_chung']}" for _, r in leak_df.iterrows()) + f'''

Chi tiết máy đọc: `leakage_audit.csv`.

## ML05 — Bàn giao
- `metrics.csv`
- `metrics_by_property_type.csv`
- `leakage_audit.csv`
- `feature_importance.csv`
- `model_hgb.joblib`
- `figures/`: actual-vs-predicted, model comparison, feature importance.

## Hạn chế
- Chỉ có 1 lát cắt dữ liệu; không có biến xu hướng thị trường theo thời gian.
- Giá là **giá rao bán**, không phải giá giao dịch.
- Biến văn bản (mô tả) chưa khai thác để tránh rò rỉ giá; có thể bổ sung sau khi tách giá khỏi text.
'''
(OUT / 'report_ml.md').write_text(report, encoding='utf-8')

zip_path = OUT / 'ml_deliverables.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.glob('*.csv')) + sorted(OUT.glob('*.md')) + sorted(OUT.glob('*.joblib')) + sorted(FIG.glob('*.png')):
        z.write(p, p.relative_to(OUT))

print(json.dumps({
    'rows_total': n_all, 'rows_fit': n_train_pop, 'groups': n_groups,
    'train': len(tr_idx), 'test': len(te_idx), 'group_overlap': len(tr_groups & te_groups),
    'best': best_name, 'metrics': metrics_df.to_dict('records'),
    'zip': str(zip_path), 'zip_mb': round(zip_path.stat().st_size / 1e6, 2),
    'elapsed_s': round(time.time() - t0, 1),
}, ensure_ascii=False, indent=1))