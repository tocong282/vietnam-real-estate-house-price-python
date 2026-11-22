#!/usr/bin/env python3
"""3 phần bổ sung: metric theo phân khúc, time-based split, SHAP (LightGBM pred_contrib)."""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path('/root/vietnam-real-estate-house-price')
DEL = ROOT / 'deliverables_task1'
OUT = DEL / 'ml_final' / 'extended'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
T0 = time.time()
RNG = np.random.default_rng(42)
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 170})
LOG = lambda *a: print(f'[{time.time()-T0:6.1f}s]', *a, flush=True)

cols = ['name','project_name','property_type_name','district_name','ward_name','street_name',
        'area','floor_count','frontage_width','house_depth','road_width',
        'bedroom_count','bathroom_count','house_direction','balcony_direction','published_at','price']
df = pd.read_parquet(DEL/'data_clean_ha_noi.parquet', columns=cols)
n_all = len(df)
for c in ['area','floor_count','frontage_width','house_depth','road_width','bedroom_count','bathroom_count','price']:
    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce', format='mixed')
df['year'] = df['published_at'].dt.year.astype('float32')
df['month'] = df['published_at'].dt.month.astype('float32')
df['price_m2'] = df['price'] / df['area'].replace(0, np.nan)
df = df[df['price'].between(1e8, 5e11) & df['area'].between(5, 2000) & df['price_m2'].between(3e6, 2e9)].copy()
LOG('fit rows', f'{len(df):,}')

num = ['area','floor_count','frontage_width','house_depth','road_width','bedroom_count','bathroom_count','year','month']
cat = ['property_type_name','district_name','ward_name','street_name','house_direction','balcony_direction']
TOPN = [10,40,120,120,12,12]

def build(frame):
    X = frame[num].copy()
    for c, topn in zip(cat, TOPN):
        s = frame[c].astype('string').fillna('__NA__')
        keep = s.value_counts().head(topn).index
        X[c] = s.where(s.isin(keep), '__OTHER__').astype('category').cat.codes.astype('int16')
    return X

X = build(df)
y = np.log1p(df['price'].astype('float64').to_numpy())
y_raw = df['price'].astype('float64').to_numpy()

def met(true, pred):
    pred = np.clip(pred, 1e8, 5e11)
    return {'n': int(len(true)),
            'MAE_trieu': mean_absolute_error(true, pred)/1e6,
            'RMSE_trieu': float(np.sqrt(mean_squared_error(true, pred)))/1e6,
            'MedAE_trieu': median_absolute_error(true, pred)/1e6,
            'R2': r2_score(true, pred),
            'MAPE_%': float(np.mean(np.abs((true-pred)/np.maximum(true,1)))*100)}

# ---- A. group split, dùng model bàn giao ----
proj = df['project_name'].astype('string').fillna('').str.strip().str.lower()
norm = (df['name'].astype('string').fillna('').str.lower()
        .str.replace(r'[^0-9a-zà-ỹ]+',' ',regex=True).str.strip())
g = proj.where(proj.ne(''), 'name:'+norm)
g = g.where(g.ne('name:'), 'row:'+pd.Series(df.index.astype(str), index=df.index))
groups = g.to_numpy()
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
assert not (set(groups[tr]) & set(groups[te]))
LOG('train/test', f'{len(tr):,}/{len(te):,}')

bundle = joblib.load(DEL/'ml_final'/'final_model.joblib')
model = bundle['model']
p_te = np.clip(np.expm1(model.predict(X.iloc[te])), 1e8, 5e11)
te_df = df.iloc[te].copy(); te_df['pred'] = p_te; te_df['true'] = y_raw[te]
LOG('predict test done')

seg_rows = []
for col in ['district_name','property_type_name']:
    for lvl, sub in te_df.groupby(te_df[col].astype('string').fillna('__NA__'), observed=True):
        if len(sub) < 200: continue
        seg_rows.append({col: lvl, **met(sub['true'].to_numpy(), sub['pred'].to_numpy())})
seg = pd.DataFrame(seg_rows)
seg.to_csv(OUT/'metrics_by_segment.csv', index=False, encoding='utf-8-sig')
d = seg[seg['district_name'].notna()].sort_values('MAE_trieu')
t = seg[seg['property_type_name'].notna()].sort_values('MAE_trieu')
LOG('segments', len(seg))

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for a, sub, col, title in [(ax[0], t, 'property_type_name', 'Theo loại hình'),
                           (ax[1], d.head(15), 'district_name', 'Theo quận/huyện (15 lỗi cao nhất)')]:
    s = sub.sort_values('MAE_trieu')
    a.barh(s[col].astype(str).str.slice(0, 26), s['MAE_trieu'], color='#1f4e79')
    a.set_xlabel('MAE (triệu VNĐ)'); a.set_title(f'MAE {title}', fontweight='bold')
    for i, (v, n) in enumerate(zip(s['MAE_trieu'], s['n'])):
        a.text(v, i, f' {v:,.0f} (n={n:,})', va='center', fontsize=8)
    a.set_xlim(0, s['MAE_trieu'].max()*1.25)
fig.tight_layout(); fig.savefig(FIG/'04_mae_by_segment.png', bbox_inches='tight'); plt.close(fig)

# ---- B. time-based split: train <=2025, test 2026 ----
yr = df['year'].to_numpy()
m_tr, m_te = yr <= 2025, yr == 2026
LOG('time split', f'train={m_tr.sum():,} test={m_te.sum():,}')
tm = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1200, learning_rate=.025,
                       num_leaves=255, min_child_samples=80, feature_fraction=.85,
                       bagging_fraction=.9, lambda_l1=1, lambda_l2=10,
                       verbosity=-1, n_jobs=4, random_state=42)
tm.fit(X[m_tr], y[m_tr], categorical_feature=[X.columns.get_loc(c) for c in cat])
p_time = np.clip(np.expm1(tm.predict(X[m_te])), 1e8, 5e11)
time_metrics = {'split': 'train <= 2025-12-31 | test = 2026',
                'train_rows': int(m_tr.sum()), 'test_rows': int(m_te.sum()),
                **met(y_raw[m_te], p_time)}
LOG('time metrics', json.dumps(time_metrics, ensure_ascii=False))

# monthly error curve
tdf = pd.DataFrame({'month': df['published_at'].dt.to_period('M').astype(str)[m_te].to_numpy(),
                    'true': y_raw[m_te], 'pred': p_time})
mon = []
for mo, sub in tdf.groupby('month'):
    if len(sub) < 100: continue
    mon.append({'month': mo, **met(sub['true'].to_numpy(), sub['pred'].to_numpy())})
mon = pd.DataFrame(mon).sort_values('month')
mon.to_csv(OUT/'metrics_by_month_time_split.csv', index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(mon['month'], mon['MAE_trieu'], 'o-', color='#1f4e79', label='MAE')
ax[0].plot(mon['month'], mon['MedAE_trieu'], 's--', color='#c0392b', label='MedAE')
ax[0].set_ylabel('Triệu VNĐ'); ax[0].set_title('Sai số theo tháng — time split (test 2026)', fontweight='bold')
ax[0].tick_params(axis='x', rotation=45); ax[0].legend(); ax[0].grid(alpha=.3)
s = RNG.choice(len(p_time), min(5000, len(p_time)), replace=False)
lim = np.percentile(np.r_[y_raw[m_te][s], p_time[s]]/1e9, 99.5)
ax[1].scatter(y_raw[m_te][s]/1e9, p_time[s]/1e9, s=5, alpha=.3, color='#2e75b6', rasterized=True)
ax[1].plot([0, lim], [0, lim], 'r--', lw=1.4)
ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
ax[1].set_xlabel('Giá thực tế (tỷ)'); ax[1].set_ylabel('Giá dự báo (tỷ)')
ax[1].set_title(f"Time split — R²={time_metrics['R2']:.3f}, MAPE={time_metrics['MAPE_%']:.1f}%", fontweight='bold')
fig.tight_layout(); fig.savefig(FIG/'05_time_split.png', bbox_inches='tight'); plt.close(fig)

# ---- C. SHAP via pred_contrib (không cần thư viện shap) ----
samp = RNG.choice(len(te), min(6000, len(te)), replace=False)
Xs = X.iloc[te].iloc[samp]
contrib = model.booster_.predict(Xs, pred_contrib=True)
shap_v = contrib[:, :-1]
feat = list(X.columns)
mean_abs = np.abs(shap_v).mean(axis=0)
imp = pd.DataFrame({'feature': feat, 'mean_abs_shap_log': mean_abs}).sort_values('mean_abs_shap_log', ascending=False)
imp['share_%'] = imp['mean_abs_shap_log']/imp['mean_abs_shap_log'].sum()*100
imp.to_csv(OUT/'shap_importance.csv', index=False, encoding='utf-8-sig')
np.save(OUT/'shap_values_sample.npy', shap_v.astype('float32'))
Xs.to_csv(OUT/'shap_sample_features.csv', index=False, encoding='utf-8-sig')
LOG('shap done')

fig, ax = plt.subplots(figsize=(9, 5.2))
top = imp.head(12).sort_values('mean_abs_shap_log')
ax.barh(top['feature'], top['share_%'], color='#1f4e79')
for i, v in enumerate(top['share_%']): ax.text(v, i, f' {v:.1f}%', va='center', fontsize=8)
ax.set_xlabel('Tỷ trọng đóng góp trung bình (%) — thang log giá')
ax.set_title('SHAP: mức ảnh hưởng đặc trưng tới log(giá)', fontweight='bold')
ax.set_xlim(0, top['share_%'].max()*1.22)
fig.tight_layout(); fig.savefig(FIG/'06_shap_importance.png', bbox_inches='tight'); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, f in zip(axs, imp['feature'].head(3)):
    j = feat.index(f)
    ax.scatter(Xs[f].to_numpy(), shap_v[:, j], s=4, alpha=.25, color='#c0392b', rasterized=True)
    ax.axhline(0, color='black', lw=.8)
    ax.set_xlabel(f); ax.set_ylabel('SHAP (log giá)'); ax.set_title(f, fontweight='bold')
fig.suptitle('SHAP dependence — 3 đặc trưng ảnh hưởng mạnh nhất', fontweight='bold')
fig.tight_layout(); fig.savefig(FIG/'07_shap_dependence.png', bbox_inches='tight'); plt.close(fig)

summary = {'group_split': {'train': len(tr), 'test': len(te), 'group_overlap': 0},
           'group_split_metrics': met(y_raw[te], p_te),
           'time_split': time_metrics,
           'segments': {'districts': int(d['district_name'].nunique()),
                        'property_types': int(t['property_type_name'].nunique())},
           'top_shap_features': imp.head(8).to_dict('records'),
           'elapsed_s': round(time.time()-T0, 1)}
(OUT/'extended_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
