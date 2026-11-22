#!/usr/bin/env python3
"""Evaluate every model on one identical group-split test set, then export the winner."""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path('/root/vietnam-real-estate-house-price')
DEL = ROOT / 'deliverables_task1'
OUT = DEL / 'ml_final'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
T0 = time.time()
RNG = np.random.default_rng(42)
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 170})

cols = ['name', 'project_name', 'property_type_name', 'district_name', 'ward_name', 'street_name',
        'area', 'floor_count', 'frontage_width', 'house_depth', 'road_width',
        'bedroom_count', 'bathroom_count', 'house_direction', 'balcony_direction',
        'published_at', 'price']
df = pd.read_parquet(DEL / 'data_clean_ha_noi.parquet', columns=cols)
n_all = len(df)
for c in ['area','floor_count','frontage_width','house_depth','road_width','bedroom_count','bathroom_count','price']:
    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
for c in ['property_type_name','district_name','ward_name','street_name','house_direction','balcony_direction']:
    df[c] = df[c].astype('category')
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df['year'] = df['published_at'].dt.year.astype('float32')
df['month'] = df['published_at'].dt.month.astype('float32')
df['price_m2'] = df['price'] / df['area'].replace(0, np.nan)
df = df[df['price'].between(1e8, 5e11) & df['area'].between(5, 2000) & df['price_m2'].between(3e6, 2e9)].copy()
n_fit = len(df)

num = ['area','floor_count','frontage_width','house_depth','road_width','bedroom_count','bathroom_count','year','month']
cat = ['property_type_name','district_name','ward_name','street_name','house_direction','balcony_direction']
X = df[num].copy()
cat_codes = {}
for c, topn in zip(cat, [10,40,120,120,12,12]):
    s = df[c].astype('string').fillna('__NA__')
    keep = s.value_counts().head(topn).index
    codes = s.where(s.isin(keep), '__OTHER__').astype('category').cat.codes.astype('int16')
    X[c] = codes
    cat_codes[c] = codes

y_raw = df['price'].astype('float64').to_numpy()
y = np.log1p(y_raw)
types = df['property_type_name'].astype(str).to_numpy()

proj = df['project_name'].astype('string').fillna('').str.strip().str.lower()
norm_name = (df['name'].astype('string').fillna('').str.lower()
             .str.replace(r'[^0-9a-zà-ỹ]+', ' ', regex=True).str.strip())
g = proj.where(proj.ne(''), 'name:' + norm_name)
g = g.where(g.ne('name:'), 'row:' + pd.Series(df.index.astype(str), index=df.index))
groups = g.to_numpy()
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
assert not (set(groups[tr]) & set(groups[te]))
print(f'rows={n_all:,} fit={n_fit:,} train={len(tr):,} test={len(te):,} groups={len(set(groups)):,}', flush=True)

Xtr, Xte = X.iloc[tr], X.iloc[te]
ytr, yte = y[tr], y[te]
yte_raw, te_types = y_raw[te], types[te]
cat_mask = [c in cat for c in Xtr.columns]

def met(true, pred):
    pred = np.clip(pred, 1e8, 5e11)
    return {'MAE_trieu': mean_absolute_error(true, pred)/1e6,
            'RMSE_trieu': float(np.sqrt(mean_squared_error(true, pred)))/1e6,
            'MedAE_trieu': median_absolute_error(true, pred)/1e6,
            'R2': r2_score(true, pred),
            'MAPE_%': float(np.mean(np.abs((true-pred)/np.maximum(true,1)))*100)}

rows, preds, models = [], {}, {}

# 1. baseline
med_by_type = pd.Series(ytr, index=types[tr]).groupby(level=0).median()
p = pd.Series(te_types).map(med_by_type).fillna(np.median(ytr)).to_numpy()
preds['Baseline (median theo loại hình)'] = np.expm1(p)
rows.append({'model': 'Baseline (median theo loại hình)', **met(yte_raw, np.expm1(p))})
print('baseline', round(time.time()-T0,1), flush=True)

# 2. ridge
raw_tr, raw_te = df[num+cat].iloc[tr], df[num+cat].iloc[te]
pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)), ('sc', StandardScaler())]), num),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('oh', OneHotEncoder(handle_unknown='ignore', min_frequency=20))]), cat)])
ridge = Pipeline([('pre', pre), ('m', Ridge(alpha=100.0))]).fit(raw_tr, ytr)
p = np.expm1(np.clip(ridge.predict(raw_te), 0, np.log1p(5e11)))
preds['Ridge Regression'] = p
rows.append({'model': 'Ridge Regression', **met(yte_raw, p)})
models['Ridge Regression'] = ridge
print('ridge', round(time.time()-T0,1), flush=True); del ridge, pre; gc.collect()

# 3. random forest (subsample, dense)
rf_imp = SimpleImputer(strategy='median').fit(Xtr)
pos = RNG.choice(len(tr), min(120_000, len(tr)), replace=False)
rf = RandomForestRegressor(n_estimators=60, max_depth=18, min_samples_leaf=5,
                           max_features=0.5, n_jobs=4, random_state=42)
rf.fit(rf_imp.transform(Xtr.iloc[pos]), ytr[pos])
p = np.expm1(rf.predict(rf_imp.transform(Xte)))
preds['Random Forest'] = p
rows.append({'model': 'Random Forest', **met(yte_raw, p)})
models['Random Forest'] = rf
print('rf', round(time.time()-T0,1), flush=True); del rf, rf_imp; gc.collect()

# 4. HGB (winning config from tuning)
hgb = HistGradientBoostingRegressor(learning_rate=.06, max_iter=500, max_leaf_nodes=127,
                                    min_samples_leaf=40, l2_regularization=2.0,
                                    loss='absolute_error', categorical_features=cat_mask,
                                    early_stopping=True, validation_fraction=.1,
                                    n_iter_no_change=30, random_state=42).fit(Xtr, ytr)
p = np.expm1(hgb.predict(Xte))
preds['HistGradientBoosting'] = p
rows.append({'model': 'HistGradientBoosting', **met(yte_raw, p)})
models['HistGradientBoosting'] = hgb
print('hgb', round(time.time()-T0,1), flush=True)

# 5. XGBoost (best tuning config)
xgb = XGBRegressor(n_estimators=1200, learning_rate=.03, max_depth=10, min_child_weight=20,
                   subsample=.9, colsample_bytree=.9, reg_lambda=10, objective='reg:squarederror',
                   tree_method='hist', max_bin=63, n_jobs=4, random_state=42).fit(Xtr, ytr)
p = np.expm1(xgb.predict(Xte))
preds['XGBoost'] = p
rows.append({'model': 'XGBoost', **met(yte_raw, p)})
models['XGBoost'] = xgb
print('xgb', round(time.time()-T0,1), flush=True); del xgb; gc.collect()

# 6. LightGBM (best tuning config)
lgbm = lgb.LGBMRegressor(objective='regression_l1', n_estimators=1200, learning_rate=.025,
                         num_leaves=255, min_child_samples=80, feature_fraction=.85,
                         bagging_fraction=.9, lambda_l1=1, lambda_l2=10,
                         verbosity=-1, n_jobs=4, random_state=42)
lgbm.fit(Xtr, ytr, categorical_feature=[X.columns.get_loc(c) for c in cat])
p = np.expm1(lgbm.predict(Xte))
preds['LightGBM'] = p
rows.append({'model': 'LightGBM', **met(yte_raw, p)})
models['LightGBM'] = lgbm
print('lgbm', round(time.time()-T0,1), flush=True)

res = pd.DataFrame(rows).sort_values('MAE_trieu')
res.to_csv(OUT/'metrics_all_models.csv', index=False, encoding='utf-8-sig')
best_name = res.iloc[0]['model']
best = preds[best_name]
best_model = models[best_name]

by_type = []
for t in sorted(set(te_types)):
    m = te_types == t
    by_type.append({'property_type_name': t, 'n_test': int(m.sum()), **met(yte_raw[m], best[m])})
by_type = pd.DataFrame(by_type).sort_values('MAE_trieu')
by_type.insert(0, 'model', best_name)
by_type.to_csv(OUT/'metrics_by_property_type.csv', index=False, encoding='utf-8-sig')

# feature importance from LightGBM gain (cheap, no extra fits)
imp = pd.DataFrame({'feature': X.columns, 'importance': lgbm.booster_.feature_importance('gain')}).sort_values('importance', ascending=False)
imp.to_csv(OUT/'feature_importance.csv', index=False, encoding='utf-8-sig')

# figures
s = RNG.choice(len(yte_raw), min(6000, len(yte_raw)), replace=False)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].scatter(yte_raw[s]/1e9, best[s]/1e9, s=6, alpha=.3, color='#1f4e79', rasterized=True)
lim = np.percentile(np.r_[yte_raw[s], best[s]]/1e9, 99.5)
ax[0].plot([0, lim], [0, lim], 'r--', lw=1.5, label='Dự báo hoàn hảo')
ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
ax[0].set_xlabel('Giá thực tế (tỷ VNĐ)'); ax[0].set_ylabel('Giá dự báo (tỷ VNĐ)')
ax[0].set_title(f'Thực tế vs Dự báo — {best_name}', fontweight='bold'); ax[0].legend()
r = (best - yte_raw)/1e9
ax[1].hist(r[np.abs(r) < np.percentile(np.abs(r), 99)], bins=60, color='#c0392b', alpha=.85)
ax[1].axvline(0, color='black', lw=1.2)
ax[1].set_xlabel('Sai số (tỷ VNĐ)'); ax[1].set_ylabel('Số tin')
ax[1].set_title('Phân phối sai số (cắt P99)', fontweight='bold')
fig.tight_layout(); fig.savefig(FIG/'01_actual_vs_predicted.png', bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.8))
mm = res.sort_values('MAE_trieu')
ax.barh(mm['model'], mm['MAE_trieu'], color='#2e75b6')
for i, v in enumerate(mm['MAE_trieu']): ax.text(v, i, f' {v:,.0f}', va='center')
ax.set_xlabel('MAE (triệu VNĐ) — thấp hơn tốt hơn')
ax.set_title('So sánh mô hình trên cùng tập test', fontweight='bold')
ax.set_xlim(0, mm['MAE_trieu'].max()*1.18)
fig.tight_layout(); fig.savefig(FIG/'02_model_comparison.png', bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
top = imp.head(12).sort_values('importance')
ax.barh(top['feature'], top['importance']/top['importance'].max(), color='#1f4e79')
ax.set_xlabel('Mức quan trọng tương đối (gain, chuẩn hoá)')
ax.set_title('Đặc trưng quan trọng — LightGBM', fontweight='bold')
fig.tight_layout(); fig.savefig(FIG/'03_feature_importance.png', bbox_inches='tight'); plt.close(fig)

joblib.dump({'model': best_model, 'family': best_name, 'numeric_features': num,
             'categorical_features': cat, 'cat_top': dict(zip(cat,[10,40,120,120,12,12])),
             'target': 'log1p(price)', 'metrics': res.iloc[0].to_dict()},
            OUT/'final_model.joblib', compress=3)

summary = {'rows_total': n_all, 'rows_fit': n_fit, 'train': len(tr), 'test': len(te),
           'groups': len(set(groups)), 'group_overlap': 0, 'best_model': best_name,
           'metrics': res.to_dict('records'), 'elapsed_s': round(time.time()-T0, 1)}
(OUT/'final_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
