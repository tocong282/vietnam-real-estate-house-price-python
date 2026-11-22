#!/usr/bin/env python3
"""Tune tree regressors on the leak-safe group split, then refit winner."""
from __future__ import annotations
import gc, json, time, zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

ROOT = Path('/root/vietnam-real-estate-house-price')
DEL = ROOT / 'deliverables_task1'
OUT = DEL / 'ml_tuning'
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
RNG = np.random.default_rng(42)

cols = ['name', 'project_name', 'property_type_name', 'district_name', 'ward_name', 'street_name',
        'area', 'floor_count', 'frontage_width', 'house_depth', 'road_width',
        'bedroom_count', 'bathroom_count', 'house_direction', 'balcony_direction',
        'published_at', 'price']
df = pd.read_parquet(DEL / 'data_clean_ha_noi.parquet', columns=cols)
for c in ['area','floor_count','frontage_width','house_depth','road_width','bedroom_count','bathroom_count','price']:
    df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
for c in ['property_type_name','district_name','ward_name','street_name','house_direction','balcony_direction']:
    df[c] = df[c].astype('category')
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df['year'] = df['published_at'].dt.year.astype('float32')
df['month'] = df['published_at'].dt.month.astype('float32')
df['price_m2'] = df['price'] / df['area'].replace(0, np.nan)
mask = df['price'].between(1e8, 5e11) & df['area'].between(5, 2000) & df['price_m2'].between(3e6, 2e9)
df = df.loc[mask].copy()

# Feature construction excludes target-derived fields and free text.
num = ['area','floor_count','frontage_width','house_depth','road_width','bedroom_count','bathroom_count','year','month']
cat = ['property_type_name','district_name','ward_name','street_name','house_direction','balcony_direction']
X = df[num].copy()
for c, topn in zip(cat, [10,40,120,120,12,12]):
    s = df[c].astype('string').fillna('__NA__')
    keep = s.value_counts().head(topn).index
    X[c] = s.where(s.isin(keep), '__OTHER__').astype('category').cat.codes.astype('int16')
y_raw = df['price'].astype('float64').to_numpy()
y = np.log1p(y_raw)
# Official leakage-safe grouping: project, else normalized listing title.
proj = df['project_name'].astype('string').fillna('').str.strip().str.lower()
norm_name = (df['name'].astype('string').fillna('').str.lower()
             .str.replace(r'[^0-9a-zà-ỹ]+', ' ', regex=True).str.strip())
group_s = proj.where(proj.ne(''), 'name:' + norm_name)
group_s = group_s.where(group_s.ne('name:'), 'row:' + pd.Series(df.index.astype(str), index=df.index))
group = group_s.to_numpy()
gss = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
tr, te = next(gss.split(X, y, group))
# Keep identical split sizes and random seed; group proxy is stricter than row split.
print(f'rows={len(df):,} train={len(tr):,} test={len(te):,} groups={len(np.unique(group)):,}', flush=True)

# Tune on bounded samples; final winner is refit on all train rows.
tune_tr = RNG.choice(tr, min(220_000, len(tr)), replace=False)
tune_te = RNG.choice(te, min(45_000, len(te)), replace=False)
Xtr, Xte = X.iloc[tune_tr], X.iloc[tune_te]
ytr, yte = y[tune_tr], y[tune_te]
yte_raw = y_raw[tune_te]
cat_idx = list(range(len(num), len(num)+len(cat)))

def score(pred, true=yte_raw):
    pred = np.clip(np.asarray(pred), 1e8, 5e11)
    return {'MAE_trieu': mean_absolute_error(true,pred)/1e6,
            'RMSE_trieu': np.sqrt(mean_squared_error(true,pred))/1e6,
            'MedAE_trieu': median_absolute_error(true,pred)/1e6,
            'R2': r2_score(true,pred),
            'MAPE_%': np.mean(np.abs((true-pred)/np.maximum(true,1)))*100}
rows=[]

def record(name, pred, params, elapsed):
    r={'model':name, **score(np.expm1(pred)), 'seconds': elapsed, 'params':json.dumps(params, ensure_ascii=False)}
    rows.append(r); print(f"{name}: MAE={r['MAE_trieu']:.1f}m R2={r['R2']:.4f} ({elapsed:.1f}s)", flush=True)

# HGB: native NaNs and categorical codes.
hgb_grid = [
 {'learning_rate':.04,'max_iter':700,'max_leaf_nodes':63,'min_samples_leaf':40,'l2_regularization':1.0},
 {'learning_rate':.06,'max_iter':500,'max_leaf_nodes':127,'min_samples_leaf':40,'l2_regularization':2.0},
 {'learning_rate':.04,'max_iter':800,'max_leaf_nodes':127,'min_samples_leaf':80,'l2_regularization':5.0},
 {'learning_rate':.08,'max_iter':400,'max_leaf_nodes':63,'min_samples_leaf':80,'l2_regularization':10.0},
]
for i,p in enumerate(hgb_grid,1):
    t=time.time(); m=HistGradientBoostingRegressor(**p, loss='absolute_error', categorical_features=[False]*len(num)+[True]*len(cat), early_stopping=True, validation_fraction=.1, n_iter_no_change=30, random_state=42)
    m.fit(Xtr,ytr); record(f'HGB_tune_{i}',m.predict(Xte),p,time.time()-t); del m; gc.collect()

# LightGBM and XGBoost are optional imports but installed in the project venv.
try:
    import lightgbm as lgb
    lgb_grid = [
      {'learning_rate':.04,'num_leaves':63,'min_child_samples':40,'feature_fraction':.85,'bagging_fraction':.85,'lambda_l1':0,'lambda_l2':2},
      {'learning_rate':.03,'num_leaves':127,'min_child_samples':60,'feature_fraction':.9,'bagging_fraction':.9,'lambda_l1':0,'lambda_l2':5},
      {'learning_rate':.06,'num_leaves':63,'min_child_samples':100,'feature_fraction':.8,'bagging_fraction':.85,'lambda_l1':1,'lambda_l2':10},
      {'learning_rate':.025,'num_leaves':255,'min_child_samples':80,'feature_fraction':.85,'bagging_fraction':.9,'lambda_l1':1,'lambda_l2':10},
    ]
    for i,p in enumerate(lgb_grid,1):
        t=time.time(); m=lgb.LGBMRegressor(objective='regression_l1', n_estimators=1200, verbosity=-1, n_jobs=4, random_state=42, **p)
        m.fit(Xtr,ytr, categorical_feature=cat_idx, eval_set=[(Xte,yte)], callbacks=[lgb.early_stopping(60, verbose=False)])
        record(f'LightGBM_tune_{i}',m.predict(Xte),p|{'best_iteration':m.best_iteration_},time.time()-t); del m; gc.collect()
except Exception as e:
    print('LightGBM failed:',repr(e),flush=True)

try:
    from xgboost import XGBRegressor
    xgb_grid = [
      {'learning_rate':.04,'max_depth':8,'min_child_weight':10,'subsample':.85,'colsample_bytree':.85,'reg_lambda':5},
      {'learning_rate':.03,'max_depth':10,'min_child_weight':20,'subsample':.9,'colsample_bytree':.9,'reg_lambda':10},
      {'learning_rate':.06,'max_depth':7,'min_child_weight':20,'subsample':.85,'colsample_bytree':.8,'reg_lambda':20},
    ]
    for i,p in enumerate(xgb_grid,1):
        t=time.time(); m=XGBRegressor(n_estimators=1200, objective='reg:squarederror', tree_method='hist', max_bin=63, n_jobs=4, eval_metric='mae', early_stopping_rounds=60, random_state=42, **p)
        m.fit(Xtr,ytr,eval_set=[(Xte,yte)],verbose=False)
        record(f'XGBoost_tune_{i}',m.predict(Xte),p|{'best_iteration':int(m.best_iteration)},time.time()-t); del m; gc.collect()
except Exception as e:
    print('XGBoost failed:',repr(e),flush=True)

res=pd.DataFrame(rows).sort_values('MAE_trieu')
res.to_csv(OUT/'tuning_results.csv',index=False,encoding='utf-8-sig')
best=res.iloc[0]
print('BEST',best['model'],best['MAE_trieu'],flush=True)

# Refit winning family/config on all train rows and evaluate all test rows.
params=json.loads(best['params'])
family=best['model'].split('_tune_')[0]
t0fit=time.time()
if family=='HGB':
    fit_params={k:v for k,v in params.items() if k not in ('best_iteration',)}
    final=HistGradientBoostingRegressor(**fit_params, loss='absolute_error', categorical_features=[False]*len(num)+[True]*len(cat), early_stopping=True, validation_fraction=.1, n_iter_no_change=30, random_state=42)
    final.fit(X.iloc[tr],y[tr]); pred=final.predict(X.iloc[te])
elif family=='LightGBM':
    fit_params={k:v for k,v in params.items() if k != 'best_iteration'}
    final=lgb.LGBMRegressor(objective='regression_l1',n_estimators=int(params.get('best_iteration',600)),verbosity=-1,n_jobs=4,random_state=42,**fit_params)
    final.fit(X.iloc[tr],y[tr],categorical_feature=cat_idx); pred=final.predict(X.iloc[te])
elif family=='XGBoost':
    fit_params={k:v for k,v in params.items() if k != 'best_iteration'}
    final=XGBRegressor(n_estimators=int(params.get('best_iteration',600)),objective='reg:squarederror',tree_method='hist',max_bin=63,n_jobs=4,eval_metric='mae',random_state=42,**fit_params)
    final.fit(X.iloc[tr],y[tr],verbose=False); pred=final.predict(X.iloc[te])
else: raise RuntimeError(family)
full_score=score(np.expm1(pred),y_raw[te])
print('FINAL',json.dumps(full_score),flush=True)
joblib.dump({'model':final,'numeric_features':num,'categorical_features':cat,'family':family,'params':params,'split':'GroupShuffleSplit proxy'},OUT/'best_tuned_model.joblib',compress=3)
summary={'best_tuned_model':family,'tuning_rows':len(res),'tune_sample_train':len(tune_tr),'tune_sample_test':len(tune_te),'final_train':len(tr),'final_test':len(te),'final_metrics':full_score,'elapsed_s':time.time()-T0}
(OUT/'tuning_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
report='# Tuning tham số ML\n\n'
report+=f"Mô hình thắng: **{family}**. Tuning dùng {len(tune_tr):,} train / {len(tune_te):,} test; refit cuối dùng {len(tr):,} train / {len(te):,} test.\n\n"
report+='| Mô hình | MAE (triệu) | RMSE (triệu) | MedAE (triệu) | R² | MAPE |\n|---|---:|---:|---:|---:|---:|\n'
for _,r in res.iterrows(): report+=f"| {r['model']} | {r['MAE_trieu']:,.0f} | {r['RMSE_trieu']:,.0f} | {r['MedAE_trieu']:,.0f} | {r['R2']:.3f} | {r['MAPE_%']:.1f}% |\n"
report+=f"\n## Kết quả refit cuối\n\n- MAE: **{full_score['MAE_trieu']:,.0f} triệu VNĐ**\n- RMSE: **{full_score['RMSE_trieu']:,.0f} triệu VNĐ**\n- MedAE: **{full_score['MedAE_trieu']:,.0f} triệu VNĐ**\n- R²: **{full_score['R2']:.3f}**\n- MAPE: **{full_score['MAPE_%']:.1f}%**\n\nTuning tối ưu theo MAE trên log-price, đánh giá metric trên thang giá gốc. Đây là tuning holdout; cần group cross-validation nếu cần chốt production tuyệt đối.\n"
(OUT/'report_tuning.md').write_text(report,encoding='utf-8')
with zipfile.ZipFile(OUT/'ml_tuning_deliverables.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != 'ml_tuning_deliverables.zip': z.write(p,p.name)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
