"""CPU baseline; preview run is a smoke test, never a quality experiment."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare import CAT, NUM


def make_model(estimator, log_target):
    prep = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True,
                                                  keep_empty_features=True)),
                          ('scale', StandardScaler())]), NUM),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CAT),
    ])
    pipe = Pipeline([('prep', prep), ('model', estimator)])
    return (TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
            if log_target else pipe)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='outputs')
    a = ap.parse_args()
    df = pd.read_parquet(a.data)
    if df.group_id.nunique() < 5:
        raise ValueError('Need at least 5 title groups for split; collect more real rows')
    # Convert nullable pandas strings to object/np.nan at sklearn boundary.
    X = df[NUM + CAT].copy()
    for c in CAT:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    y = df.price.to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Target must be positive and finite')
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
                  .split(X, y, groups=df.group_id))
    assert set(df.group_id.iloc[tr]).isdisjoint(df.group_id.iloc[te])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    fitted = None
    for name, estimator, log_target in [
        ('Median_baseline', DummyRegressor(strategy='median'), False),
        ('LinearRegression', LinearRegression(), False),
        ('Ridge_log_target', Ridge(alpha=10., solver='lsqr'), True),
    ]:
        model = make_model(estimator, log_target)
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        if not np.isfinite(pred).all():
            raise ValueError(f'{name}: nonfinite predictions; audit target extremes')
        rows.append({'model': name, 'MAE_VND': mean_absolute_error(y[te], pred),
                     'RMSE_VND': float(np.sqrt(mean_squared_error(y[te], pred))),
                     'R2_original_price': float(r2_score(y[te], pred)) if len(te) > 1 else None,
                     'MdAPE_percent': float(100 * np.median(np.abs(y[te] - pred) / y[te])),
                     'negative_prediction_count': int((pred < 0).sum())})
        if name == 'Ridge_log_target':
            fitted = model
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    joblib.dump(fitted, out / 'ridge_log.joblib')
    scope = {'train_rows': len(tr), 'test_rows': len(te), 'seed': 42,
             'evaluation': 'SMOKE_TEST_ONLY' if len(df) < 1000 else 'BASELINE_NOT_TUNED',
             'split': 'nonoverlapping normalized-title proxy groups; edited reposts can remain',
             'price_unit_assumption': 'VND; audit against listing text before final run',
             'note': 'No hyperparameters selected using test metrics; alpha=10 is fixed.'}
    (out / 'run.json').write_text(json.dumps(scope, ensure_ascii=False, indent=2))
    print(json.dumps(scope, ensure_ascii=False, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    # Exercise saved inference on real held-out rows.
    loaded = joblib.load(out / 'ridge_log.joblib')
    assert np.allclose(loaded.predict(X.iloc[te]), fitted.predict(X.iloc[te]))
    assert np.isfinite(loaded.predict(X.iloc[te])).all()


if __name__ == '__main__':
    main()
