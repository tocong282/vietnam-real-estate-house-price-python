"""Runnable real-preview smoke test + bounded-histogram regression check."""
import json
import os
from pathlib import Path

os.environ['VIETNAM_RAW'] = str(Path(__file__).with_name('preview_50_kaggle.csv'))
os.environ['MPLBACKEND'] = 'Agg'
nb = json.loads(Path(__file__).with_name('kaggle_hanoi_all_property_types.ipynb').read_text())
assert len({c['id'] for c in nb['cells']}) == len(nb['cells'])
ns = {'__name__': '__main__'}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        exec(compile(''.join(c['source']), f'cell-{i}', 'exec'), ns)

pd, np = ns['pd'], ns['np']
raw = pd.read_csv(os.environ['VIETNAM_RAW'])
expected = raw[raw.province_name.eq('Hà Nội') & raw.price.gt(0) & raw.area.gt(0)]
assert set(ns['df'].property_type_name) == set(expected.property_type_name)
assert ns['df'].province_name.eq('Hà Nội').all()
assert len(ns['df']) == 19 and ns['df'].property_type_name.nunique() == 4
assert 'property_type_name' in ns['CAT']
assert set(ns['df'].group_id.iloc[ns['tr']]).isdisjoint(ns['df'].group_id.iloc[ns['te']])

# Synthetic input tests binning only, never reported as dataset/experiment results.
fig, axes = ns['plt'].subplots(1, 2)
values = np.r_[np.linspace(1, 30, 200), 2.424e6]
counts = ns['zoom_hist'](axes[0], values, 'test', '#0072B2')
assert counts.sum() == 199 and np.count_nonzero(counts) > 10
ns['zoom_hist'](axes[1], np.ones(20), 'constant edge case', '#009E73')
ns['plt'].close(fig)

import joblib
loaded = joblib.load(ns['OUT'] / 'ridge_log.joblib')
assert np.allclose(loaded.predict(ns['X'].iloc[ns['te']]), ns['preds']['Ridge + log(target)'])
assert all(not c.get('outputs') for c in nb['cells'] if c['cell_type'] == 'code')
print('PASS: every cell; 19 real Hanoi rows / 4 types; no fixed price/area cutoff; group split; bounded bins with extreme tail + constant input; model reload.')
