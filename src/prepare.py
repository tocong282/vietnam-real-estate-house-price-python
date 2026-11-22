"""Extract Hanoi house listings in chunks; leave fitted preprocessing to train.py."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUM = ['area', 'bedroom_count', 'bathroom_count', 'frontage_width']
CAT = ['district_name', 'ward_name']
COLS = ['name', 'property_type_name', 'province_name', 'published_at', 'price'] + NUM + CAT


def clean(chunk):
    d = chunk.copy()
    for col in COLS:
        if col not in NUM + ['price']:
            d[col] = d[col].astype('string').str.normalize('NFC').str.strip()
    for col in NUM + ['price']:
        d[col] = pd.to_numeric(d[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    d = d.loc[
        d.property_type_name.eq('Nhà') & d.province_name.eq('Hà Nội')
        & d.area.between(10, 500) & d.price.gt(0) & d.name.notna() & d.name.ne('')
    ].copy()
    # ponytail: declared 10-500 m2 study domain; expand only after checking other segments.
    for col, maximum in [('bedroom_count', 30), ('bathroom_count', 30), ('frontage_width', 50)]:
        d.loc[~d[col].between(0.1, maximum), col] = np.nan
    d['published_at'] = pd.to_datetime(d.published_at, errors='coerce', format='mixed')
    # Same-title repost proxy, NOT a verified physical-property identifier.
    keys = (d.province_name.fillna('') + '|' + d.district_name.fillna('') + '|'
            + d.name.str.lower().str.replace(r'\s+', ' ', regex=True))
    d['group_id'] = keys.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', default='data/clean.parquet')
    ap.add_argument('--chunksize', type=int, default=100_000)
    a = ap.parse_args()
    if a.chunksize < 1:
        ap.error('--chunksize must be positive')
    parts, total = [], 0
    for chunk in pd.read_csv(a.raw, usecols=COLS, dtype='string', chunksize=a.chunksize):
        total += len(chunk)
        kept = clean(chunk)
        if not kept.empty:
            parts.append(kept)
    if not parts:
        raise ValueError('No rows match scope: Hanoi, Nha, 10-500 m2, positive price, nonempty title')
    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    # Do not drop unrelated homes simply because they share area, street and price.
    df = df.drop_duplicates(subset=['group_id', 'price'] + NUM + CAT)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    report = {'raw_rows': total, 'scope_rows': before, 'deduplicated_rows': len(df),
              'groups': int(df.group_id.nunique()),
              'warning': 'Positive prices retained, not verified. Audit extreme prices before final experiment.',
              'missing_fraction': df[NUM + CAT].isna().mean().to_dict()}
    out.with_suffix('.audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
