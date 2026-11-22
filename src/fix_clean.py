"""Rewrite clean parquet with GLOBAL dedupe using a compact hash array (no Python set)."""
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path('deliverables_task1')
SRC = OUT / '_tmp' / 'hn_valid.parquet'
DST = OUT / 'data_clean_ha_noi.parquet'
pf = pq.ParquetFile(SRC)
n = pf.metadata.num_rows


def key_hash(d):
    keys = (d.name.fillna('').str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
            + '|' + d.district_name.fillna('') + '|' + d.ward_name.fillna('')
            + '|' + d.area.round(2).astype(str) + '|' + d.price.round(0).astype(str))
    return pd.util.hash_pandas_object(keys, index=False).to_numpy()


hashes = np.empty(n, dtype=np.uint64)
i = 0
for batch in pf.iter_batches(batch_size=100_000, columns=['name', 'district_name',
                                                          'ward_name', 'area', 'price']):
    h = key_hash(batch.to_pandas())
    hashes[i:i + len(h)] = h
    i += len(h)
print('hashed', i, flush=True)
mask = ~pd.Series(hashes).duplicated().to_numpy()
print('keep', int(mask.sum()), flush=True)
del hashes
gc.collect()

schema = pf.schema_arrow
writer = pq.ParquetWriter(DST, schema)
pos = 0
first = True
for batch in pf.iter_batches(batch_size=100_000):
    ln = batch.num_rows
    m = mask[pos:pos + ln]
    pos += ln
    d = batch.to_pandas().loc[m]
    if first and len(d):
        d.head(5000).to_csv(OUT / '08_preview_clean_5000.csv', index=False, encoding='utf-8-sig')
        first = False
    if len(d):
        writer.write_table(pa.Table.from_pandas(d, preserve_index=False).cast(schema))
    del d
    gc.collect()
writer.close()
print('written', pq.ParquetFile(DST).metadata.num_rows)
