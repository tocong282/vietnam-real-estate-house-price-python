import pyarrow.parquet as pq, pandas as pd
n = pq.ParquetFile('data_clean_ha_noi.parquet').metadata.num_rows
print('clean rows:', n)
d = pd.read_csv('08_preview_clean_5000.csv')
print('preview rows:', len(d))
t = pq.read_table('data_clean_ha_noi.parquet', columns=['property_type_name', 'price', 'area'])
df = t.to_pandas()
print(df.groupby('property_type_name').size().sort_values(ascending=False).to_string())
print('median price (ty):', round(df.price.median() / 1e9, 3), 'median area:', df.area.median())
