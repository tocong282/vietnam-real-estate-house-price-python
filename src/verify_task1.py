import pandas as pd, pyarrow.parquet as pq
print('clean rows:', pq.ParquetFile('data_clean_ha_noi.parquet').metadata.num_rows)
x = pd.ExcelFile('data_dictionary.xlsx'); print(x.sheet_names)
dd = pd.read_excel(x, 'Data Dictionary')
print(dd[['Tên cột', 'Kiểu dữ liệu', 'Tỷ lệ thiếu % (toàn bộ)', 'Tỷ lệ thiếu % (Hà Nội hợp lệ)']].to_string(index=False))
print(pd.read_csv('06_top30_price_per_m2.csv')[['name', 'area', 'price (tỷ)', 'price/m² (triệu)']].head(5).to_string(index=False))
print(pd.read_csv('05_top30_price.csv')[['name', 'area', 'price (tỷ)']].head(3).to_string(index=False))
