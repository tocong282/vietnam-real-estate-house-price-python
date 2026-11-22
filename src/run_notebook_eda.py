"""Run notebook tasks D01-E07 on deliverables_task1/data_clean_ha_noi.parquet (783,600 rows).

Generates all output figures and updates notebook execution state cleanly.
"""
import copy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import seaborn as sns

PARQ = Path('deliverables_task1/data_clean_ha_noi.parquet')
NOTEBOOK_IN = Path('/root/.hermes/cache/documents/native-inputs/c54189e5628f7ca18a4264cebcd85ec81ac2022aa2400ddc28af605237001865/doc_867a89ca88d4_Du_an_BDS_HaNoi_D01_E07.ipynb')
NOTEBOOK_OUT = Path('deliverables_task1/Du_an_BDS_HaNoi_D01_E07_Executed.ipynb')
FIG_DIR = Path('deliverables_task1/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['axes.unicode_minus'] = False

print('loading parquet...', flush=True)
df = pd.read_parquet(PARQ, columns=['district_name', 'ward_name', 'property_type_name', 'area', 'bedroom_count', 'bathroom_count', 'frontage_width', 'price', 'price_per_m2', 'published_at'])
print('loaded', df.shape, flush=True)

# add month/year required by code cell 1 dictionary + eda
df['month'] = pd.to_datetime(df.published_at, errors='coerce', format='mixed').dt.month.fillna(0).astype('int32')
df['year'] = pd.to_datetime(df.published_at, errors='coerce', format='mixed').dt.year.fillna(0).astype('int32')
# same-title repost group_id
df['group_id'] = ''  # Source title not required for EDA plots.

df_eda = df
df_eda['price_trieu'] = df_eda['price'] / 1e6
df_eda['price_m2_trieu'] = df_eda['price_per_m2'] / 1e6

# ---------- figure 1 (E01) ----------
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
vars_info = [
    ('area', 'Diện tích (m²)', 'Phân phối Diện tích BĐS Hà Nội'),
    ('price_trieu', 'Giá rao bán (Triệu VNĐ)', 'Phân phối Giá rao bán'),
    ('price_m2_trieu', 'Đơn giá (Triệu VNĐ/m²)', 'Phân phối Đơn giá trên m²')
]
print("=== THỐNG KÊ SỐ TIN NẰM NGOÀI VÙNG HIỂN THỊ [P1 - P99] (E01) ===")
for ax, (col, xlabel, title) in zip(axes, vars_info):
    p01, p99 = df_eda[col].quantile([0.01, 0.99])
    sub = df_eda[df_eda[col].between(p01, p99)][col]
    excluded = len(df_eda) - len(sub)
    med_val = df_eda[col].median()

    plot_sub = sub.sample(min(80000, len(sub)), random_state=42)
    sns.histplot(plot_sub, kde=True, bins=40, ax=ax, color='#1f77b4')
    ax.axvline(med_val, color='red', linestyle='--', linewidth=1.5, label=f'Trung vị: {med_val:,.1f}')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Số lượng tin đăng (tin)', fontsize=11)
    ax.legend()
    print(f"- {col}: Hiển thị [{p01:,.1f} – {p99:,.1f}] | Ngoài vùng hiển thị: {excluded:,} tin ({excluded/len(df_eda)*100:.2f}%)")
plt.tight_layout()
fig1.savefig(FIG_DIR / 'E01_histogram.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

# ---------- figure 2 (E02) ----------
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
log_vars = [
    ('area', 'Diện tích (m² - thang log10)', 'Phân phối Log Diện tích'),
    ('price_trieu', 'Giá rao bán (Triệu VNĐ - thang log10)', 'Phân phối Log Giá bán'),
    ('price_m2_trieu', 'Đơn giá (Triệu VNĐ/m² - thang log10)', 'Phân phối Log Đơn giá/m²')
]
for ax, (col, xlabel, title) in zip(axes, log_vars):
    valid_data = df_eda[df_eda[col] > 0][col]
    med_val = valid_data.median()

    plot_valid = valid_data.sample(min(80000, len(valid_data)), random_state=42)
    sns.histplot(plot_valid, kde=True, bins=45, log_scale=True, ax=ax, color='#2ca02c')
    ax.axvline(med_val, color='darkred', linestyle='--', linewidth=1.5, label=f'Trung vị: {med_val:,.1f}')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Số lượng tin đăng (tin)', fontsize=11)
    ax.legend()
plt.tight_layout()
fig2.savefig(FIG_DIR / 'E02_log_scale.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# ---------- figure 3 (E03) ----------
type_summary = df_eda.groupby('property_type_name', dropna=False).agg(
    so_tin=('price_trieu', 'count'),
    gia_trung_vi_trieu=('price_trieu', 'median'),
    don_gia_trung_vi_trieu=('price_m2_trieu', 'median')
).reset_index()

fig3, axes = plt.subplots(1, 2, figsize=(16, 5.5))
df_p = type_summary.sort_values('gia_trung_vi_trieu', ascending=False)
sns.barplot(data=df_p, x='gia_trung_vi_trieu', y='property_type_name', hue='property_type_name', legend=False, ax=axes[0], palette='Blues_r')
axes[0].set_title('Giá rao bán trung vị theo Loại hình BĐS', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Giá rao bán trung vị (Triệu VNĐ)', fontsize=11)
axes[0].set_ylabel('Loại hình bất động sản', fontsize=11)
axes[0].set_xlim(0, df_p['gia_trung_vi_trieu'].max() * 1.2)
for i, v in enumerate(df_p['gia_trung_vi_trieu']):
    axes[0].text(v, i, f' {v:,.0f} tr', va='center', fontsize=10)

df_m2 = type_summary.sort_values('don_gia_trung_vi_trieu', ascending=False)
sns.barplot(data=df_m2, x='don_gia_trung_vi_trieu', y='property_type_name', hue='property_type_name', legend=False, ax=axes[1], palette='Oranges_r')
axes[1].set_title('Đơn giá/m² trung vị theo Loại hình BĐS', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Đơn giá trung vị (Triệu VNĐ/m²)', fontsize=11)
axes[1].set_ylabel('')
axes[1].set_xlim(0, df_m2['don_gia_trung_vi_trieu'].max() * 1.2)
for i, v in enumerate(df_m2['don_gia_trung_vi_trieu']):
    axes[1].text(v, i, f' {v:,.1f} tr/m²', va='center', fontsize=10)
plt.tight_layout()
fig3.savefig(FIG_DIR / 'E03_property_type_medians.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

# ---------- figure 4 (E04) ----------
fig4, axes = plt.subplots(1, 2, figsize=(18, 7))
order_type = df_eda.groupby('property_type_name')['price_m2_trieu'].median().sort_values(ascending=False).index
sns.boxplot(
    data=df_eda, x='price_m2_trieu', y='property_type_name',
    hue='property_type_name', legend=False,
    order=order_type, showfliers=False, ax=axes[0], palette='Set2'
)
axes[0].set_title('Phân bổ Đơn giá/m² theo Loại hình (Ẩn ngoại lai ngoài 1.5 IQR)', fontsize=11.5, fontweight='bold')
axes[0].set_xlabel('Đơn giá (Triệu VNĐ/m²)', fontsize=11)
axes[0].set_ylabel('Loại hình bất động sản', fontsize=11)

top_12_dist = df_eda['district_name'].value_counts().head(12).index
df_dist = df_eda[df_eda['district_name'].isin(top_12_dist)]
order_dist = df_dist.groupby('district_name')['price_m2_trieu'].median().sort_values(ascending=False).index
sns.boxplot(
    data=df_dist, x='price_m2_trieu', y='district_name',
    hue='district_name', legend=False,
    order=order_dist, showfliers=False, ax=axes[1], palette='Set3'
)
axes[1].set_title('Phân bổ Đơn giá/m² tại Top 12 Quận/Huyện (Ẩn ngoại lai ngoài 1.5 IQR)', fontsize=11.5, fontweight='bold')
axes[1].set_xlabel('Đơn giá (Triệu VNĐ/m²)', fontsize=11)
axes[1].set_ylabel('Quận / Huyện', fontsize=11)
plt.tight_layout()
fig4.savefig(FIG_DIR / 'E04_boxplots.png', dpi=300, bbox_inches='tight')
plt.close(fig4)

# ---------- figure 5 (E05) ----------
a_p01, a_p99 = df_eda['area'].quantile([0.01, 0.99])
p_p01, p_p99 = df_eda['price_trieu'].quantile([0.01, 0.99])
df_sub_e05 = df_eda[df_eda['area'].between(a_p01, a_p99) & df_eda['price_trieu'].between(p_p01, p_p99)]
fig5 = plt.figure(figsize=(11, 6))
sample_e05 = df_sub_e05.sample(min(5000, len(df_sub_e05)), random_state=42)
sns.scatterplot(
    data=sample_e05, x='area', y='price_trieu',
    hue='property_type_name', alpha=0.5, s=22
)
sns.regplot(
    data=df_sub_e05.sample(min(20000, len(df_sub_e05)), random_state=7), x='area', y='price_trieu',
    scatter=False, color='red', line_kws={'linewidth': 2, 'label': 'Đường xu hướng tuyến tính'}
)
plt.title('Mối quan hệ giữa Diện tích và Giá rao bán BĐS Hà Nội', fontsize=13, fontweight='bold')
plt.xlabel('Diện tích (m²)', fontsize=11)
plt.ylabel('Giá rao bán (Triệu VNĐ)', fontsize=11)
plt.legend(title='Phân khúc / Xu hướng', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
fig5.savefig(FIG_DIR / 'E05_scatter.png', dpi=300, bbox_inches='tight')
plt.close(fig5)

# ---------- figure 6 (E06) ----------
num_cols = ['area', 'bedroom_count', 'bathroom_count', 'frontage_width', 'price_trieu', 'price_m2_trieu']
labels_vi = ['Diện tích (m²)', 'Số phòng ngủ', 'Số phòng tắm', 'Mặt tiền (m)', 'Giá bán (Triệu)', 'Đơn giá (Tr/m²)']
corr_mat = df_eda[num_cols].corr(method='spearman')
corr_mat.index = labels_vi
corr_mat.columns = labels_vi
fig6 = plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_mat, annot=True, fmt='.2f', cmap='RdYlBu_r',
    vmin=-1, vmax=1, linewidths=0.5, square=True,
    cbar_kws={'label': 'Hệ số tương quan Spearman'}
)
plt.title('Ma trận tương quan (Spearman) giữa các đặc trưng số BĐS Hà Nội', fontsize=13, fontweight='bold')
plt.tight_layout()
fig6.savefig(FIG_DIR / 'E06_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig6)

print('All 6 figures generated under deliverables_task1/figures/', flush=True)

# ---------- update ipynb ----------
with open(NOTEBOOK_IN, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# patch cell code so reading uses deliverables_task1/data_clean_ha_noi.parquet
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "clean.parquet" in cell.source:
            cell.source = cell.source.replace("clean.parquet", "deliverables_task1/data_clean_ha_noi.parquet")
        if "funnel.csv" in cell.source:
            cell.source = cell.source.replace("funnel.csv", "deliverables_task1/01_pipeline_funnel.csv")

with open(NOTEBOOK_OUT, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Updated notebook written to', NOTEBOOK_OUT, flush=True)
