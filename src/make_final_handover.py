import os, zipfile
from pathlib import Path

base = Path('/root/vietnam-real-estate-house-price')
out_dir = base / 'final_handover'
out_dir.mkdir(parents=True, exist_ok=True)

manifest = {
  'Phần 1 — Lọc & Từ điển': [
    'deliverables_task1/data_dictionary.xlsx',
    'deliverables_task1/report_task1.docx',
    'deliverables_task1/report_task1.md',
    'deliverables_task1/01_pipeline_funnel.csv',
    'deliverables_task1/02_data_quality_report.csv',
    'deliverables_task1/03_segment_summary.csv',
    'deliverables_task1/04_outlier_thresholds.csv',
    'deliverables_task1/05_top30_price.csv',
    'deliverables_task1/06_top30_price_per_m2.csv',
    'deliverables_task1/07_data_flags.csv',
    'deliverables_task1/08_preview_clean_5000.csv',
  ],
  'Phần 2 — EDA': [
    'deliverables_task1/Du_an_BDS_HaNoi_D01_E07_Executed.ipynb',
    'deliverables_task1/EDA_E01_distribution_ranges.csv',
    'deliverables_task1/EDA_E03_summary_by_type.csv',
    'deliverables_task1/EDA_E04_top12_districts.csv',
    'deliverables_task1/EDA_E06_spearman_correlation.csv',
  ],
  'Phần 3 — Phân tích chi tiết (M01-M08)': [
    'deliverables_task1/part3_analysis/part3_analysis_tables.xlsx',
    'deliverables_task1/part3_analysis/report_part3.md',
    'deliverables_task1/part3_analysis/01_type_summary_full.csv',
    'deliverables_task1/part3_analysis/02_district_summary_full.csv',
    'deliverables_task1/part3_analysis/03_belt_summary_full.csv',
    'deliverables_task1/part3_analysis/04_correlation_full.csv',
    'deliverables_task1/part3_analysis/05_area_band_summary_full.csv',
    'deliverables_task1/part3_analysis/06_missing_summary_by_type_full.csv',
    'deliverables_task1/part3_analysis/07_missing_matrix_by_type_full.csv',
  ],
  'Phần 4 — Báo cáo đối soát & Review': [
    'deliverables_task1/review_report/E07_commentary.md',
    'deliverables_task1/review_report/M04_M08_insights.md',
  ],
  'Phần 5 — Machine Learning Baseline': [
    'deliverables_task1/ml/report_ml.md',
    'deliverables_task1/ml/metrics.csv',
    'deliverables_task1/ml/metrics_by_property_type.csv',
    'deliverables_task1/ml/leakage_audit.csv',
    'deliverables_task1/ml/feature_importance.csv',
    'deliverables_task1/ml/model_hgb.joblib',
  ],
  'Phần 6 — Machine Learning Tuned (LightGBM)': [
    'deliverables_task1/ml_tuning/report_tuning.md',
    'deliverables_task1/ml_tuning/tuning_results.csv',
    'deliverables_task1/ml_tuning/tuning_summary.json',
    'deliverables_task1/ml_tuning/best_tuned_model.joblib',
  ],
}

full_zip = out_dir / 'BAO_CAO_VA_DELIVERABLES_TONG_HOP_BDS_HA_NOI.zip'
with zipfile.ZipFile(full_zip, 'w', zipfile.ZIP_DEFLATED) as z:
  for sec, files in manifest.items():
    for f in files:
      p = base / f
      if p.exists():
        z.write(p, p.relative_to(base / 'deliverables_task1'))

print('ZIP bytes:', full_zip.stat().st_size)
