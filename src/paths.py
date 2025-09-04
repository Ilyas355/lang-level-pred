# src/paths.py
from pathlib import Path

CLEAN_DATA   = Path("data/clean/cleaned_lang_proficiency_results.csv")
MODEL_PATH   = Path("models/final_logistic_regression_pipeline.pkl")
PREPROCESSOR_PATH = Path("models/preprocessing_pipeline.pkl")

REPORT_TEST  = Path("reports/test_metrics_tuned.csv")
REPORT_CV    = Path("reports/cv_summary.csv")
