# app_pages/about.py
import streamlit as st
from pathlib import Path

# Expected artefacts/paths
PROCESSED_DATA = Path("data/processed/cleaned_lang_proficiency_results.csv")
MODEL_PATH     = Path("models/final_logistic_regression_pipeline.pkl")
TEST_CSV       = Path("reports/test_metrics_tuned.csv")
CV_CSV         = Path("reports/cv_summary.csv")

# Performance targets (from your notebook)
TARGET_ACC  = 0.75
TARGET_F1_M = 0.70

def _badge(path: Path) -> str:
    return "✅" if path.exists() else "❌"

def render():
    st.header("About & Methods")

    st.subheader("Business Requirements")
    st.markdown("""
1. **The client has asked us to** analyse the learner dataset to surface patterns driving CEFR outcomes, **quantify class imbalance**, and **detect target leakage** from raw exam scores. From these findings, define **engineered, non-leaky features**.

2. **The client has asked us to** build and operationalise a **fair, interpretable CEFR classifier** using the engineered features, meeting targets (**Accuracy ≥ 0.75**, **Macro-F1 ≥ 0.70**), and expose a Streamlit **Predict CEFR** page with predicted label + class probabilities.
    """)

    st.subheader("ML Business Case (Classification)")
    st.markdown(f"""
- **Aim:** Predict CEFR level (A1–C2) from **engineered, leakage-free** features to support fair placement decisions.
- **Learning method:** Compare **Logistic Regression**, **Random Forest**, and **XGBoost**; tune via `GridSearchCV` (cv=5, `f1_macro`), then evaluate on a **hold-out test set**.
- **Performance targets (test set):** **Accuracy ≥ {TARGET_ACC:.2f}**, **Macro-F1 ≥ {TARGET_F1_M:.2f}**.
- **Decision rule:** maximise **Macro-F1**; tie-break on **Accuracy**.
- **Success criteria:** Targets met on test set; **failure** if either metric falls below target.
- **Model output & relevance:** Predicted CEFR + class probabilities; supports teacher/examiner/developer **verification** of placement decisions.
- **Fairness & imbalance:** Prefer **Macro-F1** for class balance; **Weighted-F1** is also reported but can mask minority-class performance.
- **Leakage control:** Raw exam totals/aggregates **excluded**. Training uses only **engineered features** (e.g., strongest/weakest skill, `*_minus_avg`, strength–weakness gap, learning profile, productive/receptive balance, per-skill level bins).
    """)

    st.subheader("Pipeline Summary")
    st.markdown("""
- **Data split & labels:** Training target = `cefr_encoded` (0–5). Human-readable target in EDA = `overall_cefr` (A1–C2).
- **Modelling set:** Only **engineered features** (no raw totals).  
- **Tuning:** Grid search per model with `f1_macro`, **cv=5**.  
- **Validation:**  
  - **Cross-validation:** out-of-fold metrics to check generalisation.  
  - **Final selection:** based on **test-set** metrics (Macro-F1 → Accuracy).  
- **Saved artefacts:** final tuned pipeline serialized for dashboard inference.
    """)

    st.subheader("Artefacts & Paths")
    st.markdown(f"""
- Processed data: `{PROCESSED_DATA.as_posix()}` {_badge(PROCESSED_DATA)}
- Final pipeline: `{MODEL_PATH.as_posix()}` {_badge(MODEL_PATH)}
- Test metrics (tuned): `{TEST_CSV.as_posix()}` {_badge(TEST_CSV)}
- CV summary: `{CV_CSV.as_posix()}` {_badge(CV_CSV)}
    """)

    st.subheader("Reproduce & Deploy")
    st.markdown("""
- **Reproduce metrics:** Run the Modelling & Evaluation notebook cells that export:
  - `reports/test_metrics_tuned.csv` (rows = models; cols = Accuracy, F1 (macro), F1 (weighted))
  - `reports/cv_summary.csv` (columns = Model, CV Accuracy, CV Macro-F1, CV Weighted-F1)
- **Keep paths consistent** with the dashboard (`models/…`, `reports/…`, `data/processed/…`).
- **Run locally:** `streamlit run app.py`
- **Deploy:** ensure `requirements.txt` (Streamlit, scikit-learn, xgboost, pandas, numpy, joblib), and deployment files (e.g., `Procfile` if Heroku).
    """)

    st.subheader("Limitations & Next Steps")
    st.markdown("""
- **Data imbalance:** minority CEFR levels can depress Macro-F1—consider re-sampling or class-balanced losses.
- **Feature scope:** recommendations are simple and rule-based; could be expanded with curriculum-aligned guidance.
- **Robustness:** consider calibration (`CalibratedClassifierCV`) to improve probability quality if needed.
- **Monitoring:** log live predictions to audit class distribution and drift over time.
    """)

    st.caption("This page documents the rationale, targets, and artefacts behind the CEFR classifier to support assessment (LO3.2, LO5.x).")