# app_pages/about.py
import streamlit as st
from src.paths import CLEAN_DATA, MODEL_PATH, REPORT_TEST, REPORT_CV
from src.constants import TARGET_ACC, TARGET_F1_MACRO

def _badge(path) -> str: return "✅" if path.exists() else "❌"

def render():
    st.header("About & Methods")

    st.subheader("Business Requirements")
    st.markdown("""
1. **The client has asked us to** analyse the learner dataset to surface patterns driving CEFR outcomes, **quantify class imbalance**, and **detect target leakage**. From these findings, define **engineered, non-leaky features**.  
2. **The client has asked us to** build and operationalise a **fair, interpretable CEFR classifier** using the engineered features, meeting targets (**Accuracy ≥ 0.75**, **Macro-F1 ≥ 0.70**), and expose a Streamlit **Predict CEFR** page with predicted label + class probabilities.
    """)

    st.subheader("ML Business Case")
    st.markdown(f"""
- **Aim:** Predict CEFR (A1–C2) from engineered features to support fair placement.  
- **Learning:** Compare LR / RF / XGB; tune via `GridSearchCV` (cv=5, **f1_macro**), evaluate on a **hold-out test set**.  
- **Targets:** Acc ≥ **{TARGET_ACC:.2f}**, Macro-F1 ≥ **{TARGET_F1_MACRO:.2f}**.  
- **Decision rule:** maximise **Macro-F1**; tie-break on **Accuracy**.  
- **Leakage control:** No raw totals; only **engineered** features (strengths/weaknesses, `*_minus_avg`, gap, profile, per-skill bins, productive/receptive balance).
    """)

    st.subheader("Artefacts & Paths")
    st.markdown(f"""
- Cleaned data: `{CLEAN_DATA.as_posix()}` {_badge(CLEAN_DATA)}  
- Final pipeline: `{MODEL_PATH.as_posix()}` {_badge(MODEL_PATH)}  
- Test metrics: `{REPORT_TEST.as_posix()}` {_badge(REPORT_TEST)}  
- CV summary: `{REPORT_CV.as_posix()}` {_badge(REPORT_CV)}  
    """)

    st.subheader("Reproduce & Deploy")
    st.markdown("""
- Export from notebooks: `reports/test_metrics_tuned.csv` and `reports/cv_summary.csv`.  
- Run locally: `streamlit run app.py`  
- Deploy (e.g., Heroku): ensure `requirements.txt`, `Procfile`, `setup.sh`, `runtime.txt`.
    """)
