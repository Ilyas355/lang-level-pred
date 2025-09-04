# app_pages/home.py
import streamlit as st
from pathlib import Path
import pandas as pd

TEST_METRICS_PATH = Path("reports/test_metrics_tuned.csv")
MODEL_PATH = Path("models/final_logistic_regression_pipeline.pkl")  # adjust if different

# Performance targets (from your notebook)
TARGET_ACC  = 0.75
TARGET_F1_M = 0.70

def _load_test_metrics():
    if not TEST_METRICS_PATH.exists():
        return None
    df = pd.read_csv(TEST_METRICS_PATH)
    if "Model" in df.columns:
        df = df.set_index("Model")
    else:
        df = pd.read_csv(TEST_METRICS_PATH, index_col=0)
    return df

def _select_winner(test_df: pd.DataFrame):
    ranked = test_df.sort_values(["F1 (macro)", "Accuracy"], ascending=False)
    return ranked.index[0], ranked.iloc[0], ranked

def render():
    st.title("Quick Project Summary")
    st.caption("CRISP-DM: Business Understanding → Data Understanding → Modelling → Evaluation → Deployment")

    # (Optional) repo link
    REPO_URL = "https://github.com/yourname/yourrepo"  # <- replace
    st.markdown(f"[Open project repository]({REPO_URL})")

    st.divider()

    with st.expander("Project Terms & Jargon", expanded=True):
        st.markdown("""
- **CEFR**: A1–C2 language proficiency scale.
- **Target leakage**: Inputs that directly encode the target (e.g., raw totals). We **avoid** this by training on **engineered features** only.
- **Macro-F1**: Class-balanced F1 (fairness across CEFR levels).
- **Weighted-F1**: F1 weighted by class frequency.
- **cv=5**: Out-of-fold estimate of generalisation.
        """)

    st.subheader("Dataset")
    st.markdown("""
- **Human target (EDA):** `overall_cefr` (A1–C2)  
- **Training target:** `cefr_encoded` (0–5)  
- **Local paths:**  
  - Raw: `data/raw/`  
  - Cleaned: `data/clean/cleaned_lang_proficiency_results.csv`  
- **Leakage control:** Raw exam totals are **not** used in modelling; only **engineered, leakage-free features** are used.
    """)

    cleaned_path = Path("data/clean/cleaned_lang_proficiency_results.csv")
    c1, c2 = st.columns(2)
    c1.caption(f"Cleaned dataset present? **{'Yes' if cleaned_path.exists() else 'No'}**")
    c2.caption(f"Saved pipeline present? **{'Yes' if MODEL_PATH.exists() else 'No'}**")

    st.divider()

    st.subheader("Business Requirements")
    st.markdown("""
1. **The client has asked us to** analyse the learner dataset to surface patterns driving CEFR outcomes, **quantify class imbalance**, and **detect target leakage** from raw exam scores. From these findings, define **engineered, non-leaky features**.

2. **The client has asked us to** build and operationalise a **fair, interpretable CEFR classifier** using the engineered features, meeting targets (**Accuracy ≥ 0.75**, **Macro-F1 ≥ 0.70**), and expose a Streamlit **Predict CEFR** page with predicted label + class probabilities.
    """)

    with st.expander("Performance Targets & Decision Rule", expanded=True):
        st.markdown("""
- **Targets (test set):** Accuracy ≥ **0.75**, Macro-F1 ≥ **0.70**  
- **Decision rule:** maximise **Macro-F1**; use **Accuracy** as tie-breaker
        """)

    st.divider()

    st.subheader("Final Model & Performance (Test Set)")
    test_df = _load_test_metrics()
    if test_df is None:
        st.warning("`reports/test_metrics_tuned.csv` not found. See Model Evaluation page for details.")
        st.stop()

    winner, row, ranked = _select_winner(test_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy (test)", f"{row['Accuracy']:.3f}")
    c2.metric("Macro-F1 (test)", f"{row['F1 (macro)']:.3f}")
    c3.metric("Weighted-F1 (test)", f"{row['F1 (weighted)']:.3f}")

    meets = (row["Accuracy"] >= TARGET_ACC) and (row["F1 (macro)"] >= TARGET_F1_M)
    if meets:
        st.success(f"✅ **Meets targets**. Selected model: **{winner}**.")
    else:
        st.warning(
            f"⚠️ **Targets not fully met** "
            f"(Acc {row['Accuracy']:.3f} vs {TARGET_ACC:.2f}; "
            f"Macro-F1 {row['F1 (macro)']:.3f} vs {TARGET_F1_M:.2f}). "
            f"Selected model: **{winner}** (best overall & interpretable)."
        )

    st.caption("See **Model Evaluation** for full comparison, confusion matrix, and cv=5 summary.")

    st.divider()
    st.subheader("How to use this dashboard")
    st.markdown("""
- **Data Explorer**: inspect class balance, correlations/PPS, and leakage risks.  
- **Model Evaluation**: tuned test-set comparison, confusion matrix, **cv=5** summary.  
- **Predict CEFR**: enter engineered features → CEFR + class probabilities, QA flags & brief recommendations.  
- **About & Methods**: business case, pipeline steps, artefact paths.
    """)