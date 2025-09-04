# app_pages/home.py
import streamlit as st
from src.paths import REPORT_TEST, CLEAN_DATA, MODEL_PATH
from src.constants import TARGET_ACC, TARGET_F1_MACRO
from src.eval import load_test_metrics, select_winner

def render():
    st.title("Quick Project Summary")
    st.caption("CRISP-DM: Business Understanding → Data Understanding → Modelling → Evaluation → Deployment")

    st.markdown("[Open project repository](https://github.com/yourname/yourrepo)")

    st.subheader("Dataset")
    st.markdown("""
- **Human target (EDA):** `overall_cefr` (A1–C2)  
- **Training target:** `cefr_encoded` (0–5)  
- **Leakage control:** Raw exam totals excluded from modelling; we train on **engineered features** only.
    """)
    c1, c2 = st.columns(2)
    c1.caption(f"Cleaned dataset present? **{'Yes' if CLEAN_DATA.exists() else 'No'}**")
    c2.caption(f"Saved pipeline present? **{'Yes' if MODEL_PATH.exists() else 'No'}**")

    st.subheader("Business Requirements")
    st.markdown("""
1. **The client has asked us to** analyse the learner dataset to surface patterns driving CEFR outcomes, **quantify class imbalance**, and **detect target leakage**. From these findings, define **engineered, non-leaky features**.  
2. **The client has asked us to** build and operationalise a **fair, interpretable CEFR classifier** using the engineered features, meeting targets (**Accuracy ≥ 0.75**, **Macro-F1 ≥ 0.70**), and expose a Streamlit **Predict CEFR** page with predicted label + class probabilities.
    """)

    st.markdown("> **Targets** (test set): Accuracy ≥ **0.75**, Macro-F1 ≥ **0.70** · **Decision rule:** maximise **Macro-F1**, tie-break on **Accuracy**")

    st.subheader("Final Model & Performance (Test Set)")
    if not REPORT_TEST.exists():
        st.warning("`reports/test_metrics_tuned.csv` not found. See Model Evaluation page.")
        return

    test_df = load_test_metrics()
    st.dataframe(test_df.round(3))
    winner, row, ranked = select_winner(test_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy (test)", f"{row['Accuracy']:.3f}")
    c2.metric("Macro-F1 (test)", f"{row['F1 (macro)']:.3f}")
    c3.metric("Weighted-F1 (test)", f"{row['F1 (weighted)']:.3f}")

    meets = (row["Accuracy"] >= TARGET_ACC) and (row["F1 (macro)"] >= TARGET_F1_MACRO)
    if meets:
        st.success(f"✅ Meets targets. Selected model: **{winner}**.")
    else:
        st.warning(
            f"⚠️ Targets not fully met (Acc {row['Accuracy']:.3f} vs {TARGET_ACC:.2f}; "
            f"Macro-F1 {row['F1 (macro)']:.3f} vs {TARGET_F1_MACRO:.2f}). "
            f"Selected model: **{winner}** as best overall."
        )

    st.caption("See **Model Evaluation** for full comparison, confusion matrix, and cv=5 summary.")
