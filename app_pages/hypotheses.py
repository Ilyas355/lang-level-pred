# app_pages/hypotheses.py
import streamlit as st
from src.eval import load_test_metrics, select_winner

def render():
    st.header("Hypotheses & Validation")

    st.subheader("Hypothesis 1 — Raw scores cause target leakage")
    st.markdown("""
**Statement.** Features that directly aggregate raw exam scores will trivially encode CEFR and **must not** be used for training.  
**Validation.** EDA showed very high correlation/PPS to `overall_cefr`.  
**Conclusion.** Confirmed. Modelling uses **engineered, non-leaky features** only.
    """)

    st.subheader("Hypothesis 2 — Engineered features can predict CEFR fairly")
    st.markdown("""
**Statement.** Using engineered features only, a classifier can meet targets on a hold-out test set (Acc ≥ 0.75, Macro-F1 ≥ 0.70).  
**Validation.** Tuned LR/RF/XGB via `GridSearchCV (cv=5, f1_macro)`, evaluated on test set.
    """)

    try:
        test_df = load_test_metrics()
        st.dataframe(test_df.round(3))
        winner, row, _ = select_winner(test_df)
        st.success(f"Selected model: **{winner}** — Acc {row['Accuracy']:.3f}, Macro-F1 {row['F1 (macro)']:.3f}.")
    except Exception:
        st.info("Export `reports/test_metrics_tuned.csv` to display results here.")
