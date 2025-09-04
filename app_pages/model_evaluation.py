# app_pages/model_evaluation.py
import streamlit as st
from src.paths import REPORT_TEST, REPORT_CV
from src.constants import TARGET_ACC, TARGET_F1_MACRO
from src.eval import (
    load_test_metrics, load_cv_summary, select_winner,
    test_bar_chart, cv_bar_figure, quick_interpretation
)

def render():
    st.header("Model Evaluation")
    st.caption("Tuned models on the **hold-out test set**. CV (cv=5) shown for generalisation.")

    st.info("**Targets:** Accuracy ≥ **0.75**, Macro-F1 ≥ **0.70** • **Decision rule:** maximise **Macro-F1**, tie-break on **Accuracy**.")

    if not REPORT_TEST.exists():
        st.error("`reports/test_metrics_tuned.csv` missing. Export from your notebook.")
        return

    test_df = load_test_metrics()
    st.subheader("Tuned Models — Test Set")
    st.dataframe(test_df.round(3))
    st.altair_chart(test_bar_chart(test_df), use_container_width=True)

    winner, row, ranked = select_winner(test_df)
    meets = (row["Accuracy"] >= TARGET_ACC) and (row["F1 (macro)"] >= TARGET_F1_MACRO)
    if meets:
        st.success(f"Selected model: **{winner}** — meets targets.")
    else:
        st.warning(
            f"Selected model: **{winner}** — targets not fully met "
            f"(Acc {row['Accuracy']:.3f}/{TARGET_ACC:.2f}; Macro-F1 {row['F1 (macro)']:.3f}/{TARGET_F1_MACRO:.2f})."
        )

    with st.expander("Interpretation", expanded=True):
        st.markdown(quick_interpretation(test_df))

    st.subheader("Cross-Validation Summary (cv=5, out-of-fold)")
    if REPORT_CV.exists():
        cv_df = load_cv_summary()
        st.dataframe(cv_df.round(3))
        if st.checkbox("Show CV bar chart", value=False):
            fig = cv_bar_figure(cv_df, test_df)
            st.pyplot(fig)
    else:
        st.info("Optional: `reports/cv_summary.csv` not found; skip CV plot.")

    st.caption("CV guards against overfitting; final selection is based on the test set.")
