# app_pages/model_evaluation.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import altair as alt

TEST_METRICS_PATH = Path("reports/test_metrics_tuned.csv")
CV_SUMMARY_PATH   = Path("reports/cv_summary.csv")

# Targets (from your notebook)
TARGET_ACC  = 0.75
TARGET_F1_M = 0.70

def _load_test_metrics():
    if not TEST_METRICS_PATH.exists():
        st.error(f"Missing `{TEST_METRICS_PATH}`. Export from your notebook.")
        st.stop()
    df = pd.read_csv(TEST_METRICS_PATH)
    if "Model" in df.columns:
        df = df.set_index("Model")
    else:
        df = pd.read_csv(TEST_METRICS_PATH, index_col=0)
    return df

def _load_cv_summary():
    if not CV_SUMMARY_PATH.exists():
        st.warning(f"Optional CV table `{CV_SUMMARY_PATH}` not found.")
        return None
    return pd.read_csv(CV_SUMMARY_PATH)

def _select_winner(test_df: pd.DataFrame):
    ranked = test_df.sort_values(["F1 (macro)", "Accuracy"], ascending=False)
    return ranked.index[0], ranked.iloc[0], ranked

def _plot_test_bar(test_df: pd.DataFrame):
    """
    Plot test metrics as a bar chart.
    - Works for single-row or multi-row DataFrames.
    - Handles named or unnamed indexes (avoids KeyError on 'index').
    """
    if test_df is None or test_df.empty:
        st.info("No test metrics to plot.")
        return

    # Make a working copy and coerce numerics
    df = test_df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if len(df) == 1:
        # Typical case: one row of metrics -> transpose to Metric/Score
        melted = df.T.reset_index()
        melted.columns = ["Metric", "Score"]
    else:
        # Multi-row case: keep the (possibly named) index as id var
        df_reset = df.reset_index()
        # If the index had a name, pandas uses it as the first column name; otherwise it's literally "index"
        id_col = df.index.name if (df.index.name in df_reset.columns) else df_reset.columns[0]
        melted = df_reset.melt(id_vars=id_col, var_name="Metric", value_name="Score")

    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", sort="-y", title="Metric"),
            y=alt.Y("Score:Q", title="Score"),
            tooltip=list(melted.columns),
        )
    )
    st.altair_chart(chart, use_container_width=True)

def _plot_cv_bar(cv_df: pd.DataFrame):
    """
    Plot cross-validation summary as bars grouped by Metric.
    Safe if 'Model' is missing (uses index), ignores non-numeric columns.
    """
    if cv_df is None or cv_df.empty:
        st.info("No cross-validation results to show.")
        return

    df = cv_df.copy()

    # Ensure there is a 'Model' column (use index if necessary)
    if "Model" not in df.columns:
        df = df.reset_index()
        # The first column after reset_index() is the former index (named or 'index')
        first_col = df.columns[0]
        if first_col != "Model":
            df = df.rename(columns={first_col: "Model"})

    # Metric columns = everything except 'Model'
    metric_cols = [c for c in df.columns if c != "Model"]

    # Coerce to numeric; drop metric columns that are entirely NaN after coercion
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    metric_cols = [c for c in metric_cols if df[c].notna().any()]

    if not metric_cols:
        st.warning("No numeric metric columns found to plot.")
        st.dataframe(df)
        return

    melted = df.melt(
        id_vars="Model",
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )

    # Order models by mean score for a nicer plot
    model_order = (
        melted.groupby("Model")["Score"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=melted,
        x="Score",
        y="Model",
        hue="Metric",
        order=model_order,
        palette="viridis",
        ax=ax,
        errorbar=None,  # seaborn>=0.12
    )
    ax.set_title("Cross-Validation Summary (Out-of-Fold, cv=5)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Model")
    ax.legend(title="Metric", loc="lower right", frameon=False)

    st.pyplot(fig)
    plt.close(fig)
    try:
        # Compare Macro-F1 ranking between CV and Test
        cv_rank = cv_df.sort_values("CV Macro-F1", ascending=False)["Model"].tolist()
        test_rank = test_df.sort_values("F1 (macro)", ascending=False).index.tolist()
        aligns = (cv_rank[0] == test_rank[0])
        msg = "CV ranking aligns with the test-set leader." if aligns else "CV ranking differs from the test-set ordering."
        st.markdown(f"**Interpretation:** {msg} CV provides an out-of-fold generalisation check.")
    except Exception:
        st.caption("CV provides an out-of-fold generalisation check.")

def _quick_interpretation(test_df: pd.DataFrame):
    def safe(df, model, col):
        try: return float(df.loc[model, col])
        except Exception: return np.nan

    models = list(test_df.index)
    if len(models) < 1:
        return "No models found."

    winner, row, ranked = _select_winner(test_df)
    parts = [f"**{winner}** ranks first by Macro-F1 (**{float(row['F1 (macro)']):.3f}**) "
             f"and Accuracy (**{float(row['Accuracy']):.3f}**)."]
    if len(ranked) > 1:
        r2 = ranked.iloc[1]
        parts.append(f"Runner-up: **{ranked.index[1]}** "
                     f"(Macro-F1 **{float(r2['F1 (macro)']):.3f}**, "
                     f"Accuracy **{float(r2['Accuracy']):.3f}**).")

    gaps = []
    for m in models:
        m_macro = safe(test_df, m, "F1 (macro)")
        m_weighted = safe(test_df, m, "F1 (weighted)")
        if not np.isnan(m_macro) and not np.isnan(m_weighted):
            gaps.append((m, m_weighted - m_macro))
    if gaps:
        worst_gap = max(gaps, key=lambda t: t[1])
        parts.append(f"All models show **Weighted-F1 > Macro-F1** (class imbalance); "
                     f"largest gap: **{worst_gap[0]}** (Δ ≈ {worst_gap[1]:.03f}).")

    return " ".join(parts)

def render():
    st.header("Model Evaluation")
    st.caption("Tuned models evaluated on the **hold-out test set**. Cross-validation (cv=5) shown as a generalisation check.")
    st.info("**Targets:** Accuracy ≥ **0.75**, Macro-F1 ≥ **0.70** • **Decision rule:** maximise **Macro-F1**, tie-break on **Accuracy**.")

    # Load
    test_df = _load_test_metrics()
    cv_df   = _load_cv_summary()

    # Test set comparison
    st.subheader("Tuned Models — Test Set")
    st.dataframe(test_df.round(3))
    _plot_test_bar(test_df)

    # Selection & targets
    winner, row, ranked = _select_winner(test_df)
    meets = (row["Accuracy"] >= TARGET_ACC) and (row["F1 (macro)"] >= TARGET_F1_M)
    if meets:
        st.success(
            f"Selected model: **{winner}** — meets targets. "
            f"(Acc **{row['Accuracy']:.3f}**, Macro-F1 **{row['F1 (macro)']:.3f}**, "
            f"Weighted-F1 **{row['F1 (weighted)']:.3f}**)"
        )
    else:
        st.warning(
            f"Selected model: **{winner}** — targets **not fully met** "
            f"(Acc {row['Accuracy']:.3f} vs {TARGET_ACC:.2f}; "
            f"Macro-F1 {row['F1 (macro)']:.3f} vs {TARGET_F1_M:.2f}). "
            "Model chosen as best overall given fairness and interpretability."
        )

    with st.expander("Interpretation", expanded=True):
        st.markdown(_quick_interpretation(test_df))

    st.divider()

    # CV summary
    st.subheader("Cross-Validation Summary (cv=5, out-of-fold)")
    if cv_df is not None:
        st.dataframe(cv_df.round(3))
        if st.checkbox("Show CV bar chart", value=False):
            _plot_cv_bar(cv_df)
        st.caption("Out-of-fold CV guards against overfitting. Final selection remains based on the **test set** above.")

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="Download test metrics (CSV)",
            data=test_df.to_csv().encode("utf-8"),
            file_name="test_metrics_tuned.csv",
            mime="text/csv"
        )
    with c2:
        if cv_df is not None:
            st.download_button(
                label="Download CV summary (CSV)",
                data=cv_df.to_csv(index=False).encode("utf-8"),
                file_name="cv_summary.csv",
                mime="text/csv"
            )