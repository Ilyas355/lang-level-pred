# app_pages/data_explorer.py
import streamlit as st
import pandas as pd
from src.paths import CLEAN_DATA
from src.data import load_csv, leakage_columns_present, map_cefr_to_ordinal
from src.viz import class_balance_figure, corr_heatmap_figure, feature_hist_figure

TARGET_COL = "overall_cefr"

def render():
    st.header("Data Explorer")
    st.caption("Answering **BR1** — class balance, correlations, and **target leakage** checks.")

    up = st.file_uploader("Upload CSV (optional). If omitted, loads default cleaned dataset.", type=["csv"])
    if up is not None:
        df = pd.read_csv(up); source = f"Uploaded: `{up.name}`"
    else:
        if CLEAN_DATA.exists():
            df = load_csv(CLEAN_DATA); source = f"Default: `{CLEAN_DATA.as_posix()}`"
        else:
            st.error("No upload and default cleaned dataset not found.")
            return
    st.caption(source)

    with st.expander("Inspect data", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        if st.checkbox("Show head (10 rows)", value=True):
            st.dataframe(df.head(10))

    # Leakage
    leaks = leakage_columns_present(df)
    if leaks:
        st.error(
            "⚠️ **Target leakage risk**: raw exam-like columns present.\n\n"
            f"- Detected: `{', '.join(leaks[:12])}{' ...' if len(leaks) > 12 else ''}`\n"
            "- These directly drive CEFR and **must not** be used in modelling."
        )
    else:
        st.success("✅ No obvious raw exam total columns detected.")

    # Target column
    target = TARGET_COL if TARGET_COL in df.columns else None
    if not target:
        for c in df.columns:
            if "cefr" in c.lower() and df[c].dtype == "object":
                target = c; st.info(f"Using detected target column: **{target}**")
                break

    # Class balance
    st.subheader("CEFR class balance")
    if target:
        fig, caption = class_balance_figure(df, target)
        st.pyplot(fig); st.caption(caption)
    else:
        st.warning("Target column not found; skipping class balance.")

    # Correlations (numeric + CEFR_ordinal)
    st.subheader("Numeric correlation heatmap")
    fig, note, interp = corr_heatmap_figure(df, target)
    if fig: st.pyplot(fig)
    if note: st.caption(note)
    if interp: st.markdown(interp)

    # Per-feature distributions
    st.subheader("Per-feature distributions (by CEFR)")
    num_cols = df.select_dtypes("number").columns.tolist()
    selected = st.multiselect("Numeric features to plot:", options=num_cols, default=[c for c in num_cols if c.endswith("_minus_avg")][:2])
    for feat in selected:
        fig, interp = feature_hist_figure(df, feat, target)
        if fig: st.pyplot(fig)
        if interp: st.markdown(interp)

    with st.expander("Notes"):
        st.markdown("""
- **Target leakage**: raw totals (e.g., `*_score`) must not be used in training.  
- **Fairness**: **Macro-F1** evaluates balance across CEFR levels; **Weighted-F1** reflects majority-class performance.
        """)
