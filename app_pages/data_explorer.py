# app_pages/data_explorer.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_DATA_PATH = Path("data/clean/cleaned_lang_proficiency_results.csv")
TARGET_COL = "overall_cefr"  # human target for EDA (A1–C2)

RAW_SCORE_COLUMNS = {
    "speaking_score","reading_score","listening_score","writing_score","overall_score"
}

# Whitelist engineered features so they aren't flagged as leakage
ENGINEERED_OK_SUFFIXES = ("_minus_avg", "_level")
ENGINEERED_OK_EXACT = {
    "productive_dominant","strongest_skill","weakest_skill","second_weakest_skill",
    "strength_weakness_gap","learning_profile"
}

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def map_cefr_to_ordinal(series: pd.Series) -> pd.Series:
    order = {"A1":1, "A2":2, "B1":3, "B2":4, "C1":5, "C2":6}
    return series.map(order)

def leakage_columns_present(df: pd.DataFrame) -> list:
    present = [c for c in RAW_SCORE_COLUMNS if c in df.columns]
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ENGINEERED_OK_EXACT or lc.endswith(ENGINEERED_OK_SUFFIXES):
            continue
        if any(k in lc for k in ["speaking","reading","listening","writing"]):
            if lc.endswith("_score") or lc in {"speaking","reading","listening","writing"}:
                if c not in present:
                    present.append(c)
    return present

def plot_class_balance(df: pd.DataFrame, target: str):
    counts = df[target].value_counts(dropna=False)
    order = ["A1","A2","B1","B2","C1","C2"]
    idx = [c for c in order if c in counts.index] or counts.index.tolist()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(idx, counts[idx].values)
    ax.set_title("CEFR Class Balance"); ax.set_xlabel("CEFR Level"); ax.set_ylabel("Count")
    for i, v in enumerate(counts[idx].values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    st.pyplot(fig)
    st.caption(f"Total rows: **{int(counts.sum())}**. Imbalance can depress **Macro-F1** vs **Weighted-F1**.")

def plot_numeric_correlations(df: pd.DataFrame, target: str):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in df.columns:
        cefr_ord = map_cefr_to_ordinal(df[target])
        if cefr_ord.notna().any():
            df = df.copy()
            df["CEFR_ordinal"] = cefr_ord
            num_cols = list(dict.fromkeys(num_cols + ["CEFR_ordinal"]))
    if not num_cols:
        st.info("No numeric columns found to compute correlations.")
    else:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6.5,5))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
        ax.set_title("Numeric Correlation Heatmap")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        st.caption("Note: `CEFR_ordinal` is for EDA only (not used as a training feature).")

def plot_feature_hist(df: pd.DataFrame, feature: str, target: str | None = None):
    fig, ax = plt.subplots(figsize=(6,4))
    if target and target in df.columns:
        levels = ["A1","A2","B1","B2","C1","C2"]
        available = [l for l in levels if l in df[target].unique().tolist()]
        for l in available:
            vals = df.loc[df[target] == l, feature].dropna()
            ax.hist(vals, bins=20, alpha=0.5, label=l)
        ax.legend(title=target)
    else:
        ax.hist(df[feature].dropna(), bins=20, alpha=0.8)
    ax.set_title(f"Distribution: {feature}")
    st.pyplot(fig)

def render():
    st.header("Data Explorer")
    st.caption("Answering **BR1** — data understanding, class balance, correlations, and **target leakage** check.")

    st.subheader("Load data")
    up = st.file_uploader("Upload a CSV (optional). If omitted, the app loads the default processed dataset.", type=["csv"])
    if up is not None:
        df = pd.read_csv(up); source = f"Uploaded: `{up.name}`"
    else:
        if DEFAULT_DATA_PATH.exists():
            df = load_csv(DEFAULT_DATA_PATH); source = f"Default: `{DEFAULT_DATA_PATH.as_posix()}`"
        else:
            st.error("No upload and default dataset not found. Please upload a CSV.")
            return
    st.caption(source)

    with st.expander("🔎 Inspect data", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        if st.checkbox("Show first 10 rows", value=True):
            st.dataframe(df.head(10))
        if st.checkbox("Show column types", value=False):
            st.write(pd.DataFrame({"dtype": df.dtypes.astype(str)}))
        if st.checkbox("Show descriptive stats (numeric)", value=False):
            st.write(df.describe().T)

    # Leakage check
    leaks = leakage_columns_present(df)
    if leaks:
        st.error(
            "⚠️ **Target leakage risk detected**: raw exam-like columns present.\n\n"
            f"- Detected: `{', '.join(leaks[:12])}{' ...' if len(leaks) > 12 else ''}`\n"
            "- These directly drive CEFR and **must not** be used in modelling. "
            "The modelling pipeline uses **engineered, leakage-free features** only."
        )
    else:
        st.success("✅ No obvious raw exam total columns detected (good for leakage avoidance).")

    # Target column
    target = TARGET_COL if TARGET_COL in df.columns else None
    if not target:
        candidates = [c for c in df.columns if "cefr" in c.lower() and df[c].dtype == "object"]
        if candidates:
            target = candidates[0]
            st.info(f"Using detected target column: **{target}**")
    if "cefr_encoded" in df.columns:
        st.caption("Training label `cefr_encoded` (0–5) also present — used in notebooks only.")

    # Class balance
    st.subheader("CEFR class balance")
    if target:
        try:
            plot_class_balance(df, target)
        except Exception as e:
            st.warning(f"Could not plot class balance: {e}")
    else:
        st.warning("Target column not found; skipping class balance plot.")

    # Correlations
    st.subheader("Numeric correlation heatmap")
    if st.checkbox("Show correlation heatmap (includes CEFR_ordinal for EDA)", value=True):
        try:
            plot_numeric_correlations(df, target or "")
        except Exception as e:
            st.warning(f"Could not compute correlations: {e}")

    # Per-feature distributions
    st.subheader("Per-feature distributions")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_feats = [c for c in num_cols if c.endswith("_minus_avg")] or num_cols[:3]
    selected = st.multiselect(
        "Select numeric features to plot (overlayed by CEFR class if available):",
        options=num_cols, default=default_feats
    )
    if selected:
        for feat in selected:
            try:
                plot_feature_hist(df, feat, target)
            except Exception as e:
                st.warning(f"Could not plot {feat}: {e}")

    with st.expander("Notes"):
        st.markdown("""
- **Target leakage**: raw totals (e.g., `*_score`) must not be used in training. We train on **engineered features** (e.g., strengths/weaknesses, `*_minus_avg`, gaps, profiles).
- **Fairness**: **Macro-F1** evaluates balance across CEFR levels; **Weighted-F1** reflects majority-class performance.
- **Next**: See **Model Evaluation** for tuned test metrics, confusion matrix, and cv=5 summary.
        """)