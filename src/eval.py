# src/eval.py
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from src.paths import REPORT_TEST, REPORT_CV
from typing import Tuple

def load_test_metrics() -> pd.DataFrame:
    df = pd.read_csv(REPORT_TEST)
    if "Model" in df.columns: df = df.set_index("Model")
    else: df = pd.read_csv(REPORT_TEST, index_col=0)
    return df

def load_cv_summary() -> pd.DataFrame:
    return pd.read_csv(REPORT_CV)

def select_winner(test_df: pd.DataFrame):
    ranked = test_df.sort_values(["F1 (macro)", "Accuracy"], ascending=False)
    return ranked.index[0], ranked.iloc[0], ranked

def test_bar_chart(test_df: pd.DataFrame):
    df = test_df.copy()
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if len(df) == 1:
        melted = df.T.reset_index(); melted.columns = ["Metric","Score"]
        return (alt.Chart(melted).mark_bar().encode(x="Metric:N", y="Score:Q", tooltip=list(melted.columns)))
    df_reset = df.reset_index()
    id_col = df.index.name if (df.index.name in df_reset.columns) else df_reset.columns[0]
    melted = df_reset.melt(id_vars=id_col, var_name="Metric", value_name="Score")
    return (
        alt.Chart(melted)
        .mark_bar()
        .encode(x=alt.X("Metric:N", title="Metric"), y=alt.Y("Score:Q"), color="Metric:N", column=f"{id_col}:N", tooltip=list(melted.columns))
        .properties(height=250)
    )

def cv_bar_figure(cv_df: pd.DataFrame, test_df: pd.DataFrame):
    df = cv_df.copy()
    if "Model" not in df.columns:
        df = df.reset_index().rename(columns={df.columns[0]: "Model"})
    metric_cols = [c for c in df.columns if c != "Model"]
    for c in metric_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    melted = df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
    order = (melted.groupby("Model")["Score"].mean().sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=melted, x="Score", y="Model", hue="Metric", order=order, palette="viridis", ax=ax, errorbar=None)
    ax.set_title("Cross-Validation Summary (cv=5)"); ax.set_xlabel("Score"); ax.set_ylabel("Model")
    ax.legend(title="Metric", loc="lower right", frameon=False)
    return fig

def quick_interpretation(test_df: pd.DataFrame) -> str:
    ranked = test_df.sort_values(["F1 (macro)", "Accuracy"], ascending=False)
    top = ranked.iloc[0]
    parts = [f"**{ranked.index[0]}** ranks first by Macro-F1 (**{top['F1 (macro)']:.3f}**) and Accuracy (**{top['Accuracy']:.3f}**)."]
    if len(ranked) > 1:
        r2 = ranked.iloc[1]
        parts.append(f"Runner-up: **{ranked.index[1]}** (Macro-F1 **{r2['F1 (macro)']:.3f}**, Accuracy **{r2['Accuracy']:.3f}**).")
    gaps = []
    for m, row in test_df.iterrows():
        try: gaps.append((m, float(row["F1 (weighted)"])-float(row["F1 (macro)"])))
        except: pass
    if gaps:
        worst = max(gaps, key=lambda t:t[1])
        parts.append(f"All models show **Weighted-F1 > Macro-F1** (class imbalance); largest gap: **{worst[0]}** (Δ≈{worst[1]:.03f}).")
    return " ".join(parts)
