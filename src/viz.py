# src/viz.py
import numpy as np
import matplotlib.pyplot as plt
from src.data import map_cefr_to_ordinal
import pandas as pd

def class_balance_figure(df: pd.DataFrame, target: str):
    counts = df[target].value_counts(dropna=False)
    levels_order = ["A1","A2","B1","B2","C1","C2"]
    idx = [l for l in levels_order if l in counts.index] or counts.index.tolist()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(idx, counts[idx].values)
    ax.set_title("CEFR Class Balance"); ax.set_xlabel("CEFR Level"); ax.set_ylabel("Count")
    for i, v in enumerate(counts[idx].values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    total = int(counts.sum()); maj = counts.idxmax(); maj_n = int(counts.max())
    caption = (f"Total rows: **{total}**. Majority class: **{maj}** ({maj_n}/{total}, {maj_n/total:.1%}). "
               "Imbalance can inflate Accuracy/Weighted-F1 while depressing Macro-F1.")
    return fig, caption

def corr_heatmap_figure(df: pd.DataFrame, target: str|None):
    tmp = df.copy()
    note, interp = None, None
    if target and target in tmp.columns:
        ords = map_cefr_to_ordinal(tmp[target])
        if ords.notna().any():
            tmp["CEFR_ordinal"] = ords
            note = "Note: `CEFR_ordinal` is for EDA only (not used as a feature)."
    num_cols = tmp.select_dtypes("number").columns.tolist()
    if not num_cols:
        return None, note, "**Interpretation:** No numeric features available."
    corr = tmp[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6.5,5))
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
    ax.set_title("Numeric Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if "CEFR_ordinal" in corr.columns:
        s = corr["CEFR_ordinal"].drop(labels=["CEFR_ordinal"], errors="ignore").abs().sort_values(ascending=False).head(5)
        interp = "**Interpretation:** Top absolute correlations to CEFR_ordinal → " + "; ".join([f"`{k}`: {v:.2f}" for k,v in s.items()])
    else:
        interp = "**Interpretation:** Numeric correlations shown for reference."
    return fig, note, interp

def feature_hist_figure(df: pd.DataFrame, feature: str, target: str|None):
    if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]):
        return None, f"**Interpretation:** `{feature}` not numeric or not found."
    fig, ax = plt.subplots(figsize=(6,4))
    if target and target in df.columns:
        order = ["A1","A2","B1","B2","C1","C2"]
        classes = [l for l in order if l in df[target].unique()]
        for l in classes:
            vals = df.loc[df[target] == l, feature].dropna()
            ax.hist(vals, bins=20, alpha=0.5, label=l)
        ax.legend(title=target)
        means = df.groupby(target)[feature].mean().sort_values(ascending=False)
        top, bot = means.index[0], means.index[-1]
        interp = (f"**Interpretation:** Higher **{feature}** tends to occur in **{top}**; "
                  f"lower in **{bot}** (class means).")
    else:
        ax.hist(df[feature].dropna(), bins=20, alpha=0.8); interp="**Interpretation:** Univariate distribution shown."
    ax.set_title(f"Distribution: {feature}")
    return fig, interp
