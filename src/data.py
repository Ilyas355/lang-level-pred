# src/data.py
import pandas as pd
from src.constants import RAW_SCORE_COLUMNS, ENGINEERED_OK_SUFFIXES, ENGINEERED_OK_EXACT

def load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)

def map_cefr_to_ordinal(series: pd.Series) -> pd.Series:
    order = {"A1":1, "A2":2, "B1":3, "B2":4, "C1":5, "C2":6}
    return series.map(order)

def leakage_columns_present(df: pd.DataFrame) -> list[str]:
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
