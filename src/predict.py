# src/predict.py
import joblib
import numpy as np
from src.paths import MODEL_PATH, PREPROCESSOR_PATH
from src.constants import ID2CEFR

def load_pipeline():
    return joblib.load(MODEL_PATH)

def load_preprocessor_if_exists():
    try:
        return joblib.load(PREPROCESSOR_PATH)
    except Exception:
        return None

def has_preprocessor(pipe) -> bool:
    try:
        return hasattr(pipe, "named_steps") and ("preprocessor" in pipe.named_steps)
    except Exception:
        return False

def get_label_order(pipe):
    try:
        return list(pipe.classes_)
    except Exception:
        try:
            last = list(pipe.named_steps.keys())[-1]
            return list(pipe.named_steps[last].classes_)
        except Exception:
            return None

def simple_recs(pred_label, proba_by_class, feats):
    overview, actions = [], []
    LOW_CONF, BORDERLINE = 0.55, 0.10
    HIGH_VAR = 0.70
    items = sorted(proba_by_class.items(), key=lambda kv: kv[1], reverse=True) if proba_by_class else []
    if items:
        top_label, top_p = items[0]
        second_label, second_p = items[1] if len(items) > 1 else (None, 0.0)
        if top_p < LOW_CONF:
            overview.append(f"Low confidence (p={top_p:.2f}). Borderline **{top_label}** vs **{second_label}**.")
        elif (top_p - second_p) < BORDERLINE:
            overview.append(f"Borderline **{top_label}** vs **{second_label}** (Δ={(top_p-second_p):.02f}).")
        else:
            overview.append(f"Prediction **{pred_label}** with good confidence (p={top_p:.2f}).")
    else:
        overview.append(f"Prediction **{pred_label}**.")

    diffs = [
        float(feats.get("speaking_minus_avg", 0.0)),
        float(feats.get("reading_minus_avg", 0.0)),
        float(feats.get("listening_minus_avg", 0.0)),
        float(feats.get("writing_minus_avg", 0.0)),
    ]
    var_std = float(np.std(diffs))
    weakest = str(feats.get("weakest_skill", "")).lower()
    SKILL_TIPS = {
        "speaking":  ["Daily 5–10 min monologue; record & self-review.",
                      "Shadow short audio to improve fluency/phonology."],
        "listening": ["Short clips with transcript; pause & summarise.",
                      "Re-listen to catch numbers/dates/names."],
        "reading":   ["Skim for gist, then scan for detail.",
                      "Build topic wordlists; spaced repetition."],
        "writing":   ["One paragraph/day; focus on linkers & accuracy.",
                      "Rewrite after feedback to reduce common errors."],
    }
    if var_std > HIGH_VAR and weakest in SKILL_TIPS:
        overview.append(f"Profile is **uneven**; weakest area: **{weakest}**.")
        actions.extend(SKILL_TIPS[weakest])
    elif var_std > HIGH_VAR:
        overview.append("Profile is **uneven** across skills.")
        actions.append("Prioritise practice in your weakest skill this week.")
    return overview, actions[:3]

# Re-export constants for convenience in pages
ID2CEFR = ID2CEFR
PREPROCESSOR_PATH = PREPROCESSOR_PATH
