# app_pages/predict_cefr.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = "models/final_logistic_regression_pipeline.pkl"  # adjust if different

CEFR2ID = {"A1":0,"A2":1,"B1":2,"B2":3,"C1":4,"C2":5}
ID2CEFR = {v:k for k,v in CEFR2ID.items()}

def simple_recs(pred_label, proba_by_class, feats):
    overview, actions = [], []
    LOW_CONF, BORDERLINE = 0.55, 0.10
    IMBAL_TILT, HIGH_VAR = 0.50, 0.70

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

    prb = float(feats.get("productive_dominant", 0.0))
    if prb > IMBAL_TILT:
        overview.append("Stronger in **productive** skills than receptive.")
        actions.append("Increase input: daily reading/listening with short summaries.")
    elif prb < -IMBAL_TILT:
        overview.append("Stronger in **receptive** skills than productive.")
        actions.append("Add output: timed speaking prompts and short writing tasks.")
    else:
        overview.append("Overall skill balance is reasonably even.")

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
                      "Build topic wordlists; review with spaced repetition."],
        "writing":   ["One paragraph/day; focus on linkers and accuracy.",
                      "Rewrite after feedback to reduce common errors."],
    }
    if var_std > HIGH_VAR and weakest in SKILL_TIPS:
        overview.append(f"Profile is **uneven**; weakest area: **{weakest}**.")
        actions.extend(SKILL_TIPS[weakest])
    elif var_std > HIGH_VAR:
        overview.append("Profile is **uneven** across skills.")
        actions.append("Prioritise practice in your weakest skill this week.")

    return overview, actions[:3]

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

def get_classes_from_pipeline(pipe):
    try:
        return list(pipe.classes_)
    except Exception:
        try:
            last = list(pipe.named_steps.keys())[-1]
            return list(pipe.named_steps[last].classes_)
        except Exception:
            return None

def render():
    st.header("Predict CEFR Level")
    st.caption("Final **Logistic Regression** pipeline on **engineered (leakage-free)** features.")
    if not Path(MODEL_PATH).exists():
        st.error(f"Model not found at `{MODEL_PATH}`.")
        return

    pipe = load_pipeline()
    raw_classes = get_classes_from_pipeline(pipe)
    labels = [ID2CEFR.get(int(c), str(c)) for c in raw_classes] if raw_classes else None

    # Exact engineered feature schema (matches training)
    skills = ["speaking","reading","listening","writing"]
    levels = ["Beginner","Intermediate","Advanced"]
    profiles = ["Balanced","Uneven Development"]
    feature_schema = {
        "strongest_skill":        ("category", skills),
        "weakest_skill":          ("category", skills),
        "second_weakest_skill":   ("category", skills),
        "strength_weakness_gap":  ("number", 0.0),
        "learning_profile":       ("category", profiles),
        "speaking_minus_avg":     ("number", 0.0),
        "reading_minus_avg":      ("number", 0.0),
        "listening_minus_avg":    ("number", 0.0),
        "writing_minus_avg":      ("number", 0.0),
        "productive_dominant":    ("number", 0.0),
        "speaking_level":         ("category", levels),
        "reading_level":          ("category", levels),
        "listening_level":        ("category", levels),
        "writing_level":          ("category", levels),
    }

    st.subheader("Enter engineered features")
    cols = st.columns(2)
    X_live = {}
    for i, (feat, (ftype, opts)) in enumerate(feature_schema.items()):
        with cols[i % 2]:
            if ftype == "number":
                X_live[feat] = st.number_input(feat, value=float(opts))
            elif ftype == "category":
                X_live[feat] = st.selectbox(feat, options=opts)
            else:
                st.warning(f"Unknown type for {feat}: {ftype}")

    st.markdown("---")
    st.subheader("(Optional) Rule-based CEFR for QA")
    rule_cefr = st.selectbox("Compare with a known/estimated CEFR (optional):",
                              options=["(none)","A1","A2","B1","B2","C1","C2"], index=0)

    if st.button("Predict CEFR"):
        X_df = pd.DataFrame([X_live])

        raw_pred = pipe.predict(X_df)[0]
        pred = ID2CEFR.get(int(raw_pred), str(raw_pred))
        st.success(f"**Predicted CEFR:** {pred}")

        proba_dict = {}
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_df)[0]
            if labels:
                proba_dict = {lab: float(np.round(p, 3)) for lab, p in zip(labels, proba)}
                st.write("Class probabilities:")
                st.write(proba_dict)

        # QA flags
        flags = []
        if proba_dict:
            top_p = max(proba_dict.values())
            if top_p < 0.55:
                flags.append(f"Low confidence prediction (max p = {top_p:.2f}).")
        if rule_cefr != "(none)" and rule_cefr != pred:
            flags.append(f"Model disagrees with rule-based CEFR (**{rule_cefr}** vs **{pred}**).")

        if flags:
            st.warning("⚠️ Review flags:\n\n" + "\n".join(f"- {f}" for f in flags))
        else:
            st.info("No QA flags for this prediction.")

        # Recommendations
        st.subheader("Personalised recommendations")
        overview, actions = simple_recs(pred, proba_dict, X_live)
        for line in overview:
            st.write(f"• {line}")
        if actions:
            st.markdown("**Suggested next steps:**")
            for tip in actions:
                st.write(f"- {tip}")

        with st.expander("Show input row"):
            st.dataframe(X_df)

    with st.expander("Notes"):
        st.markdown("""
- Inputs must match the **engineered feature schema** used during training (no raw totals to avoid **target leakage**).
- The optional rule-based CEFR lets you triangulate the model; mismatches or low confidence are flagged.
- Recommendations are **rule-based and lightweight**, based on balance and dispersion across skills.
""")