# app_pages/predict_cefr.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Adjust these if your filenames differ
MODEL_PATH = "models/final_logistic_regression_pipeline.pkl"   # or "models/logreg_tuned_pipeline.joblib"
PREPROCESSOR_PATH = "models/preprocessing_pipeline.pkl"        # <- make sure you saved this from the notebook

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

@st.cache_resource
def load_preprocessor_if_exists():
    p = Path(PREPROCESSOR_PATH)
    return joblib.load(p) if p.exists() else None

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

def render():
    st.header("Predict CEFR Level")
    st.caption("Works with: (a) a unified pipeline (with preprocessor) or (b) a separate preprocessor + model.")

    if not Path(MODEL_PATH).exists():
        st.error(f"Model not found at `{MODEL_PATH}`.")
        return

    pipe = load_pipeline()
    separate_preproc = load_preprocessor_if_exists() if not has_preprocessor(pipe) else None

    # === Schema + user-friendly labels & tooltips ===
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

    LABELS = {
        "strongest_skill":        "strongest_skill (pick your strongest of the four skills)",
        "weakest_skill":          "weakest_skill (pick your weakest skill)",
        "second_weakest_skill":   "second_weakest_skill (next weakest skill)",
        "strength_weakness_gap":  "strength_weakness_gap (max(skill_minus_avg) − min(skill_minus_avg))",
        "learning_profile":       "learning_profile (Balanced vs Uneven Development)",

        "speaking_minus_avg":     "speaking_minus_avg (speaking − mean of all four)",
        "reading_minus_avg":      "reading_minus_avg (reading − mean of all four)",
        "listening_minus_avg":    "listening_minus_avg (listening − mean of all four)",
        "writing_minus_avg":      "writing_minus_avg (writing − mean of all four)",

        "productive_dominant":    "productive_dominant ((speaking_minus_avg + writing_minus_avg) − (reading_minus_avg + listening_minus_avg))",

        "speaking_level":         "speaking_level (0–33=Beginner, 34–66=Intermediate, 67–100=Advanced)",
        "reading_level":          "reading_level (0–33=Beginner, 34–66=Intermediate, 67–100=Advanced)",
        "listening_level":        "listening_level (0–33=Beginner, 34–66=Intermediate, 67–100=Advanced)",
        "writing_level":          "writing_level (0–33=Beginner, 34–66=Intermediate, 67–100=Advanced)",
    }

    HELPS = {
        "strength_weakness_gap":  "Difference between your strongest and weakest skill deviations.",
        "learning_profile":       "Balanced = skills near the mean; Uneven = larger spread across skills.",
        "productive_dominant":    "Positive → productive (speaking/writing) stronger; negative → receptive (reading/listening) stronger; near 0 → balanced.",
        "speaking_minus_avg":     "How far speaking sits above/below your own average across the four skills.",
        "reading_minus_avg":      "How far reading sits above/below your own average across the four skills.",
        "listening_minus_avg":    "How far listening sits above/below your own average across the four skills.",
        "writing_minus_avg":      "How far writing sits above/below your own average across the four skills.",
        "speaking_level":         "Guidance: 0–33 Beginner, 34–66 Intermediate, 67–100 Advanced.",
        "reading_level":          "Guidance: 0–33 Beginner, 34–66 Intermediate, 67–100 Advanced.",
        "listening_level":        "Guidance: 0–33 Beginner, 34–66 Intermediate, 67–100 Advanced.",
        "writing_level":          "Guidance: 0–33 Beginner, 34–66 Intermediate, 67–100 Advanced.",
    }

    st.subheader("Enter engineered features")
    st.caption("Tip: If unsure, leave numeric fields at 0.0 (balanced). Use the labels for quick formulas/ranges.")
    cols = st.columns(2)
    X_live = {}
    for i, (feat, (ftype, opts)) in enumerate(feature_schema.items()):
        label = LABELS.get(feat, feat)
        help_txt = HELPS.get(feat, None)
        with cols[i % 2]:
            if ftype == "number":
                X_live[feat] = float(st.number_input(label, value=float(opts), step=0.1, help=help_txt))
            elif ftype == "category":
                X_live[feat] = st.selectbox(label, options=opts, help=help_txt)
            else:
                st.warning(f"Unknown type for {feat}: {ftype}")

    st.markdown("---")
    st.subheader("(Optional) Rule-based CEFR for QA")
    rule_cefr = st.selectbox("Compare with a known/estimated CEFR (optional):",
                              options=["(none)","A1","A2","B1","B2","C1","C2"], index=0)

    if st.button("Predict CEFR"):
        X_df = pd.DataFrame([X_live])

        # ---------- Transform & predict ----------
        try:
            if has_preprocessor(pipe):
                # Unified pipeline: pass raw engineered DataFrame
                yhat = pipe.predict(X_df)[0]
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(X_df)[0]
                else:
                    proba = None
                label_order = get_label_order(pipe)
            else:
                if separate_preproc is None:
                    st.error(
                        "Your saved model does not include a preprocessor and "
                        f"`{PREPROCESSOR_PATH}` was not found.\n\n"
                        "Fix: in your notebook, save the fitted preprocessor:\n"
                        f"`joblib.dump(preprocessor, '{PREPROCESSOR_PATH}')`"
                    )
                    return
                # Transform with the same fitted preprocessor used in training
                X_enc = separate_preproc.transform(X_df)
                yhat = pipe.predict(X_enc)[0]
                if hasattr(pipe, "predict_proba"):
                    proba = pipe.predict_proba(X_enc)[0]
                else:
                    proba = None
                label_order = get_label_order(pipe)

        except ValueError as e:
            st.error(
                "Prediction failed while converting inputs. This usually means the model expects **numeric** "
                "features (already encoded), but the UI is sending raw categorical strings.\n\n"
                "Ensure either:\n"
                "1) Your model file contains a `preprocessor` step; **or**\n"
                f"2) You saved and placed the fitted preprocessor at `{PREPROCESSOR_PATH}`.\n\n"
                f"Details: {e}"
            )
            return

        pred_label = ID2CEFR.get(int(yhat), str(yhat))
        st.success(f"**Predicted CEFR:** {pred_label}")

        # Map probabilities to CEFR labels (if available)
        proba_dict = {}
        if proba is not None and label_order is not None:
            ce_labels = [ID2CEFR.get(int(c), str(c)) for c in label_order]
            proba_dict = {lab: float(np.round(p, 3)) for lab, p in zip(ce_labels, proba)}
            st.write("Class probabilities:")
            st.write(proba_dict)

        # QA flags
        flags = []
        if proba_dict:
            top_p = max(proba_dict.values())
            if top_p < 0.55:
                flags.append(f"Low confidence prediction (max p = {top_p:.2f}).")
        if rule_cefr != "(none)" and rule_cefr != pred_label:
            flags.append(f"Model disagrees with rule-based CEFR (**{rule_cefr}** vs **{pred_label}**).")
        if flags:
            st.warning("⚠️ Review flags:\n\n" + "\n".join(f"- {f}" for f in flags))
        else:
            st.info("No QA flags for this prediction.")

        # Recommendations
        st.subheader("Personalised recommendations")
        overview, actions = simple_recs(pred_label, proba_dict, X_live)
        for line in overview:
            st.write(f"• {line}")
        if actions:
            st.markdown("**Suggested next steps:**")
            for tip in actions:
                st.write(f"- {tip}")

        with st.expander("Show input row"):
            st.dataframe(X_df)

    with st.expander("Notes"):
        st.markdown(f"""
- If your model **doesn't** include a preprocessor, place the fitted encoder at `{PREPROCESSOR_PATH}`.
- In your notebook, save it with: `joblib.dump(preprocessor, '{PREPROCESSOR_PATH}')`.
- Set `handle_unknown='ignore'` on your `OneHotEncoder` during training to avoid errors at inference when a new category appears.
""")