# app_pages/predict_cefr.py
import streamlit as st
import pandas as pd
from src.predict import (
    load_pipeline, load_preprocessor_if_exists, has_preprocessor, get_label_order,
    simple_recs, ID2CEFR, PREPROCESSOR_PATH
)

def render():
    st.header("Predict CEFR Level")
    st.caption("Works with a unified pipeline **or** a separate preprocessor + model.")

    try:
        pipe = load_pipeline()
    except Exception as e:
        st.error(f"Model not found or unreadable. {e}")
        return

    separate_preproc = None if has_preprocessor(pipe) else load_preprocessor_if_exists()

    with st.expander("Quick entry hints", expanded=True):
        st.markdown("""
- **Skill levels:** 0–33 = *Beginner*, 34–66 = *Intermediate*, 67–100 = *Advanced*.  
- **`<skill>_minus_avg`:** skill − mean(speaking, reading, listening, writing).  
- **`strength_weakness_gap`:** max(`<skill>_minus_avg`) − min(`<skill>_minus_avg`).  
- **`learning_profile`:** *Balanced* (skills near average) or *Uneven Development* (spread).  
- **`productive_dominant`:** choose **0** if receptive (reading+listening ≥ speaking+writing), **1** if productive (speaking+writing > reading+listening).  
        """)

    # Inputs
    skills = ["speaking","reading","listening","writing"]
    levels = ["Beginner","Intermediate","Advanced"]
    profiles = ["Balanced","Uneven Development"]

    c1, c2 = st.columns(2)
    strongest = c1.selectbox("strongest_skill", options=skills)
    weakest   = c2.selectbox("weakest_skill", options=skills)
    second_wk = c1.selectbox("second_weakest_skill", options=skills)
    profile   = c2.selectbox("learning_profile", options=profiles)

    st.markdown("**Per-skill level bins**")
    l1, l2, l3, l4 = st.columns(4)
    speaking_level   = l1.selectbox("speaking_level", options=levels)
    reading_level    = l2.selectbox("reading_level", options=levels)
    listening_level  = l3.selectbox("listening_level", options=levels)
    writing_level    = l4.selectbox("writing_level", options=levels)

    st.markdown("**Numeric deltas (can leave at 0.0 if unsure)**")
    n1, n2 = st.columns(2)
    speaking_minus_avg  = n1.number_input("speaking_minus_avg", value=0.0, step=0.1)
    reading_minus_avg   = n2.number_input("reading_minus_avg",  value=0.0, step=0.1)
    listening_minus_avg = n1.number_input("listening_minus_avg", value=0.0, step=0.1)
    writing_minus_avg   = n2.number_input("writing_minus_avg",   value=0.0, step=0.1)
    strength_weakness_gap = st.number_input("strength_weakness_gap", value=0.0, step=0.1)

    st.markdown("**Productive vs Receptive balance**")
    mode = st.radio("Set productive_dominant", ["Simple (0/1)", "Manual numeric"], horizontal=True)
    if mode == "Simple (0/1)":
        pd_flag = st.selectbox("productive_dominant_flag (0=receptive, 1=productive)", options=[0, 1], index=0)
        productive_dominant = float(pd_flag)
    else:
        productive_dominant = st.number_input("productive_dominant (numeric)", value=0.0, step=0.1)

    X_live = {
        "strongest_skill": strongest,
        "weakest_skill": weakest,
        "second_weakest_skill": second_wk,
        "strength_weakness_gap": float(strength_weakness_gap),
        "learning_profile": profile,
        "speaking_minus_avg": float(speaking_minus_avg),
        "reading_minus_avg":  float(reading_minus_avg),
        "listening_minus_avg":float(listening_minus_avg),
        "writing_minus_avg":  float(writing_minus_avg),
        "productive_dominant":float(productive_dominant),
        "speaking_level": speaking_level,
        "reading_level": reading_level,
        "listening_level": listening_level,
        "writing_level": writing_level,
    }

    st.markdown("---")
    rule_cefr = st.selectbox("Optional: rule-based/known CEFR for QA", options=["(none)","A1","A2","B1","B2","C1","C2"], index=0)

    if st.button("Predict CEFR"):
        X_df = pd.DataFrame([X_live])
        try:
            if has_preprocessor(pipe):
                yhat = pipe.predict(X_df)[0]
                proba = pipe.predict_proba(X_df)[0] if hasattr(pipe, "predict_proba") else None
                label_order = get_label_order(pipe)
            else:
                if separate_preproc is None:
                    st.error(
                        "Your saved model does not include a preprocessor and "
                        f"`{PREPROCESSOR_PATH}` was not found.\n\n"
                        "Fix: save the fitted preprocessor from the notebook:\n"
                        f"`joblib.dump(preprocessor, '{PREPROCESSOR_PATH}')`"
                    )
                    return
                X_enc = separate_preproc.transform(X_df)
                yhat = pipe.predict(X_enc)[0]
                proba = pipe.predict_proba(X_enc)[0] if hasattr(pipe, "predict_proba") else None
                label_order = get_label_order(pipe)
        except ValueError as e:
            st.error(
                "Prediction failed during type conversion/encoding. Ensure the pipeline contains a "
                "`preprocessor` or you placed the fitted preprocessor at the path shown above.\n\n"
                f"Details: {e}"
            )
            return

        pred_label = ID2CEFR.get(int(yhat), str(yhat))
        st.success(f"**Predicted CEFR:** {pred_label}")

        # Map probabilities to CEFR labels (if available)
        proba_dict = {}
        if proba is not None and label_order is not None:
            ce_labels = [ID2CEFR.get(int(c), str(c)) for c in label_order]
            proba_dict = {lab: float(round(p, 3)) for lab, p in zip(ce_labels, proba)}
            st.write("Class probabilities:")
            st.write(proba_dict)

        flags = []
        if proba_dict and max(proba_dict.values()) < 0.55:
            flags.append("Low confidence prediction (max p < 0.55).")
        if rule_cefr != "(none)" and rule_cefr != pred_label:
            flags.append(f"Model disagrees with rule-based CEFR (**{rule_cefr}** vs **{pred_label}**).")
        if flags: st.warning("⚠️ Review flags:\n\n" + "\n".join(f"- {f}" for f in flags))
        else: st.info("No QA flags for this prediction.")

        st.subheader("Personalised recommendations")
        overview, actions = simple_recs(pred_label, proba_dict, X_live)
        for line in overview: st.write(f"• {line}")
        if actions:
            st.markdown("**Suggested next steps:**")
            for tip in actions: st.write(f"- {tip}")

        with st.expander("Show input row"):
            st.dataframe(X_df)
