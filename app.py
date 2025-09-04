# app.py
import streamlit as st
from app_pages.home import render as page_home
from app_pages.data_explorer import render as page_data
from app_pages.model_evaluation import render as page_eval
from app_pages.predict_cefr import render as page_predict
from app_pages.hypotheses import render as page_hypotheses
from app_pages.about import render as page_about

PAGES = {
    "🏠 Quick Project Summary": page_home,
    "🔎 Data Explorer": page_data,
    "📈 Model Evaluation": page_eval,
    "🎯 Predict CEFR": page_predict,
    "🧪 Hypotheses & Validation": page_hypotheses,
    "ℹ️ About & Methods": page_about,
}

def main():
    st.set_page_config(page_title="CEFR Classifier Dashboard", page_icon="🎓", layout="wide")
    with st.sidebar:
        st.title("CEFR Dashboard")
        choice = st.radio("Navigate", list(PAGES.keys()), index=0)
    PAGES[choice]()

if __name__ == "__main__":
    main()
