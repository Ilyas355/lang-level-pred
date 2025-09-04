# app.py (snippet)
import streamlit as st
from app_pages import home, data_explorer, model_evaluation, predict_cefr, about

PAGES = {
    "🏠 Quick Project Summary": home.render,
    "📊 Data Explorer": data_explorer.render,
    "🧪 Model Evaluation": model_evaluation.render,
    "🎯 Predict CEFR": predict_cefr.render,
    "ℹ️ About & Methods": about.render,
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()