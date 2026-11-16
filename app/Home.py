"""Streamlit entry point. Run with:  uv run streamlit run app/Home.py

Each page lives in app/pages/ and reads the model only through the API (app/client.py).
"""

import streamlit as st

from app.ui import banner

st.set_page_config(page_title="UK Exports Predictor", layout="wide")
st.title("UK Exports Predictor")
banner()
st.markdown(
    "Find where UK goods exports fall short of what the UK's fundamentals suggest. "
    "Start with **Product → Markets** in the sidebar."
)
