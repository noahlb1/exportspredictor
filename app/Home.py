"""Streamlit entry point: the router. Run with:  uv run streamlit run app/Home.py

It declares the app's pages, sets what every page shares once (`app.frame.setup`), and runs the chosen page only
while the API is up. Each page lives in app/pages/, wraps its content in `app.frame.page` and reads the model only
through the API (app/client.py).
"""

import streamlit as st

from app import frame

frame.setup()
chosen = st.navigation(
    [
        st.Page("pages/leaderboard.py", title="Leaderboard", default=True),
        st.Page("pages/blocked_opportunities.py", title="Blocked Opportunities", url_path="Blocked_Opportunities"),
        st.Page("pages/scenarios.py", title="Scenario Builder", url_path="Scenario_Builder"),
        st.Page("pages/model_evaluation.py", title="Model Evaluation", url_path="Model_Evaluation"),
        st.Page("pages/how_the_model_works.py", title="How the model works", url_path="How_the_model_works"),
    ]
)
if frame.api_ready():
    chosen.run()
