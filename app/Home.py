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
        st.Page("pages/scenarios.py", title="Scenarios", url_path="Scenarios"),
        st.Page("pages/model_evaluation.py", title="Model Evaluation", url_path="Model_Evaluation"),
        # Out of the menu but still reachable by their old links, until later tickets of spec #103 delete them.
        st.Page("pages/product_to_markets.py", title="Product → Markets", url_path="Product_to_Markets", visibility="hidden"),
        st.Page("pages/market_to_products.py", title="Market → Products", url_path="Market_to_Products", visibility="hidden"),
        st.Page("pages/why.py", title="Why?", url_path="Why", visibility="hidden"),
    ]
)
if frame.api_ready():
    chosen.run()
