"""Streamlit entry point and the Leaderboard. Run with:  uv run streamlit run app/Home.py

Home opens on the Leaderboard: every UK Product–Market pair, ranked from the precomputed
Baseline table, so Advisors see the top Opportunities without searching. Each page lives in
app/pages/ and reads the model only through the API (app/client.py).
"""

import streamlit as st

from app import client
from app.ui import (
    banner,
    excluded_products_list,
    fetch,
    gap_period_label,
    ranking_filters,
    show_ranking,
)
from exportspredictor.currency import format_usd

ROWS_SHOWN = 500  # the top of the Leaderboard sent to the page; the summary covers every filtered pair

st.set_page_config(page_title="UK Exports Predictor", layout="wide")
st.title("UK Exports Predictor")
if banner() is not None:
    st.header("Leaderboard")
    st.markdown(
        "Every UK Product–Market pair, ranked by how far UK goods exports fall short of what the UK's "
        "fundamentals suggest. Filter to your portfolio, or look up one Product or Market in the sidebar. "
        "Pairs the model associates with gains from a new UK Trade Agreement are on the Blocked Opportunities page."
    )
    try:
        filters = ranking_filters(market_filter=True)
        body = fetch("/opportunities/top", client.api_url(), **filters, limit=ROWS_SHOWN)
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
    else:
        excluded_products_list()
        summary = body["summary"]
        pairs_column, gap_column = st.columns(2)
        pairs_column.metric("Product–Market pairs", f"{summary['pairs']:,}")
        gap_column.metric(
            f"Total positive Export Gap, {gap_period_label(body['gap_years'])}",
            format_usd(summary["positive_export_gap_usd"]),
        )
        if summary["pairs"] > len(body["opportunities"]):
            st.caption(f"Showing the top {len(body['opportunities']):,} of {summary['pairs']:,} pairs.")
        st.caption("Open **Why?** on a row to see which factors raised or lowered its Export Potential.")
        columns = {"hs6": "HS6", "description": "Product", "sector": "Sector", "market": "Market", "region": "Region"}
        show_ranking(body, columns, file_name="leaderboard")
