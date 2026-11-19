"""Streamlit entry point and the Leaderboard. Run with:  uv run streamlit run app/Home.py

Home opens on the Leaderboard: every UK Product–Market pair, ranked from the precomputed
Baseline table, so Advisors see the top Opportunities without searching. Each page lives in
app/pages/ and reads the model only through the API (app/client.py).
"""

import os

import streamlit as st

from app import (  # PROTOTYPE (#95, #97): never merge
    client,
    prototype_banner,
    prototype_pages,
    prototype_leaderboard,
    prototype_logo,
)
from app.format import format_usd
from app.ui import (
    ROWS_SHOWN,
    banner,
    excluded_products_list,
    fetch,
    ranking_filters,
    show_leaderboard,
)

PROTOTYPE = bool(os.environ.get("EXPORTSPREDICTOR_PROTOTYPE"))  # off unless asked for, so the app tests see today's page
if PROTOTYPE:  # PROTOTYPE (#100): the four pages under the settled menu, each drawn by the prototype
    prototype_pages.run()
    st.stop()
if PROTOTYPE:
    prototype_logo.page_setup()
else:
    st.set_page_config(page_title="UK Exports Predictor", layout="wide")
    st.title("UK Exports Predictor")

if banner() is None:
    pass
elif PROTOTYPE:  # the split view settled in #95, under the logo being chosen
    prototype_leaderboard.variant_c(None, "0.0")
else:
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
            f"Total positive Export Gap, {body['gap_period_label']}",
            format_usd(summary["positive_export_gap_usd"]),
        )
        show_leaderboard(body, file_name="leaderboard")

if PROTOTYPE:
    prototype_logo.bar()
