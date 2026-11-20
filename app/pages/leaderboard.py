"""The Leaderboard, the app's default page: every UK Product–Market pair, ranked from the precomputed Baseline
table, so Advisors see the top Opportunities without searching."""

import streamlit as st

from app import client
from app.format import format_usd
from app.frame import page
from app.ui import (
    ROWS_SHOWN,
    excluded_products_list,
    fetch,
    ranking_filters,
    show_leaderboard,
)

with page(
    "Leaderboard",
    "Every UK Product–Market pair, ranked by how far UK goods exports fall short of what the UK's fundamentals "
    "suggest.",
):
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
