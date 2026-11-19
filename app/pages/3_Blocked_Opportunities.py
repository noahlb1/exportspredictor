"""Blocked Opportunities: every UK Product–Market pair ranked by Blocked Opportunity value.

A page of its own, apart from the Leaderboard, because a Blocked Opportunity shows no Export
Gap: the missing Trade Agreement already holds its Export Potential down. It has the
Leaderboard's filters, summary and table, with one fixed order, and reads the API only.
"""

import streamlit as st

from app import client
from app.format import format_usd
from app.ui import (
    ROWS_SHOWN,
    banner,
    excluded_products_list,
    fetch,
    illustrative,
    ranking_filters,
    scenario_note,
    show_leaderboard,
)

st.title("Blocked Opportunities")
if banner() is not None:
    st.markdown(
        "UK Product–Market pairs where the model associates adding a UK Trade Agreement, and removing the tariff, "
        "with a material rise in Export Potential. A missing agreement is a Barrier the model has learned to associate with lower "
        "Export Potential, so these pairs need not show an Export Gap; they point to trade-policy work rather "
        "than export promotion. Each Blocked Opportunity value is a tendency the model has learned, not a forecast."
    )
    try:
        filters = ranking_filters(market_filter=True, rank_by_toggle=False)
        thresholds = fetch("/scenarios/thresholds", client.api_url())
        body = fetch("/opportunities/blocked", client.api_url(), **filters, limit=ROWS_SHOWN)
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
    else:
        excluded_products_list()
        summary = body["summary"]
        pairs_column, count_column, value_column = st.columns(3)
        pairs_column.metric("Product–Market pairs", f"{summary['pairs']:,}")
        count_column.metric("Blocked Opportunities", f"{summary['blocked_opportunities']:,}")
        value_column.metric(f"Total Blocked Opportunity value, {illustrative()}", format_usd(summary["blocked_opportunity_value_usd"]))
        scenario_note(body["note"])
        st.caption(
            f"The Blocked Opportunity value is the change in {body['baseline_year']} Export Potential the model "
            "associates with adding a UK Trade Agreement and removing the tariff where no agreement is in force; "
            "it is 0 where one is. "
            f"A pair is a Blocked Opportunity when that value is at least {thresholds['min_effect_share']:.0%} of its "
            f"{body['baseline_year']} Export Potential and at least {format_usd(thresholds['min_effect_usd'])}. "
            "Try other agreements and thresholds on the Scenarios page."
        )
        show_leaderboard(body, file_name="blocked_opportunities")
