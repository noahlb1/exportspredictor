"""Blocked Opportunities: only the Blocked Opportunities, grouped by Market, in Baseline-year figures (spec #103).

A page of its own, apart from the Leaderboard, because a Blocked Opportunity is measured apart from the Export Gap:
the missing Trade Agreement already holds its Export Potential down. The Export Gap does not appear here; every
figure is for the Baseline year.

One cached read (`/opportunities/blocked`) gives every filtered Blocked Opportunity (up to `READ_LIMIT`) with the
summary and per-Market totals. Left, the Markets table, largest total first; clicking any value chooses a Market,
remembered under `MARKET`. Right, the chosen Market's Products, every one of them, and under it the detail panel
(`app.panel`) of the chosen Product, remembered as (HS6, ISO3) under `PAIR`. Until one is chosen, or when the chosen
one is not shown, the top row is. Filters are `app.filters.blocked_filters`, kept apart from the Leaderboard's.
"""

import streamlit as st

from app import client, panel
from app.filters import blocked_filters
from app.format import format_usd
from app.frame import page
from app.tables import (
    BLOCKED_FORMATS,
    blocked_csv,
    blocked_markets_columns,
    blocked_markets_frame,
    blocked_products_columns,
    blocked_products_frame,
    millions_unit,
    pick_row,
)
from app.ui import READ_LIMIT, caveats, fetch, illustrative, labels, scenario_note

MARKET = "blocked_market"  # the chosen Market, ISO3
PAIR = "blocked_pair"  # the chosen Blocked Opportunity, (HS6, ISO3)
# In the Barrier box, only where Baseline-year Actual Exports are above Export Potential with a Trade Agreement (#102).
ABOVE_WITH_AGREEMENT = (
    "The UK already exports more here than the Export Potential the model associates with a Trade Agreement. The "
    "Blocked Opportunity value is still the rise in Export Potential the model associates with the agreement; it does "
    "not depend on Actual Exports."
)


def _explain(row: dict) -> None:
    """The chosen Blocked Opportunity's panel: the Barrier first, then three Baseline-year figures, then the shared
    panel from its facts table on, without the Export Gap."""
    body = panel.explanation(row)
    panel.opening(row, body)
    above = row["baseline_actual_exports_usd"] > row["potential_with_agreement_usd"]
    panel.barrier(row, body, note=ABOVE_WITH_AGREEMENT if above else None)
    year, word = body["baseline_year"], illustrative()
    potential, with_agreement, actual = st.columns(3)
    potential.metric(
        "Baseline Export Potential",
        format_usd(row["baseline_export_potential_usd"]),
        help=f"The Export Potential the model expects in {year}, under the trade-policy conditions in force.",
    )
    with_agreement.metric(
        "Export Potential with a Trade Agreement",
        format_usd(row["potential_with_agreement_usd"]),
        help=f"Baseline Export Potential plus the Blocked Opportunity value: the {year} level the model associates "
        f"with adding a UK Trade Agreement and removing the tariff. {word.capitalize()}.",
    )
    actual.metric(
        "Actual Exports",
        format_usd(row["baseline_actual_exports_usd"]),
        help=f"What the UK actually exported of this Product to this Market in {year} (0 where there was no Flow).",
    )
    st.caption(f"Each is for {year}; Export Potential with a Trade Agreement is {word}.")
    panel.from_facts(row, body, with_barrier=False, with_gap=False)


def _markets_and_products(body: dict, rows: list[dict]) -> None:
    """The Markets table, and beside it the chosen Market's line, its Products table and the chosen Product's panel."""
    markets = body["markets"]
    left, right = st.columns([2, 5], gap="large")
    with left:
        at = pick_row(
            blocked_markets_frame(markets), [market["iso3"] for market in markets], key="blocked_markets",
            remembered=MARKET, formats=BLOCKED_FORMATS, column_config=blocked_markets_columns(body),
        )
    market = markets[at]
    products = [row for row in rows if row["iso3"] == market["iso3"]]
    with right:
        count = market["blocked_opportunities"]
        st.markdown(
            f"**{market['market']}**: {count:,} Blocked Opportunit{'y' if count == 1 else 'ies'}, total "
            f"{format_usd(market['blocked_opportunity_value_usd'])} ({illustrative()})."
        )
        chosen = pick_row(
            blocked_products_frame(products), [(row["hs6"], row["iso3"]) for row in products],
            key=f"blocked_products_{market['iso3']}", remembered=PAIR, formats=BLOCKED_FORMATS,
            column_config=blocked_products_columns(body),
        )
        row = products[chosen]
        # A new key for each pair makes the panel a new element, so it starts from its top on a new selection.
        with st.container(border=True, key=f"blocked_panel_{row['hs6']}_{row['iso3']}"):
            _explain(row)


with page(
    "Blocked Opportunities",
    "UK Product–Market pairs where the model associates adding a UK Trade Agreement, and removing the tariff, "
    "with a material rise in Export Potential, grouped by Market.",
):
    st.markdown(
        "A missing agreement is a Barrier the model has learned to associate with lower Export Potential. These pairs "
        "point to trade-policy work rather than export promotion. Click a Market to list its Blocked Opportunities, "
        "and a Product to explain it."
    )
    try:
        filters = blocked_filters()
        thresholds = fetch("/scenarios/thresholds", client.api_url())
        body = fetch("/opportunities/blocked", client.api_url(), **filters.params, limit=READ_LIMIT)
    except client.NotFound as error:
        # An Excluded Product chosen in the search (or named in a link): its reason, and no list.
        st.warning(str(error))
        st.markdown(
            "Excluded Products are never ranked. The **Excluded Products** section of **Model Evaluation** lists "
            "each one with its reason."
        )
        st.page_link("pages/model_evaluation.py", label="Open Model Evaluation")
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
    else:
        summary, rows, year = body["summary"], body["opportunities"], body["baseline_year"]
        count, word = summary["blocked_opportunities"], illustrative()
        in_force = "; ".join(filters.named) if filters.named else "No filters"
        st.markdown(
            f"**{in_force}.** {count:,} Blocked Opportunit{'y' if count == 1 else 'ies'}. Total Blocked Opportunity "
            f"value, {word}: {format_usd(summary['blocked_opportunity_value_usd'])}."
        )
        st.caption(
            f"Amounts in {millions_unit()}. Every figure is for the Baseline year {year}. The Blocked Opportunity "
            f"value is the rise in {year} Export Potential the model associates with adding a UK Trade Agreement and "
            "removing the tariff where none is in force. A pair is a Blocked Opportunity when that value is at least "
            f"{thresholds['min_effect_share']:.0%} of its {year} Export Potential and at least "
            f"{format_usd(thresholds['min_effect_usd'])}. Try other thresholds on the Scenarios page."
        )
        if not rows:
            st.info("No Blocked Opportunities match these filters.")
        else:
            _markets_and_products(body, rows)
            flagged = [caveat for field, (_, caveat) in caveats().items() if any(row[field] for row in rows)]
            if flagged:
                st.caption(" ".join(flagged) + " Use the hide filters to leave them out.")
            if count > len(rows):
                st.caption(f"The page and the CSV hold the top {len(rows):,} of {count:,} Blocked Opportunities.")
            st.download_button(
                "Download CSV",
                blocked_csv(body, rows).to_csv(index=False),
                file_name="blocked_opportunities.csv",
                mime="text/csv",
                key="csv",
            )
        with st.expander(f"{labels()['illustrative']['badge']}: what the Blocked Opportunity value is read against"):
            scenario_note(body["note"])
