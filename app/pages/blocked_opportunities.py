"""Blocked Opportunities: only the Blocked Opportunities, grouped by Market, as a before and after in Baseline-year
figures (spec #103, spec #133, ADR 0011).

A page of its own, apart from the Leaderboard, because a Blocked Opportunity is measured apart from the Export Gap:
the missing Trade Agreement already holds its Export Potential down. The Export Gap does not appear here; every
figure is for the Baseline year.

One cached read (`/opportunities/blocked`) gives every filtered Blocked Opportunity (up to `app.ui.read_limit()`), in
the order chosen in the sidebar (by the rise with agreement, or by Tariff part), with the summary, the per-Market totals
and the default thresholds. Every figure, the Barrier, both parts of the value and the sentence splitting it come from
the API; the page computes and words none of them. Left, the Markets table, largest total first; clicking any value
chooses a Market, remembered under `MARKET` (the top Market until one is chosen), with a note on why its total differs
from a Scenario and a link to that Scenario. Right, the chosen Market's Products, and under it, once one is clicked
(remembered as (HS6, ISO3) under `PAIR`), its panel: a neutral box naming the pair and its Barrier, the three-bar before
and after, one sentence on the split, its caveats, then the shared panel (`app.panel`) from its facts on. The
carried-forward tariff caveat is stated once, under the summary, not again in the panel. Filters are
`app.filters.blocked_filters`, kept apart from the Leaderboard's.
"""

import streamlit as st

from app import client, panel
from app.filters import blocked_filters, show_chips
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
    pick_row_if_chosen,
)
from app.ui import caveat_tiers, fetch, illustrative, labels, read_limit, scenario_note

MARKET = "blocked_market"  # the chosen Market, ISO3
PAIR = "blocked_pair"  # the chosen Blocked Opportunity, (HS6, ISO3)
SCENARIOS = "pages/scenarios.py"


def _explain(row: dict) -> None:
    """The chosen Blocked Opportunity's panel: a neutral box naming the pair and its Barrier, the before and after, the
    split, the caveats, then the shared panel from its facts on, without the Export Gap or the facts given above."""
    body = panel.explanation(row)
    year, word = body["baseline_year"], illustrative()
    st.info(f"**{row['description']} → {row['market']}**\n\n**Barrier:** {row['barrier']}.")
    panel.before_after(row, year)
    st.caption(f"Each is for {year}; Export Potential with a Trade Agreement is {word}.")
    st.markdown(f"{row['value_split']} {word.capitalize()}: a tendency the model has learned, not a forecast.")
    if row["above_export_potential"]:
        st.caption(
            f"UK Actual Exports in {year} are already above Export Potential, though below the level the model "
            "associates with an agreement: the model already rates current trade here as strong."
        )
    panel.opening(row, body, titled=False, with_tariff_caveat=False)
    panel.from_facts(row, body, with_barrier=False, with_gap=False)


def _markets_and_products(body: dict, rows: list[dict]) -> None:
    """The Markets table with its note and Scenario link, and beside it the chosen Market's line, its Products table
    and, once one is clicked, the chosen Product's panel."""
    markets = body["markets"]
    left, right = st.columns([1, 2], gap="large")
    with left:
        at = pick_row(
            blocked_markets_frame(markets), [market["iso3"] for market in markets], key="blocked_markets",
            remembered=MARKET, formats=BLOCKED_FORMATS, column_config=blocked_markets_columns(body),
        )
        market = markets[at]
        st.caption(
            "A Market's total sums only its listed Blocked Opportunities. A Scenario adding a Trade Agreement with "
            "the Market covers every pair, so its Scenario Effect on the Scenarios page differs."
        )
        st.page_link(
            SCENARIOS,
            label=f"Scenario: a Trade Agreement with {market['market']}",
            icon=":material/arrow_forward:",
            query_params={"add": market["iso3"]},
        )
    products = [row for row in rows if row["iso3"] == market["iso3"]]
    with right:
        count = market["blocked_opportunities"]
        st.markdown(
            f"**{market['market']}**: {count:,} Blocked Opportunit{'y' if count == 1 else 'ies'}, total "
            f"{format_usd(market['blocked_opportunity_value_usd'])} ({illustrative()})."
        )
        chosen = pick_row_if_chosen(
            blocked_products_frame(products), [(row["hs6"], row["iso3"]) for row in products],
            key=f"blocked_products_{market['iso3']}", remembered=PAIR, formats=BLOCKED_FORMATS,
            column_config=blocked_products_columns(body),
        )
        if chosen is None:
            st.caption("Click a Product to see its before and after, and how much of the rise the model associates "
                       "with the tariff.")
            return
        row = products[chosen]
        # A new key for each pair makes the panel a new element, so it starts from its top on a new selection.
        with st.container(border=True, key=f"blocked_panel_{row['hs6']}_{row['iso3']}"):
            _explain(row)


with page(
    "Blocked Opportunities",
    "UK Product–Market pairs with no UK Trade Agreement in force where the model associates adding one with a "
    "material rise in Export Potential, grouped by Market.",
):
    st.markdown(
        "A missing Trade Agreement is a Barrier the model has learned to associate with lower Export Potential. These "
        "pairs point to trade-policy work rather than export promotion. Click a Market to list its Blocked "
        "Opportunities, and a Product to see its before and after."
    )
    try:
        filters = blocked_filters()
        body = fetch("/opportunities/blocked", client.api_url(), **filters.params, limit=read_limit())
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
        thresholds = body["thresholds"]
        count, word = summary["blocked_opportunities"], illustrative()
        show_chips(filters, "blocked")
        st.markdown(
            f"{'' if filters.chips else '**No filters.** '}{count:,} Blocked Opportunit{'y' if count == 1 else 'ies'}. Total Blocked Opportunity "
            f"value, {word}: {format_usd(summary['blocked_opportunity_value_usd'])}."
        )
        st.caption(
            f"Amounts in {millions_unit()}. Every figure is for the Baseline year {year}. The Blocked Opportunity "
            f"value is the rise in {year} Export Potential the model associates with adding a UK Trade Agreement to a "
            "pair with no UK Trade Agreement in force. A pair is a Blocked Opportunity when that value is at least "
            f"{thresholds['min_effect_share']:.0%} of its {year} Export Potential and at least "
            f"{format_usd(thresholds['min_effect_usd'])}, and it has room to grow: its {year} Actual Exports are below "
            "its Export Potential with a Trade Agreement. Try other thresholds on the Scenarios page. "
            f"{labels()['totals_note']['text']} {filters.hidden_note}"
        )
        st.caption(labels()["trade_agreement_meaning"])
        if year > labels()["last_tariff_year"]:
            st.caption(labels()["tariff_carried_forward"])
        if not rows:
            st.info("No Blocked Opportunities match these filters.")
        else:
            _markets_and_products(body, rows)
            flagged = [tier["text"] for tier in caveat_tiers() if any(row[tier["column"]] for row in rows)]
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
