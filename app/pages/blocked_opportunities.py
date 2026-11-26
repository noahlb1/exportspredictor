"""Blocked Opportunities: only the Blocked Opportunities, as a before and after in Baseline-year figures (ADR 0011),
By Market, By Product or in a Wide table.

A page of its own, apart from the Leaderboard, because a Blocked Opportunity is measured apart from the Export Gap:
the missing Trade Agreement already holds its Export Potential down. The Export Gap does not appear here; every
figure is for the Baseline year.

One cached read (`/opportunities/blocked`) gives every filtered Blocked Opportunity (up to `app.ui.read_limit()`), in
the order chosen in the sidebar (by the rise with agreement, or by Tariff part), with the summary, the per-Market and
per-Product totals and the fixed thresholds (ADR 0012). Every figure, the Barrier, both parts of the value and the
sentence splitting it come from the API; the page computes and words none of them.

The Blocked Opportunities are laid out by the shared view component (`app.views`), with the page's own view, kept in
the URL as `view` and for the session apart from the Leaderboard's. The page opens By Market: Markets on the left, each
with its number of Blocked Opportunities and total value (Illustrative), largest first, and under them a note on why a
Market's total differs from a Scenario and a link to that Scenario; on the right the chosen Market's Products, from a
second cached read with the Market's filter. By Product swaps them: Products on the left, the chosen Product's Markets
on the right. The Wide table lists every Blocked Opportunity with its Product and Market. Clicking any value of a row
chooses its group or its pair; until one is clicked, the top group and its top pair are chosen.

Under the tables, at full page width, is the chosen pair's panel (`app.panel.blocked_opportunity`, shared with the
Scenario Builder): a neutral box naming the pair and its Barrier, the three-bar before and after, one sentence on the
split, its caveats, then the rest of the shared panel from its facts on. The carried-forward tariff caveat is stated
once, under the summary, not again in the panel. Filters are `app.filters.blocked_filters`, kept apart from the
Leaderboard's.
"""

import streamlit as st

from app import client, panel
from app.filters import Filters, blocked_filters, show_chips
from app.format import format_usd
from app.frame import page
from app.tables import (
    BLOCKED_FORMATS,
    BLOCKED_SHOWN,
    blocked_csv,
    blocked_products_columns,
    blocked_products_frame,
    millions_unit,
)
from app.ui import (
    caveat_tiers,
    fetch,
    illustrative,
    labels,
    read_limit,
    scenario_note,
    terms_help,
)
from app.views import BY_MARKET, Grouping, Layout, show

SCENARIOS = "pages/scenarios.py"


def _left_helps(body: dict) -> tuple[str, str]:
    """The tooltips of the left table's two headings."""
    return (
        "Its Blocked Opportunities under the filters.",
        (
            f"The sum of their Blocked Opportunity values, {body['baseline_year']}, in {millions_unit()}: only its "
            f"Blocked Opportunities, every one under the filters. {illustrative().capitalize()}."
        ),
    )


def _group_total(grouping: Grouping, body: dict) -> None:
    """Above the chosen group's pairs: their total value, Illustrative, from the group read's summary."""
    del grouping
    st.markdown(f"Total value, {illustrative()}: {format_usd(body['summary']['blocked_opportunity_value_usd'])}.")


def _scenario_link(grouping: Grouping, code: str, name: str) -> None:
    """Under the Markets table: why a Market's total differs from a Scenario, and a link to that Scenario."""
    if grouping.view != BY_MARKET:
        return
    st.caption(
        "A Market's total sums only its listed Blocked Opportunities. A Scenario adding a Trade Agreement with "
        "the Market covers every pair, so its Scenario Effect on the Scenario Builder differs."
    )
    st.page_link(
        SCENARIOS,
        label=f"Scenario Builder: a Trade Agreement with {name}",
        icon=":material/arrow_forward:",
        query_params={"add": code},
    )


def _layout(filters: Filters) -> Layout:
    """The Blocked Opportunities page's parts of the shared layout."""

    def read_group(grouping: Grouping, code: str) -> dict:
        params = {**filters.params, grouping.param: code}
        return fetch("/opportunities/blocked", client.api_url(), **params, limit=read_limit())

    return Layout(
        page="blocked",
        count="blocked_opportunities",
        total="blocked_opportunity_value_usd",
        count_heading="Blocked Opportunities",
        total_heading=f"Total value, {illustrative()}",
        left_helps=_left_helps,
        frame=lambda rows: blocked_products_frame(rows, with_market=True),
        columns=blocked_products_columns,
        formats=BLOCKED_FORMATS,
        right_columns=BLOCKED_SHOWN,
        wide_columns=BLOCKED_SHOWN,
        panel=panel.blocked_opportunity,
        read_group=read_group,
        above_right=_group_total,
        under_left=_scenario_link,
        url_param="view",
    )


with page(
    "Blocked Opportunities",
    "UK Product–Market pairs with no UK Trade Agreement in force where the model associates adding one with a "
    "material rise in Export Potential, by Market or by Product.",
    intro_help=terms_help(("Blocked Opportunity", "blocked_opportunity"), ("Trade Agreement", "trade_agreement")),
):
    st.markdown(
        "A missing Trade Agreement is a Barrier the model has learned to associate with lower Export Potential. These "
        "pairs point to trade-policy work rather than export promotion. Click a Market or a Product to list its "
        "Blocked Opportunities, and any one of them to see its before and after below."
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
        count, word = summary["blocked_opportunities"], illustrative()
        show_chips(filters, "blocked")
        st.markdown(
            f"{'' if filters.chips else '**No filters.** '}{count:,} Blocked Opportunit{'y' if count == 1 else 'ies'}. Total Blocked Opportunity "
            f"value, {word}: {format_usd(summary['blocked_opportunity_value_usd'])}."
        )
        st.caption(
            f"Amounts in {millions_unit()}. Every figure is for the Baseline year {year}. {labels()['blocked_rule']} "
            f"{labels()['totals_note']['text']} {filters.hidden_note}"
        )
        st.caption(labels()["trade_agreement_meaning"])
        if year > labels()["last_tariff_year"]:
            st.caption(labels()["tariff_carried_forward"])
        if not rows:
            st.info("No Blocked Opportunities match these filters.")
        else:
            show(_layout(filters), body, rows, "Click any value in a row to choose it.")
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
        scenario_note(body["note"])
