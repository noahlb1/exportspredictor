"""The Leaderboard, the app's default page: every UK Product–Market pair, ranked from the precomputed Baseline
table, so Advisors see the top Opportunities without searching. Its filters sit in the sidebar and in the URL
(`app.filters`); a Product search or a Market filter ranks one Product's Markets or one Market's Products.

One cached read of up to `read_limit()` rows per filter set: the page shows the top `ROWS_SHOWN` of them (all of them
with Show all), and the CSV holds them all, in full dollars, one row per pair. The same read serves the per-Market and
per-Product totals over every filtered pair.

The pairs are laid out by the shared view component (`app.views`). The page opens By Market: Markets on the left,
each with its number of pairs and total positive Export Gap, largest first; the chosen Market's Products on the right,
from a second cached read with the Market's filter (so every pair of the Market is listed even when the main read is
capped), under the Market's Market Reach sentence. Clicking any value of a row chooses its Market or its pair, and the
detail panel (`app.panel`) under both tables, at full page width, explains the chosen pair. Until one is clicked, the
top Market and its top pair are chosen.

By Product swaps the two: Products on the left with the same counts and totals, the chosen Product's Markets on the
right from a read with the Product's filter. The Wide table lists every shown pair across the page in
`LEADERBOARD_SHOWN`, the other columns hidden but offered. The chosen pair is kept across the three views wherever it
is still shown, and each grouping keeps its own chosen group. The view is in the URL (`view=product` or `view=wide`;
By Market is left out) and kept for the session; Reset filters leaves it, as it is not a filter.
"""

import streamlit as st

from app import client, panel
from app.filters import Filters, leaderboard_filters, show_all, show_chips
from app.format import format_usd
from app.frame import page
from app.tables import (
    LEADERBOARD_FORMATS,
    LEADERBOARD_RIGHT,
    LEADERBOARD_SHOWN,
    leaderboard_columns,
    leaderboard_csv,
    leaderboard_frame,
    leaderboard_unit,
    millions_unit,
)
from app.ui import (
    ROWS_SHOWN,
    caveat_tiers,
    fetch,
    labels,
    rank_orders,
    read_limit,
    term,
    terms_help,
)
from app.views import BY_MARKET, Grouping, Layout, show

STEPS = "leaderboard_steps"  # whether the three steps are shown; closed with Got it for the browser session
STEP_TEXTS = [
    (
        "1 · Read the ranking.",
        "Markets are on the left, the largest total positive Export Gap first. Click one to list its Products on the right.",
    ),
    (
        "2 · Open a pair.",
        "Click any Product. The panel below the tables shows which factors raised or lowered its Export Potential.",
    ),
    ("3 · Narrow it down.", "Use the filters on the left to keep one Sector, Market or Product."),
]


def _set_steps(shown: bool) -> None:
    st.session_state[STEPS] = shown


def _what_is_this() -> None:
    """"What is this?" beside the heading, once the three steps are closed: it brings them back."""
    if not st.session_state.get(STEPS, True):
        st.button("What is this?", type="tertiary", on_click=_set_steps, args=(True,), key="steps_open")


def _three_steps() -> None:
    """Three bordered steps telling a first-time visitor how to read the page, with Got it to close them."""
    if not st.session_state.get(STEPS, True):
        return
    columns = st.columns([4, 4, 4, 1.5], vertical_alignment="center")  # room for "Got it" at 1280 px
    for column, (title, text) in zip(columns, STEP_TEXTS, strict=False):
        with column.container(border=True):
            st.markdown(f"**{title}** {text}")
    columns[3].button("Got it", on_click=_set_steps, args=(False,), key="steps_close")


def _left_helps(body: dict) -> tuple[str, str]:
    """The tooltips of the left table's two headings."""
    return (
        "Its UK Product–Market pairs under the filters.",
        (
            f"The sum of their positive Export Gaps, {body['gap_period_label']}, in {millions_unit()}: every pair "
            "under the filters, as in the line above."
        ),
    )


def _market_reach(grouping: Grouping, body: dict) -> None:
    """Above the chosen Market's Products: its Market Reach sentence (ADR 0010)."""
    if grouping.view == BY_MARKET and body.get("market_reach"):
        st.markdown(f"**Market Reach.** {body['market_reach']['text']}", help=term("market_reach"))


def _layout(filters: Filters) -> Layout:
    """The Leaderboard's parts of the shared layout."""

    def read_group(grouping: Grouping, code: str) -> dict:
        params = {**filters.params, grouping.param: code}
        return fetch("/opportunities/top", client.api_url(), **params, limit=read_limit())

    return Layout(
        page="leaderboard",
        count="pairs",
        total="positive_export_gap_usd",
        count_heading="Pairs",
        total_heading="Total positive Export Gap",
        left_helps=_left_helps,
        frame=leaderboard_frame,
        columns=lambda body: leaderboard_columns(),
        formats=LEADERBOARD_FORMATS,
        right_columns=LEADERBOARD_RIGHT,
        wide_columns=LEADERBOARD_SHOWN,
        panel=panel.show,
        read_group=read_group,
        above_right=_market_reach,
        url_param="view",
    )


with page(
    "Leaderboard",
    "Every UK Product–Market pair, ranked by how far UK goods exports fall short of what the UK's fundamentals "
    "suggest.",
    beside=_what_is_this,
    intro_help=terms_help(("Product–Market pair", "pair"), ("Opportunity", "opportunity")),
):
    _three_steps()
    try:
        filters = leaderboard_filters()
        body = fetch("/opportunities/top", client.api_url(), **filters.params, limit=read_limit())
    except client.NotFound as error:
        # An Excluded Product chosen in the search (or named in a link): its reason, and no ranking.
        st.warning(str(error))
        st.markdown(
            "Excluded Products are never ranked. The **Excluded Products** section of **Model Evaluation** lists "
            "each one with its reason."
        )
        st.page_link("pages/model_evaluation.py", label="Open Model Evaluation")
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
    else:
        summary, read = body["summary"], body["opportunities"]
        pairs = summary["pairs"]
        show_chips(filters, "leaderboard")
        line = st.empty()  # the summary line, written once Show all is known
        rows = read if show_all(pairs) else read[:ROWS_SHOWN]
        if pairs > read_limit():
            shown = f"Top {len(rows):,} of {pairs:,} pairs: narrow them down with the filters to show them all."
        else:
            shown = f"{len(rows):,} pair{'' if len(rows) == 1 else 's'} shown of {pairs:,}."
        line.markdown(
            f"{'' if filters.chips else '**No filters.** '}{shown} "
            f"Total positive Export Gap, {body['gap_period_label']}: {format_usd(summary['positive_export_gap_usd'])}."
        )
        order = rank_orders()[body["rank_by"]]
        st.caption(
            f"Ranked by {order['title']}, Baseline year {body['baseline_year']}. {order['description']} "
            f"{filters.hidden_note}"
        )
        st.caption(labels()["totals_note"]["text"])  # beside the summed Export Gap
        if not rows:
            st.info("No Opportunities match these filters.")
        else:
            show(_layout(filters), body, rows, leaderboard_unit(body) + " Click any value in a row to choose it.")
            flagged = [tier["text"] for tier in caveat_tiers() if any(row[tier["column"]] for row in read)]
            if flagged:
                st.caption(" ".join(flagged) + " Use the hide filters to leave them out.")
            if pairs > len(read):
                st.caption(f"The CSV holds the top {len(read):,} of {pairs:,} pairs.")
            st.download_button(
                "Download CSV",
                leaderboard_csv(body, read).to_csv(index=False),
                file_name="leaderboard.csv",
                mime="text/csv",
                key="csv",
            )
