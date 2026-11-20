"""The Leaderboard, the app's default page: every UK Product–Market pair, ranked from the precomputed Baseline
table, so Advisors see the top Opportunities without searching. Its filters sit in the sidebar and in the URL
(`app.filters`); a Product search or a Market filter ranks one Product's Markets or one Market's Products.

One cached read of up to `READ_LIMIT` rows per filter set: the page shows the top `ROWS_SHOWN` of them (all of them
with Show all), and the CSV holds them all, in full dollars.

Clicking any value of a row chooses its pair, remembered as (HS6, ISO3) under `PAIR`, and the detail panel
(`app.panel`) explains it in place: beside a three-column table (Table and detail) or under the wide table (Wide
table). Until a pair is chosen, or when the chosen pair is not among the rows shown, the top row is explained.
"""

import streamlit as st

from app import client, panel
from app.filters import leaderboard_filters, show_all
from app.format import format_usd
from app.frame import page
from app.tables import (
    LEADERBOARD_FORMATS,
    LEADERBOARD_NARROW,
    LEADERBOARD_SHOWN,
    leaderboard_columns,
    leaderboard_csv,
    leaderboard_frame,
    leaderboard_unit,
    pick_row,
)
from app.ui import (
    READ_LIMIT,
    ROWS_SHOWN,
    caveats,
    fetch,
    rank_orders,
)

PAIR = "leaderboard_pair"  # the chosen pair, (HS6, ISO3)
VIEW = "leaderboard_view"
SPLIT, WIDE = "Table and detail", "Wide table"
STEPS = "leaderboard_steps"  # whether the three steps are shown; closed with Got it for the browser session
STEP_TEXTS = [
    ("1 · Read the ranking.", "Each row is one Product in one Market. The largest Export Gap is at the top."),
    (
        "2 · Open a row.",
        "Click any value. The panel on the right shows which factors raised or lowered its Export Potential.",
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


def _table_and_panel(body: dict, rows: list[dict]) -> None:
    """The view switch, the table, and the chosen pair's detail panel beside or under it."""
    switch, unit = st.columns([2, 5], vertical_alignment="center")
    with switch:
        view = st.segmented_control(
            "View", [SPLIT, WIDE], key=VIEW, default=SPLIT, required=True, label_visibility="collapsed",
            persist_state="session",
        )
    unit.caption(leaderboard_unit(body) + " Click any value in a row to explain that pair.")
    table, ids = leaderboard_frame(rows), [(row["hs6"], row["iso3"]) for row in rows]
    config = leaderboard_columns(body)
    if view == WIDE:
        at = pick_row(
            table, ids, key="leaderboard_wide", remembered=PAIR, formats=LEADERBOARD_FORMATS,
            column_order=LEADERBOARD_SHOWN, column_config=config, height=390,
        )
        with st.container(border=True):
            panel.show(rows[at])
        return
    table_side, panel_side = st.columns([5, 4], gap="large")
    with table_side:
        at = pick_row(
            table[LEADERBOARD_NARROW], ids, key="leaderboard_narrow", remembered=PAIR, formats=LEADERBOARD_FORMATS,
            column_config={name: config[name] for name in LEADERBOARD_NARROW}, height=700,
        )
    # A new key for each pair makes the panel a new element, so it starts from its top on a new selection.
    with panel_side, st.container(border=True, height=780, key=f"leaderboard_panel_{ids[at][0]}_{ids[at][1]}"):
        panel.show(rows[at])


with page(
    "Leaderboard",
    "Every UK Product–Market pair, ranked by how far UK goods exports fall short of what the UK's fundamentals "
    "suggest.",
    beside=_what_is_this,
):
    _three_steps()
    try:
        filters = leaderboard_filters()
        body = fetch("/opportunities/top", client.api_url(), **filters.params, limit=READ_LIMIT)
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
        line = st.empty()  # the active-filters line, written once Show all is known
        rows = read if show_all(pairs) else read[:ROWS_SHOWN]
        in_force = "; ".join(filters.named) if filters.named else "No filters"
        if pairs > READ_LIMIT:
            shown = f"Top {len(rows):,} of {pairs:,} pairs: narrow them down with the filters to show them all."
        else:
            shown = f"{len(rows):,} pair{'' if len(rows) == 1 else 's'} shown of {pairs:,}."
        line.markdown(
            f"**{in_force}.** {shown} "
            f"Total positive Export Gap, {body['gap_period_label']}: {format_usd(summary['positive_export_gap_usd'])}."
        )
        order = rank_orders()[body["rank_by"]]
        st.caption(f"Ranked by {order['title']}, Baseline year {body['baseline_year']}. {order['description']}")
        if not rows:
            st.info("No Opportunities match these filters.")
        else:
            _table_and_panel(body, rows)
            flagged = [caveat for field, (_, caveat) in caveats().items() if any(row[field] for row in read)]
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
