"""One layout for the ranking pages: the Leaderboard and Blocked Opportunities show their UK
Product–Market pairs By Market, By Product or in a Wide table, with the chosen pair's detail panel underneath at full
page width.

`show(layout, body, rows)` draws, in order:

1. the view switch, a segmented control ("By Market", "By Product", "Wide table"; `VIEWS`, the first is the
   default), with the page's `caption` beside it;
2. By Market or By Product (`GROUPINGS`): the left table of groups (each Market or Product with its count and total,
   as the API serves them in `body[grouping.totals]`, largest first) and, at 1:2 width, the right table of the chosen
   group's pairs. The right table comes from `Layout.read_group(grouping, code)`, a second read with the page's
   filters plus the group's (`market=<ISO3>` or `product=<HS6>`), so it lists every pair of the group even when the
   main read is capped. A line above it names the group and its pair count; `Layout.above_right` may add to it (the
   Leaderboard's Market Reach sentence) and `Layout.under_left` may draw under the left table (Blocked
   Opportunities' Scenario note and link);
3. Wide table: every row of `rows` in `Layout.wide_columns`;
4. the chosen pair's panel (`Layout.panel(row)`) in a bordered container across the page, keyed by pair, so it starts
   at its top on a new selection.

Every table is a `pick_row` table: the top row is chosen until another is clicked, and the chosen row is tinted.

Selection state is the page's own (`Layout.page` prefixes every key), so two pages never share it:

- one chosen pair, (HS6, ISO3), under `pair_key(page)`, shared by the three views: a pair still shown stays chosen
  when the view changes, and a group whose pairs do not include it shows its top pair;
- one chosen group per grouping (an ISO3 for By Market, an HS6 for By Product);
- the view, under `view_key(page)`, kept for the session.

The view in the URL: with `Layout.url_param` set (e.g. `"view"`), the view is read from and written to that query
parameter as `app.filters` does the filters (`product` or `wide`; the default view is left out). A URL that names it,
and that this page did not write itself, sets the view; a URL that does not name it leaves the session's view. It is
off (`None`) until a page switches it on.

A page plugs in what differs through `Layout`: the left table's count and total (fields, headings and tooltips), how
to build its pair table (`frame`, `columns`, `formats`), the right-table and Wide-table columns, the panel function,
the group read and the optional hooks. The component computes no figures: every total is the API's.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from app.tables import (
    ONE_DECIMAL,
    market_column,
    millions,
    money_column,
    pick_row,
    product_column,
)

BY_MARKET, BY_PRODUCT, WIDE = "By Market", "By Product", "Wide table"
VIEWS = (BY_MARKET, BY_PRODUCT, WIDE)
_URL_VALUES = {BY_PRODUCT: "product", WIDE: "wide"}  # the view's URL value; the default view is left out
HEIGHT = 420  # px, the left and Wide tables; the right one is a little shorter, under its line


@dataclass(frozen=True)
class Grouping:
    """How By Market or By Product groups the pairs."""

    view: str
    code: str  # the row field that groups: "iso3" or "hs6"
    name: str  # the row field that names the group: "market" or "description"
    heading: str  # the left table's first column, and the right table's column it leaves out: "Market" or "Product"
    totals: str  # the body field of the per-group totals: "markets" or "products"
    param: str  # the read's filter that keeps one group: "market" or "product"
    member: str  # what one right-table row is: "Product" or "Market"
    members: str  # plural of `member`
    column: Callable[[], object]  # the column config of `heading`: `market_column` or `product_column`


GROUPINGS = {
    BY_MARKET: Grouping(
        BY_MARKET, "iso3", "market", "Market", "markets", "market", "Product", "Products", market_column
    ),
    BY_PRODUCT: Grouping(
        BY_PRODUCT, "hs6", "description", "Product", "products", "product", "Market", "Markets", product_column
    ),
}


@dataclass(frozen=True)
class Layout:
    """What one page passes to `show`.

    - `page`: the prefix of the page's session-state keys, e.g. "leaderboard".
    - `count`, `total`: the fields of a group-total row (`body[grouping.totals]`), e.g. "pairs" and
      "positive_export_gap_usd"; the total is in dollars and shown in $ millions.
    - `count_heading`, `total_heading`: the left table's headings, e.g. "Pairs", "Total positive Export Gap".
    - `left_helps(body)`: their tooltips, (count, total).
    - `frame(rows)`: the pair table of ranking rows (`leaderboard_frame`, `blocked_products_frame`); `columns(body)`
      its column config; `formats` its `pick_row` formats.
    - `right_columns`: the right table's columns, with both Product and Market: the grouping's own is left out.
    - `wide_columns`: the Wide table's columns.
    - `panel(row)`: draws the chosen pair's detail panel.
    - `read_group(grouping, code)`: the group read's body, the page's filters plus `{grouping.param: code}`; its
      `opportunities` are the right table's rows.
    - `above_right(grouping, body)`: optional, drawn above the right table after its line, with the group read's body.
    - `under_left(grouping, code, name)`: optional, drawn under the left table, with the chosen group.
    - `url_param`: the view's query parameter, or None to keep the view out of the URL.
    """

    page: str
    count: str
    total: str
    count_heading: str
    total_heading: str
    left_helps: Callable[[dict], tuple[str, str]]
    frame: Callable[[list[dict]], pd.DataFrame]
    columns: Callable[[dict], dict]
    formats: dict
    right_columns: list[str]
    wide_columns: list[str]
    panel: Callable[[dict], None]
    read_group: Callable[[Grouping, str], dict]
    above_right: Callable[[Grouping, dict], None] | None = None
    under_left: Callable[[Grouping, str, str], None] | None = None
    url_param: str | None = None


def pair_key(page: str) -> str:
    """Where a page remembers its chosen pair, (HS6, ISO3)."""
    return f"{page}_pair"


def view_key(page: str) -> str:
    """Where a page remembers its view (the switch's key)."""
    return f"{page}_view"


def show(layout: Layout, body: dict, rows: list[dict], caption: str = "") -> None:
    """The view switch with `caption` beside it, the chosen view's tables, then the chosen pair's panel at full width.

    `body` is the page's main read and `rows` the pairs it shows (not empty): the Wide table lists them, and the left
    table lists `body[grouping.totals]`.
    """
    switch, beside = st.columns([2, 5], vertical_alignment="center")
    with switch:
        view = _switch(layout)
    if caption:
        beside.caption(caption)
    if view == WIDE:
        ids = _ids(rows)
        at = pick_row(
            layout.frame(rows), ids, key=f"{layout.page}_wide", remembered=pair_key(layout.page),
            formats=layout.formats, column_order=layout.wide_columns, column_config=layout.columns(body),
            height=HEIGHT,
        )
        row = rows[at]
    else:
        row = _grouped(layout, GROUPINGS[view], body)
        if row is None:
            return
    with st.container(border=True, key=f"{layout.page}_panel_{row['hs6']}_{row['iso3']}"):
        layout.panel(row)


def _ids(rows: list[dict]) -> list[object]:
    return [(row["hs6"], row["iso3"]) for row in rows]


def _grouped(layout: Layout, grouping: Grouping, body: dict) -> dict | None:
    """The left table of groups and the right table of the chosen group's pairs; the chosen pair's row (None when the
    group read finds none)."""
    groups = body[grouping.totals]
    codes = [group[grouping.code] for group in groups]
    count_help, total_help = layout.left_helps(body)
    left, right = st.columns([1, 2], gap="large")
    with left:
        g = pick_row(
            pd.DataFrame(
                {
                    grouping.heading: [group[grouping.name] for group in groups],
                    layout.count_heading: [group[layout.count] for group in groups],
                    layout.total_heading: millions(group[layout.total] for group in groups),
                }
            ),
            codes, key=f"{layout.page}_groups_{grouping.code}", remembered=f"{layout.page}_group_{grouping.code}",
            formats={layout.count_heading: "{:,}", layout.total_heading: ONE_DECIMAL},
            column_config={
                grouping.heading: grouping.column(),
                layout.count_heading: st.column_config.NumberColumn(layout.count_heading, help=count_help, width="small"),
                layout.total_heading: money_column(layout.total_heading, total_help),
            },
            height=HEIGHT,
        )
        code, name = codes[g], groups[g][grouping.name]
        if layout.under_left:
            layout.under_left(grouping, code, name)
    with right:
        group_body = layout.read_group(grouping, code)
        members = group_body["opportunities"]
        count = len(members)
        st.markdown(
            f"**{name}**: {count:,} {grouping.member if count == 1 else grouping.members}. "
            f"Click one to see its detail below."
        )
        if layout.above_right:
            layout.above_right(grouping, group_body)
        if not members:
            return None
        at = pick_row(
            layout.frame(members), _ids(members), key=f"{layout.page}_members_{grouping.code}_{code}",
            remembered=pair_key(layout.page), formats=layout.formats,
            column_order=[column for column in layout.right_columns if column != grouping.heading],
            column_config=layout.columns(body), height=HEIGHT - 40,
        )
    return members[at]


def _switch(layout: Layout) -> str:
    """The view switch, its value kept for the session and, with `url_param`, in the URL."""
    key = view_key(layout.page)
    st.session_state.setdefault(key, VIEWS[0])
    if layout.url_param:
        _read_url(layout, key)
    view = st.segmented_control(
        "View", list(VIEWS), key=key, required=True, label_visibility="collapsed", persist_state="session"
    )
    if layout.url_param:
        _write_url(layout, view)
    return view


def _written(layout: Layout) -> str:
    """Where a page keeps the view value it last wrote to the URL."""
    return f"_{layout.page}_view_url"


def _read_url(layout: Layout, key: str) -> None:
    """Set the view from the URL when it names one this page did not write itself."""
    assert layout.url_param
    value = st.query_params.get(layout.url_param)
    if value is None or value == st.session_state.get(_written(layout)):
        return
    views = {_URL_VALUES.get(view): view for view in VIEWS}
    if value in views:
        st.session_state[key] = views[value]


def _write_url(layout: Layout, view: str) -> None:
    """Keep the view in the URL, unless it is the default."""
    assert layout.url_param
    value = None if view == VIEWS[0] else _URL_VALUES.get(view)
    if value is None:
        if layout.url_param in st.query_params:
            del st.query_params[layout.url_param]
    elif st.query_params.get(layout.url_param) != value:
        st.query_params[layout.url_param] = value
    st.session_state[_written(layout)] = value
