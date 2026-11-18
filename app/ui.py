"""Shared page furniture: cached API reads, the persistent banner, pickers and the ranking table.

The Leaderboard, the Blocked Opportunities page and both ranking pages render through `ranking_filters` and `show_ranking`, so a ranking option,
column or row link is added here once.
"""

import json
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from app import client
from app.format import format_usd

# HS2 chapters of the 2012 edition (chapter 77 is reserved and unused).
SECTORS = [f"{chapter:02d}" for chapter in range(1, 98) if chapter != 77]
_VALUES = {
    "export_potential": "Export Potential",
    "actual_exports": "Actual Exports",
    "export_gap": "Export Gap",
    "blocked_opportunity_value": "Blocked Opportunity value",
}
# The values that are Gap Period means; the Blocked Opportunity value is the Baseline year's.
_GAP_PERIOD_MEANS = ("export_potential", "actual_exports", "export_gap")
ROWS_SHOWN = 500  # the top of a Leaderboard-style ranking sent to the page; its summary covers every filtered pair
# The row fields that name each row of the Leaderboard and the Blocked Opportunities page, and their headings.
PAIR_COLUMNS = {"hs6": "HS6", "description": "Product", "sector": "Sector", "market": "Market", "region": "Region"}


OPPORTUNITY_TYPE_HELP = (
    "Expand: the UK already exports the Product to this Market. New Market: the UK exports the Product, "
    "but not to this Market. New Product: the UK exports the Product to no Market (over the Gap Period)."
)
# The Trade Agreement filter: any pair, or only pairs with or without a UK Trade Agreement in force.
_AGREEMENT_CHOICES = {None: "Any", True: "Yes", False: "No"}


@st.cache_data(ttl=300, show_spinner=False)
def fetch(path: str, api_url: str, **params) -> Any:
    """Cached API read. `api_url` is part of the cache key so a changed API is re-read."""
    del api_url
    return client.get(path, **params)


def labels() -> dict:
    """The API's wording and options (`/labels`): unit and caption, Opportunity Types, caveats, last tariff year."""
    return fetch("/labels", client.api_url())


def usd_unit() -> str:
    """The unit in column headings, e.g. "USD, nominal"."""
    return labels()["usd_unit"]


def caveats() -> dict[str, tuple[str, str]]:
    """Caveat badges (ADR 0005, 0007): the row field that sets each, its badge and what it means."""
    return {field: (caveat["badge"], caveat["text"]) for field, caveat in labels()["caveats"].items()}


def banner() -> dict | None:
    """Show the Baseline-year and caveat banner and the page's one currency caption.

    Returns health, or None if the API is down.
    """
    try:
        health = client.get("/health")
    except client.ApiUnavailable as error:
        st.error(
            f"The API is unavailable: {error} "
            "Start everything with `podman compose up`, or for development run `uv run uvicorn api.main:app`. "
            "Then reload this page."
        )
        return None
    st.info(
        f"Baseline year {health['baseline_year']} (provisional data). "
        f"Export Gaps are {health['gap_period_label']}s. "
        "Figures are associations the model has learned, not causes. "
        f"Model version {health['artefact_version']}."
    )
    st.caption(labels()["usd_caption"])
    return health


def ranking_filters(*, market_filter: bool = False, rank_by_toggle: bool = True) -> dict:
    """The ranking order and filters, shared by the Leaderboard and both ranking pages.

    Returns the API query parameters that are set: `rank_by` (with `min_potential_usd`, chosen
    from the API's options, when the order ranks only pairs above a minimum Export Potential),
    `sector`, `region`, `opportunity_type`, `trade_agreement`, `min_gap_usd`, and
    `hide_flagged` / `hide_hubs` / `hide_uncertain` when ticked (all off by default). With `market_filter`
    (the Leaderboard), one Market can be chosen too (`market`). Without `rank_by_toggle` (the
    Blocked Opportunities page, which has one order) no `rank_by` is offered or returned.
    """
    markets = fetch("/markets", client.api_url())
    regions = sorted({market["region"] for market in markets})
    sector_column, region_column, gap_column = st.columns(3)
    sector = sector_column.selectbox(
        "Sector (HS2 chapter)", [None, *SECTORS], format_func=lambda s: "All Sectors" if s is None else s, key="sector"
    )
    region = region_column.selectbox(
        "Market region", [None, *regions], format_func=lambda r: "All regions" if r is None else r, key="region"
    )
    min_gap_usd = gap_column.number_input(
        f"Minimum Export Gap ({usd_unit()})", value=None, step=1_000_000.0, key="min_gap_usd"
    )
    type_column, agreement_column, market_column = st.columns(3)
    opportunity_type = type_column.selectbox(
        "Opportunity Type",
        [None, *labels()["opportunity_types"]],
        format_func=lambda kind: "All Opportunity Types" if kind is None else kind,
        key="opportunity_type",
        help=OPPORTUNITY_TYPE_HELP,
    )
    trade_agreement = agreement_column.selectbox(
        "UK Trade Agreement in force",
        list(_AGREEMENT_CHOICES),
        format_func=_AGREEMENT_CHOICES.get,
        key="trade_agreement",
    )
    market = None
    if market_filter:
        names = {m["iso3"]: m["name"] for m in markets if region is None or m["region"] == region}
        market = market_column.selectbox(
            "Market",
            [None, *sorted(names, key=names.get)],
            format_func=lambda iso3: "All Markets" if iso3 is None else names[iso3],
            key="market",
        )
    caveat = caveats()
    flagged_column, hubs_column, uncertain_column = st.columns(3)
    hide_flagged = flagged_column.checkbox(
        "Hide Flagged Products", key="hide_flagged", help=caveat["flagged_product"][1]
    )
    hide_hubs = hubs_column.checkbox("Hide Re-export Hubs", key="hide_hubs", help=caveat["reexport_hub"][1])
    hide_uncertain = uncertain_column.checkbox(
        "Hide Uncertain Opportunities", key="hide_uncertain", help=caveat["uncertain_opportunity"][1]
    )
    rank_by = min_potential_usd = options = None
    if rank_by_toggle:
        orders = rank_orders()
        order_column, minimum_column = st.columns(2)
        rank_by = order_column.radio(
            "Rank by", list(orders), format_func=lambda name: orders[name]["title"], key="rank_by", horizontal=True
        )
        options = orders[rank_by].get("min_potential_usd_options")
    if options:
        min_potential_usd = minimum_column.selectbox(
            f"Only pairs with Export Potential of at least ({usd_unit()})",
            options,
            index=options.index(orders[rank_by]["default_min_potential_usd"]),
            format_func=format_usd,
            key="min_potential_usd",
            help="Every pair with no Actual Exports has a Relative Export Gap of 100%, so small pairs are left out.",
        )
    params = {
        "rank_by": rank_by,
        "min_potential_usd": min_potential_usd,
        "sector": sector,
        "region": region,
        "market": market,
        "opportunity_type": opportunity_type,
        "trade_agreement": trade_agreement,
        "min_gap_usd": min_gap_usd,
        "hide_flagged": hide_flagged or None,
        "hide_hubs": hide_hubs or None,
        "hide_uncertain": hide_uncertain or None,
    }
    return {name: value for name, value in params.items() if value is not None}


def caveat_badges(row: dict) -> str:
    """The caveat badges of one ranking row or explanation, comma-separated ("" if none)."""
    return ", ".join(badge for field, (badge, _) in caveats().items() if row[field])


def show_caveats(row: dict) -> None:
    """One warning per caveat of a pair (its detail view)."""
    for field, (_, caveat) in caveats().items():
        if row[field]:
            st.warning(caveat)


def product_picker(sector: str | None = None) -> str | None:
    """Search Products by HS6 code or words and pick one. Returns the chosen hs6, or None.

    With a `sector`, only that Sector's Products are offered.
    """
    query = st.text_input(
        "Search Products by HS6 code (2012 edition) or words, e.g. 851712 or smartphone",
        key="product_query",
    ).strip()
    if not query:
        return None
    try:
        found = fetch("/products", client.api_url(), q=query)
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
        return None
    found = [product for product in found if sector is None or product["sector"] == sector]
    if not found:
        in_sector = f" in Sector {sector}" if sector else ""
        st.warning(f"No Product matches '{query}'{in_sector}. Try an HS6 code or another word.")
        return None
    # Excluded Products are found and marked; choosing one shows the API's 404 with its reason.
    descriptions = {
        product["hs6"]: product["description"] + (" (Excluded Product)" if product["excluded"] else "")
        for product in found
    }
    return st.selectbox(
        "Product",
        list(descriptions),
        format_func=lambda hs6: f"{hs6} – {descriptions[hs6]}",
        key="hs6",
    )


def excluded_products_list() -> None:
    """The Excluded Products and the reason for each, in an expander under the ranking filters."""
    excluded = fetch("/products/excluded", client.api_url())
    with st.expander(f"Excluded Products ({len(excluded)}): why some Products are never ranked"):
        st.caption(
            "These Products are left out of training, scoring and every ranking, because their recorded "
            "trade follows bullion-market, pipeline or grid geography the model cannot see, or is unreliable. "
            "Their trade still counts in every country and world total."
        )
        table = pd.DataFrame(excluded, columns=["hs6", "description", "sector", "reason"])
        # A static table, so the page's interactive dataframes are only its rankings.
        st.table(
            table.rename(columns={"hs6": "HS6", "description": "Product", "sector": "Sector", "reason": "Reason"})
            .set_index("HS6")
        )


@st.cache_data(ttl=300, show_spinner=False)
def run_scenario(body_json: str, api_url: str) -> dict:
    """Cached Scenario run; `body_json` is the request body as sorted JSON (the cache key)."""
    del api_url
    return client.post("/scenarios", json.loads(body_json))


def rank_orders() -> dict[str, dict]:
    """The ranking orders the API accepts as `rank_by`, by name: title and description."""
    return {order["name"]: order for order in fetch("/opportunities/rank-by", client.api_url())}


def market_picker(region: str | None = None) -> str | None:
    """Pick a Market by name, from one region if given. Returns its ISO3 code, or None."""
    markets = [m for m in fetch("/markets", client.api_url()) if region is None or m["region"] == region]
    names = {market["iso3"]: market["name"] for market in markets}
    return st.selectbox(
        "Market", list(names), index=None, format_func=names.get, placeholder="Choose a Market", key="iso3"
    )


def show_ranking(body: dict, columns: dict[str, str], file_name: str, *, hs6: str | None = None, iso3: str | None = None) -> None:
    """The ranking table in nominal USD, with caveat badges, a Why? link per row and a CSV export.

    Export Potential, Actual Exports, Export Gap and the Relative Export Gap are Gap Period
    means, labelled as such (e.g. "Export Gap, 2022–2024 average (USD, nominal)"); each row
    shows its Opportunity Type and its Potential Range as "low–high".

    `columns` maps the row fields that name each row (e.g. `market`) to their headings. The
    ranking's fixed Product (`hs6`) or Market (`iso3`) completes each row's Why? link. The CSV
    holds exactly the table's data columns (the Why? links are page navigation, not data).
    """
    order = body.get("order") or rank_orders()[body["rank_by"]]  # Blocked Opportunities carry their own
    st.caption(f"Ranked by {order['title']}, Baseline year {body['baseline_year']}. {order['description']}")
    rows = body["opportunities"]
    if not rows:
        st.info("No Opportunities match these filters.")
        return
    raw = pd.DataFrame(rows)
    table = raw[list(columns)].rename(columns=columns)
    table["Opportunity Type"] = raw["opportunity_type"]
    table["UK Trade Agreement in force"] = raw["trade_agreement"].map({True: "Yes", False: "No"})
    averaged, unit = body["gap_period_label"], usd_unit()
    headings = {
        field: f"{heading}, {averaged} ({unit})" if field in _GAP_PERIOD_MEANS else f"{heading} ({unit})"
        for field, heading in _VALUES.items()
    }
    for field, heading in headings.items():
        table[heading] = raw[f"{field}_usd"]
    relative = f"Relative Export Gap, {averaged}"
    table[relative] = raw["relative_export_gap"]
    table[f"Potential Range, {averaged} ({unit})"] = [
        f"{format_usd(row['potential_range_low_usd'])}–{format_usd(row['potential_range_high_usd'])}" for row in rows
    ]
    table["Blocked Opportunity"] = raw["blocked_opportunity"]
    table["Caveats"] = [caveat_badges(row) for row in rows]
    links = [why_link(hs6 or row["hs6"], iso3 or row["iso3"]) for row in rows]
    st.dataframe(
        table.assign(**{"Why?": links}),
        hide_index=True,
        column_config={
            **{heading: st.column_config.NumberColumn(format="localized") for heading in headings.values()},
            relative: st.column_config.NumberColumn(format="percent"),
            "Opportunity Type": st.column_config.TextColumn(help=OPPORTUNITY_TYPE_HELP),
            "Why?": why_column(),
        },
    )
    shown = [caveat for field, (_, caveat) in caveats().items() if raw[field].any()]
    if shown:
        st.caption(" ".join(shown) + " Use the hide filters above to leave them out.")
    st.download_button(
        "Download CSV", table.to_csv(index=False), file_name=f"{file_name}.csv", mime="text/csv", key="csv"
    )


def show_leaderboard(body: dict, file_name: str) -> None:
    """A Leaderboard-style ranking of every UK pair (the Leaderboard, Blocked Opportunities): how much of it
    is shown, the Why? hint and the table, after the page's own summary metrics."""
    summary, shown = body["summary"], len(body["opportunities"])
    if summary["pairs"] > shown:
        st.caption(f"Showing the top {shown:,} of {summary['pairs']:,} pairs.")
    st.caption("Open **Why?** on a row to see which factors raised or lowered its Export Potential.")
    show_ranking(body, PAIR_COLUMNS, file_name=file_name)


WHY_PAGE_PATH = "/Why"  # the URL path Streamlit gives app/pages/5_Why.py


def why_link(hs6: str, iso3: str) -> str:
    """The Why? page URL for one Product–Market pair. Every ranking row links here."""
    return f"{WHY_PAGE_PATH}?{urlencode({'hs6': hs6, 'iso3': iso3})}"


def why_column() -> "st.column_config.LinkColumn":
    """The ranking-table column config for the `Why?` links built by `why_link`."""
    return st.column_config.LinkColumn(
        "Why?", display_text="Why?", help="Which factors raised or lowered this pair's Export Potential"
    )


def potential_range_bar(body: dict) -> None:
    """One pair's Potential Range as a bar, with its Export Potential and Actual Exports marked (Gap Period means)."""
    low, high = body["potential_range_low_usd"], body["potential_range_high_usd"]
    marks = [
        {"marker": "Export Potential", "value_usd": body["export_potential_usd"]},
        {"marker": "Actual Exports", "value_usd": body["actual_exports_usd"]},
    ]
    axis = {"title": f"{body['gap_period_label']} ({usd_unit()})", "format": "~s"}
    st.vega_lite_chart(
        {
            "height": 70,
            "layer": [
                {
                    "data": {"values": [{"marker": "Potential Range", "low_usd": low, "high_usd": high}]},
                    "mark": {"type": "bar", "height": 18, "opacity": 0.35, "tooltip": True},
                    "encoding": {
                        "x": {"field": "low_usd", "type": "quantitative", "axis": axis},
                        "x2": {"field": "high_usd"},
                    },
                },
                {
                    "data": {"values": marks},
                    "mark": {"type": "point", "filled": True, "size": 160, "tooltip": True},
                    "encoding": {
                        "x": {"field": "value_usd", "type": "quantitative"},
                        "color": {"field": "marker", "type": "nominal", "title": None},
                        "shape": {"field": "marker", "type": "nominal", "title": None},
                    },
                },
            ],
        },
        width="stretch",
    )
