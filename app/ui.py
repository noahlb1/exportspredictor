"""Shared page furniture: cached API reads, the persistent banner, pickers and the ranking table.

Both ranking pages render through `ranking_filters` and `show_ranking`, so a ranking option,
column or row link is added here once.
"""

import json
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from app import client

# HS2 chapters of the 2012 edition (chapter 77 is reserved and unused).
SECTORS = [f"{chapter:02d}" for chapter in range(1, 98) if chapter != 77]
_VALUES = {
    "export_potential": "Export Potential",
    "actual_exports": "Actual Exports",
    "export_gap": "Export Gap",
    "blocked_effect": "Blocked Opportunity value",
}
_CURRENCIES = {"GBP": ("gbp", "£"), "USD": ("usd", "USD")}


@st.cache_data(ttl=300, show_spinner=False)
def fetch(path: str, api_url: str, **params) -> dict | list:
    """Cached API read. `api_url` is part of the cache key so a changed API is re-read."""
    del api_url
    return client.get(path, **params)


def banner() -> dict | None:
    """Show the Baseline-year and caveat banner. Returns health, or None if the API is down."""
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
        "Figures are associations the model has learned, not causes. "
        f"Model version {health['artefact_version']}."
    )
    return health


def ranking_filters() -> dict:
    """The ranking order and the Sector, region and minimum-Gap filters, shared by both ranking pages.

    Returns the API query parameters that are set (`rank_by`, `sector`, `region`, `min_gap_gbp`).
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
    min_gap_gbp = gap_column.number_input("Minimum Export Gap (£)", value=None, step=1_000_000.0, key="min_gap_gbp")
    orders = rank_orders()
    rank_by = st.radio(
        "Rank by", list(orders), format_func=lambda name: orders[name]["title"], key="rank_by", horizontal=True
    )
    params = {"rank_by": rank_by, "sector": sector, "region": region, "min_gap_gbp": min_gap_gbp}
    return {name: value for name, value in params.items() if value is not None}


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


def currency_choice() -> tuple[str, str]:
    """The £/USD switch every page with money values shows (£ first). Returns (field suffix, unit label)."""
    currency = st.radio("Values in", list(_CURRENCIES), horizontal=True, key="currency")
    return _CURRENCIES[currency]


def rate_caption(rate: dict) -> str:
    """The sentence stating the USD/GBP rate a response's £ values use."""
    return (
        f"£ values use £1 = ${rate['usd_per_gbp']:.4f}, the {rate['year']} annual average ({rate['source']})."
    )


def market_picker(region: str | None = None) -> str | None:
    """Pick a Market by name, from one region if given. Returns its ISO3 code, or None."""
    markets = [m for m in fetch("/markets", client.api_url()) if region is None or m["region"] == region]
    names = {market["iso3"]: market["name"] for market in markets}
    return st.selectbox(
        "Market", list(names), index=None, format_func=names.get, placeholder="Choose a Market", key="iso3"
    )


def show_ranking(body: dict, columns: dict[str, str], file_name: str, *, hs6: str | None = None, iso3: str | None = None) -> None:
    """The ranking table with its £ rate, a currency switch, a Why? link per row and a CSV export.

    `columns` maps the row fields that name each row (e.g. `market`) to their headings. The
    ranking's fixed Product (`hs6`) or Market (`iso3`) completes each row's Why? link. The CSV
    holds exactly the table's data columns (the Why? links are page navigation, not data).
    """
    suffix, unit = currency_choice()
    order = rank_orders()[body["rank_by"]]
    st.caption(f"Ranked by {order['title']}, Baseline year {body['baseline_year']}. {rate_caption(body['gbp_rate'])}")
    st.caption(f"{order['title']}: {order['description']}")
    rows = body["opportunities"]
    if not rows:
        st.info("No Opportunities match these filters.")
        return
    raw = pd.DataFrame(rows)
    table = raw[list(columns)].rename(columns=columns)
    table["UK Trade Agreement in force"] = raw["trade_agreement"].map({True: "Yes", False: "No"})
    for field, heading in _VALUES.items():
        table[f"{heading} ({unit})"] = raw[f"{field}_{suffix}"]
    table["Blocked Opportunity"] = raw["blocked_opportunity"]
    links = [why_link(hs6 or row["hs6"], iso3 or row["iso3"]) for row in rows]
    st.dataframe(
        table.assign(**{"Why?": links}),
        hide_index=True,
        column_config={
            **{f"{heading} ({unit})": st.column_config.NumberColumn(format="localized") for heading in _VALUES.values()},
            "Why?": why_column(),
        },
    )
    st.download_button(
        "Download CSV", table.to_csv(index=False), file_name=f"{file_name}.csv", mime="text/csv", key="csv"
    )


WHY_PAGE_PATH = "/Why"  # the URL path Streamlit gives app/pages/4_Why.py


def why_link(hs6: str, iso3: str) -> str:
    """The Why? page URL for one Product–Market pair. Every ranking row links here."""
    return f"{WHY_PAGE_PATH}?{urlencode({'hs6': hs6, 'iso3': iso3})}"


def why_column() -> "st.column_config.LinkColumn":
    """The ranking-table column config for the `Why?` links built by `why_link`."""
    return st.column_config.LinkColumn(
        "Why?", display_text="Why?", help="Which factors raised or lowered this pair's Export Potential"
    )
