"""Shared page furniture: cached API reads, the persistent banner, pickers and the ranking table.

Both ranking pages render through `ranking_filters` and `show_ranking`, so a ranking option,
column or row link is added here once.
"""

import pandas as pd
import streamlit as st

from app import client

# HS2 chapters of the 2012 edition (chapter 77 is reserved and unused).
SECTORS = [f"{chapter:02d}" for chapter in range(1, 98) if chapter != 77]
_VALUES = {"export_potential": "Export Potential", "actual_exports": "Actual Exports", "export_gap": "Export Gap"}
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
    """Sector, region and minimum-Gap filters, shared by both ranking pages.

    Returns the API query parameters that are set (`sector`, `region`, `min_gap_gbp`).
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
    min_gap = gap_column.number_input("Minimum Export Gap (£)", value=None, step=1_000_000.0, key="min_gap")
    params = {"sector": sector, "region": region, "min_gap_gbp": min_gap}
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
    descriptions = {product["hs6"]: product["description"] for product in found}
    return st.selectbox(
        "Product",
        list(descriptions),
        format_func=lambda hs6: f"{hs6} – {descriptions[hs6]}",
        key="hs6",
    )


def market_picker(region: str | None = None) -> str | None:
    """Pick a Market by name, from one region if given. Returns its ISO3 code, or None."""
    markets = [m for m in fetch("/markets", client.api_url()) if region is None or m["region"] == region]
    names = {market["iso3"]: market["name"] for market in markets}
    return st.selectbox(
        "Market", list(names), index=None, format_func=names.get, placeholder="Choose a Market", key="iso3"
    )


def show_ranking(body: dict, columns: dict[str, str], file_name: str) -> None:
    """The ranking table with its £ rate, a currency switch and a CSV export of exactly that table.

    `columns` maps the row fields that name each row (e.g. `market`) to their headings.
    """
    rate = body["gbp_rate"]
    currency = st.radio("Values in", list(_CURRENCIES), horizontal=True, key="currency")
    suffix, unit = _CURRENCIES[currency]
    st.caption(
        f"Ranked by Export Gap, Baseline year {body['baseline_year']}. "
        f"£ values use £1 = ${rate['usd_per_gbp']:.4f}, the {rate['year']} annual average "
        f"({rate['source']})."
    )
    rows = body["opportunities"]
    if not rows:
        st.info("No Opportunities match these filters.")
        return
    raw = pd.DataFrame(rows)
    table = raw[list(columns)].rename(columns=columns)
    table["UK Trade Agreement in force"] = raw["trade_agreement"].map({True: "Yes", False: "No"})
    for field, heading in _VALUES.items():
        table[f"{heading} ({unit})"] = raw[f"{field}_{suffix}"]
    st.dataframe(
        table,
        hide_index=True,
        column_config={f"{heading} ({unit})": st.column_config.NumberColumn(format="localized") for heading in _VALUES.values()},
    )
    st.download_button(
        "Download CSV", table.to_csv(index=False), file_name=f"{file_name}.csv", mime="text/csv", key="csv"
    )
