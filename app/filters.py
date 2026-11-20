"""The sidebar filters of the two ranking pages, each kept in the page's URL and for the session.

- The Leaderboard (`leaderboard_filters`, spec #103 "Leaderboard"): Product search, Market, Sector, Market region,
  Opportunity Type, UK Trade Agreement in force, minimum Export Gap, the hide filters and Rank by, then Reset filters.
- Blocked Opportunities (`blocked_filters`, spec #103 "Blocked Opportunities"): Product search, Sector, Market region,
  minimum Blocked Opportunity value, the hide filters, then Reset filters.

There is one hide filter per Caveat Tier the API serves (`/caveats`, e.g. Hide Re-export Hubs), keyed and kept in the
URL under the tier's `hide_param`.

Each filter is kept in the page's URL under its own query parameter (`_params`, `_blocked_params`; a default is left
out), so a filtered view, or one pair, can be shared as a link. The Product parameter is the HS6 code and the Market
parameter the ISO3 code. The URL is read and written here by hand, not with Streamlit's `bind="query-params"`, because
a bound selectbox writes its displayed label (a Market's name), not its code.

- A URL that names any filter, and that this page did not write itself (a shared link, the browser, another page's
  link to a pair), sets every filter: the ones it names to its values, the rest to their defaults.
- A URL that names none (a return from another page through the menu) leaves the filters as they were: they persist
  for the session, and the page writes them back to the URL.

Show all (`show_all`, the toggle above the Leaderboard's table) is kept in the URL the same way, though it is not a
filter.

The Leaderboard's widget keys are the parameter names (plus `product_query`, the search text, which is not in the
URL). The Blocked Opportunities page keeps its own filter state: its keys carry `BLOCKED_KEY_PREFIX`
(`blocked_sector`, ...), while its URL uses the same parameter names.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import streamlit as st

from app import client
from app.format import format_usd
from app.ui import (
    ROWS_SHOWN,
    caveat_tiers,
    fetch,
    labels,
    rank_orders,
    read_limit,
    term,
    usd_unit,
)

QUERY = "product_query"  # the Product search text; the chosen Product's HS6 code is the `product` parameter
# Show all: every filtered row read (up to `read_limit()`) instead of the top `ROWS_SHOWN`; not an API filter
SHOW_ALL = "show_all"
# The prefix of the Blocked Opportunities page's widget keys, apart from the Leaderboard's
BLOCKED_KEY_PREFIX = "blocked_"
_AGREEMENT_CHOICES = {None: "Any", True: "Yes", False: "No"}
_YES_NO = {"yes": True, "no": False}
_TRUE = "true"


def _number(value: float) -> str:
    return f"{value:g}"


@dataclass(frozen=True)
class Param:
    """One filter kept in the URL: its query parameter name (and widget key, after the page's prefix), and how its
    value is written there."""

    name: str
    to_url: Callable[[Any], str] = str
    from_url: Callable[[str], Any] = str


def _flag(name: str) -> Param:
    """A switch kept in the URL as `name=true` while on (and left out while off)."""
    return Param(name, to_url=lambda value: _TRUE, from_url=lambda text: text == _TRUE)


_PRODUCT = Param("product")  # HS6 code
_SECTOR = Param("sector")  # HS2 chapter
_REGION = Param("region")


def _hides() -> dict[str, bool]:
    """Each Caveat Tier's hide filter (`/caveats`), by its parameter, at its default: off."""
    return {tier["hide_param"]: False for tier in caveat_tiers()}


def _params() -> list[Param]:
    """The Leaderboard's URL parameters, each Caveat Tier's hide filter among them."""
    return [
        _PRODUCT,
        Param("market"),  # ISO3 code
        _SECTOR,
        _REGION,
        Param("opportunity_type"),
        Param("trade_agreement", to_url=lambda value: "yes" if value else "no", from_url=lambda text: _YES_NO[text]),
        Param("min_gap_usd", to_url=_number, from_url=float),
        *(_flag(name) for name in _hides()),
        Param("rank_by"),
        Param("min_potential_usd", to_url=_number, from_url=float),
        _flag(SHOW_ALL),
    ]


def _blocked_params() -> list[Param]:
    """The Blocked Opportunities page's URL parameters."""
    return [
        _PRODUCT,
        _SECTOR,
        _REGION,
        Param("min_value_usd", to_url=_number, from_url=float),  # minimum Blocked Opportunity value
        *(_flag(name) for name in _hides()),
    ]


def _defaults() -> dict[str, Any]:
    """Every Leaderboard filter's default, the search text's included: the full ranking by the API's first order."""
    orders = rank_orders()
    minimums = [order["default_min_potential_usd"] for order in orders.values() if order.get("min_potential_usd_options")]
    return {
        QUERY: "",
        "product": None,
        "market": None,
        "sector": None,
        "region": None,
        "opportunity_type": None,
        "trade_agreement": None,
        "min_gap_usd": None,
        **_hides(),
        "rank_by": next(iter(orders)),
        "min_potential_usd": minimums[0] if minimums else None,
        SHOW_ALL: False,
    }


def _blocked_defaults() -> dict[str, Any]:
    """Every Blocked Opportunities filter's default (unprefixed): every Blocked Opportunity."""
    return {
        QUERY: "",
        "product": None,
        "sector": None,
        "region": None,
        "min_value_usd": None,
        **_hides(),
    }


def _keyed(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {prefix + name: value for name, value in values.items()}


def reset() -> None:
    """Reset the Leaderboard's filters: write every default (the button's callback, before the widgets are drawn)."""
    st.session_state.update(_defaults())


def reset_blocked() -> None:
    """Reset the Blocked Opportunities page's filters, as `reset` does the Leaderboard's."""
    st.session_state.update(_keyed(_blocked_defaults(), BLOCKED_KEY_PREFIX))


def _written(prefix: str) -> str:
    """Where a page keeps the parameters it last wrote to the URL."""
    return f"_{prefix or 'leaderboard_'}url"


def _read_url(params: list[Param], defaults: dict[str, Any], prefix: str = "") -> None:
    """Set the filters from the URL when it names any filter and this page did not write it (see the module doc)."""
    url = {param.name: st.query_params[param.name] for param in params if param.name in st.query_params}
    if not url or url == st.session_state.get(_written(prefix)):
        return
    values = dict(defaults)
    for param in params:
        if param.name in url:
            try:
                values[param.name] = param.from_url(url[param.name])
            except (KeyError, ValueError):
                pass  # an unreadable value keeps the default
    values[QUERY] = values["product"] or ""
    st.session_state.update(_keyed(values, prefix))


def _write_url(params: list[Param], values: dict[str, Any], defaults: dict[str, Any], prefix: str = "") -> None:
    """Keep each filter that is not at its default in the URL, and only those."""
    written = {
        param.name: param.to_url(values[param.name])
        for param in params
        if values[param.name] is not None and values[param.name] != defaults[param.name]
    }
    for param in params:
        if param.name in written:
            if st.query_params.get(param.name) != written[param.name]:
                st.query_params[param.name] = written[param.name]
        elif param.name in st.query_params:
            del st.query_params[param.name]
    st.session_state[_written(prefix)] = written


def _start(params: list[Param], defaults: dict[str, Any], prefix: str = "") -> None:
    """Before the widgets: give each filter its default the first time, then let a new URL set them."""
    for name, value in _keyed(defaults, prefix).items():
        st.session_state.setdefault(name, value)
    _read_url(params, defaults, prefix)


@dataclass(frozen=True)
class Filters:
    """The filters in force: the API query parameters that are set, and how the active-filters line names each."""

    params: dict[str, Any]
    named: list[str]  # e.g. "Market: Japan"; empty when no filter is in force
    product_found: bool  # False when the search text matches no Product


def leaderboard_filters() -> Filters:
    """Draw the Leaderboard's filters in the sidebar, kept in the URL and for the session; return those in force."""
    defaults, url_params = _defaults(), _params()
    _start(url_params, defaults)
    with st.sidebar:
        st.header("Filters")
        product, found = _product_search()
        market, region = _market_and_region()
        sector, sectors = _sector()
        opportunity_type = st.selectbox(
            "Opportunity Type",
            [None, *labels()["opportunity_types"]],
            format_func=lambda kind: "All Opportunity Types" if kind is None else kind,
            key="opportunity_type",
            help=term("opportunity_type"),
            persist_state="session",
        )
        trade_agreement = st.selectbox(
            "UK Trade Agreement in force", list(_AGREEMENT_CHOICES), format_func=_AGREEMENT_CHOICES.get,
            key="trade_agreement", persist_state="session",
        )
        min_gap_usd = st.number_input(
            f"Minimum Export Gap ({usd_unit()})", value=None, step=1_000_000.0, key="min_gap_usd",
            persist_state="session",
        )
        hides = _hide()
        orders = rank_orders()
        rank_by = st.radio(
            "Rank by", list(orders), format_func=lambda name: orders[name]["title"], key="rank_by",
            persist_state="session",
        )
        options = orders[rank_by].get("min_potential_usd_options")
        min_potential_usd = None
        if options:
            min_potential_usd = st.selectbox(
                f"Only pairs with Export Potential of at least ({usd_unit()})",
                options,
                format_func=format_usd,
                key="min_potential_usd",
                help="Every pair with no Actual Exports has a Relative Export Gap of 100%, so small pairs are left out.",
                persist_state="session",
            )
        st.button("Reset filters", key="reset_filters", on_click=reset)

    values = {
        "product": product,
        "market": market,
        "sector": sector,
        "region": region,
        "opportunity_type": opportunity_type,
        "trade_agreement": trade_agreement,
        "min_gap_usd": min_gap_usd,
        **hides,
        "rank_by": rank_by,
        # Kept (and in the URL) only while the order uses it
        "min_potential_usd": min_potential_usd if options else defaults["min_potential_usd"],
        SHOW_ALL: st.session_state[SHOW_ALL],  # its control sits above the table (`show_all`)
    }
    _write_url(url_params, values, defaults)
    params = {
        **{
            name: value
            for name, value in values.items()
            if value is not None and name not in ("min_potential_usd", SHOW_ALL)
        },
        "min_potential_usd": min_potential_usd,
        **{name: on or None for name, on in hides.items()},
    }
    named = _named(values, sectors, orders, options)
    return Filters({name: value for name, value in params.items() if value is not None}, named, found)


def blocked_filters() -> Filters:
    """Draw the Blocked Opportunities page's filters in the sidebar, under its own keys, kept in the URL and for the
    session; return those in force. There is no Market, order, Opportunity Type or Trade Agreement filter."""
    defaults, url_params = _blocked_defaults(), _blocked_params()
    _start(url_params, defaults, BLOCKED_KEY_PREFIX)
    with st.sidebar:
        st.header("Filters")
        product, found = _product_search(BLOCKED_KEY_PREFIX)
        sector, sectors = _sector(BLOCKED_KEY_PREFIX)
        markets = fetch("/markets", client.api_url())
        region = st.selectbox(
            "Market region", [None, *sorted({market["region"] for market in markets})],
            format_func=lambda r: "All regions" if r is None else r, key=f"{BLOCKED_KEY_PREFIX}region",
            persist_state="session",
        )
        min_value_usd = st.number_input(
            f"Minimum Blocked Opportunity value ({usd_unit()})", value=None, step=1_000_000.0,
            key=f"{BLOCKED_KEY_PREFIX}min_value_usd", persist_state="session",
        )
        hides = _hide(BLOCKED_KEY_PREFIX)
        st.button("Reset filters", key=f"{BLOCKED_KEY_PREFIX}reset_filters", on_click=reset_blocked)

    values = {
        "product": product,
        "sector": sector,
        "region": region,
        "min_value_usd": min_value_usd,
        **hides,
    }
    _write_url(url_params, values, defaults, BLOCKED_KEY_PREFIX)
    params = {**values, **{name: on or None for name, on in hides.items()}}
    named = []
    if product:
        named.append(f"Product: {product}")
    if sector:
        named.append(f"Sector: {sectors.get(sector, sector)}")
    if region:
        named.append(f"Market region: {region}")
    if min_value_usd is not None:
        named.append(f"Blocked Opportunity value at least {format_usd(min_value_usd)}")
    named += _hidden(hides)
    return Filters({name: value for name, value in params.items() if value is not None}, named, found)


def show_all(pairs: int) -> bool:
    """The Show all control, for a filtered ranking of `pairs` pairs: whether to show every row read.

    It is drawn only when the ranking has more than `ROWS_SHOWN` pairs and no more than `read_limit()`; outside that
    range it disappears and turns off (in the session and the URL). It is kept in the URL like the filters, and Reset
    filters switches it off.
    """
    if ROWS_SHOWN < pairs <= read_limit():
        return st.toggle(f"Show all {pairs:,} pairs", key=SHOW_ALL, persist_state="session")
    if st.session_state.get(SHOW_ALL):
        st.session_state[SHOW_ALL] = False
        if SHOW_ALL in st.query_params:
            del st.query_params[SHOW_ALL]
        written = _written("")
        st.session_state[written] = {
            name: value for name, value in st.session_state.get(written, {}).items() if name != SHOW_ALL
        }
    return False


def _product_search(prefix: str = "") -> tuple[str | None, bool]:
    """Search Products by HS6 code or words and choose one: the chosen HS6 code (or None), and whether any matched.

    Excluded Products are found and marked; choosing one makes the ranking read answer with its reason.
    """
    query = st.text_input(
        "Product search: HS6 code (2012 edition) or words, e.g. 851712 or smartphone", key=prefix + QUERY,
        persist_state="session",
    ).strip()
    if not query:
        return None, True
    found = fetch("/products", client.api_url(), q=query)
    if not found:
        st.warning(f"No Product matches '{query}'. Try an HS6 code or another word.")
        return None, False
    descriptions = {
        product["hs6"]: product["description"] + (" (Excluded Product)" if product["excluded"] else "")
        for product in found
    }
    key = f"{prefix}product"
    if st.session_state.get(key) not in descriptions:
        st.session_state[key] = next(iter(descriptions))  # new search text: its best match is chosen
    hs6 = st.selectbox(
        "Product", list(descriptions), format_func=lambda code: f"{code} – {descriptions[code]}", key=key,
        persist_state="session",
    )
    return hs6, True


def _sector(prefix: str = "") -> tuple[str | None, dict[str, str]]:
    """The Sector chosen (HS2 code, or None), and every Sector's name by code."""
    sectors = {sector["code"]: sector["name"] for sector in fetch("/sectors", client.api_url())}
    sector = st.selectbox(
        "Sector", [None, *sectors], format_func=lambda code: "All Sectors" if code is None else sectors[code],
        key=f"{prefix}sector", persist_state="session",
    )
    return sector, sectors


def _hide(prefix: str = "") -> dict[str, bool]:
    """One checkbox per Caveat Tier, e.g. Hide Re-export Hubs: whether each is on, by its hide parameter."""
    return {
        tier["hide_param"]: st.checkbox(
            f"Hide {tier['plural']}", key=f"{prefix}{tier['hide_param']}", help=tier["text"], persist_state="session"
        )
        for tier in caveat_tiers()
    }


def _market_and_region() -> tuple[str | None, str | None]:
    """The Market (ISO3 code) and Market region chosen; the Market list keeps to the region."""
    markets = fetch("/markets", client.api_url())
    regions = sorted({market["region"] for market in markets})
    names = {market["iso3"]: market["name"] for market in markets}
    region = st.session_state.get("region")
    in_region = [iso3 for iso3 in sorted(names, key=names.get) if region is None or _region(markets, iso3) == region]
    market = st.selectbox(
        "Market", [None, *in_region], format_func=lambda iso3: "All Markets" if iso3 is None else names[iso3],
        key="market", persist_state="session",
    )
    region = st.selectbox(
        "Market region", [None, *regions], format_func=lambda r: "All regions" if r is None else r, key="region",
        persist_state="session",
    )
    return market, region


def _region(markets: list[dict], iso3: str) -> str:
    return next(market["region"] for market in markets if market["iso3"] == iso3)


def _hidden(values: dict[str, Any]) -> list[str]:
    """The active-filters line's notice of each Caveat Tier hidden, e.g. "Re-export Hubs hidden"."""
    return [f"{tier['plural']} hidden" for tier in caveat_tiers() if values[tier["hide_param"]]]


def _named(values: dict[str, Any], sectors: dict[str, str], orders: dict[str, dict], options: list | None) -> list[str]:
    """How the Leaderboard's active-filters line names each filter in force."""
    markets = {market["iso3"]: market["name"] for market in fetch("/markets", client.api_url())}
    named = []
    if values["product"]:
        named.append(f"Product: {values['product']}")
    if values["market"]:
        named.append(f"Market: {markets.get(values['market'], values['market'])}")
    if values["sector"]:
        named.append(f"Sector: {sectors.get(values['sector'], values['sector'])}")
    if values["region"]:
        named.append(f"Market region: {values['region']}")
    if values["opportunity_type"]:
        named.append(f"Opportunity Type: {values['opportunity_type']}")
    if values["trade_agreement"] is not None:
        named.append(f"UK Trade Agreement in force: {_AGREEMENT_CHOICES[values['trade_agreement']]}")
    if values["min_gap_usd"] is not None:
        named.append(f"Export Gap at least {format_usd(values['min_gap_usd'])}")
    named += _hidden(values)
    if options:
        named.append(
            f"Ranked by {orders[values['rank_by']]['title']}, Export Potential at least "
            f"{format_usd(values['min_potential_usd'])}"
        )
    return named
