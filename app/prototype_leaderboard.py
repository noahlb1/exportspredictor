"""PROTOTYPE, throwaway (wayfinder ticket #95): never merge.

Three layouts of the Leaderboard plus today's, on the Leaderboard's own route, switched with
`?variant=` or the bar pinned to the bottom of the page. The bar also switches the two questions
that cut across layouts: which columns show by default, and what a pair under $50k shows.

    A  Sidebar filters, detail below    filters in the sidebar; a selected row opens under the table
    B  Filter panel, detail drawer      filters in a collapsed panel; a selected row opens a side drawer
    C  Split view                       filters in the sidebar; a three-column table beside an always-open detail
    today                               the current page, for comparison

Run it (PowerShell, from this worktree; the API must be up on :8000):

    $env:EXPORTSPREDICTOR_PROTOTYPE = "1"; ./scripts/wt.ps1 streamlit run app/Home.py

No API change: a chosen Product reads the existing by-Product ranking, and Sector names are a
hard-coded list here. Both would move to the API in the real thing.
"""

import os

import pandas as pd
import streamlit as st

from app import client
from app.format import format_usd
from app.ui import (
    OPPORTUNITY_TYPE_HELP,
    ROWS_SHOWN,
    caveat_badges,
    caveats,
    fetch,
    labels,
    potential_range_bar,
    rank_orders,
    show_caveats,
    why_link,
)

VARIANTS = {
    "A": "Sidebar filters, detail below",
    "B": "Filter panel, detail drawer",
    "C": "Split view",
    "today": "Today's page",
}
COLUMN_SETS = {
    "Short (5)": ["Product", "Market", "Export Gap", "Export Potential", "Actual Exports"],
    "Standard (7)": [
        "Product", "Market", "Export Gap", "Export Potential", "Actual Exports", "Opportunity Type", "Caveats",
    ],
    "Everything (15)": None,
}
SMALL_RULES = ["0.0", "<0.1", "Second decimal"]

# HS 2012 chapter names, shortened.
SECTOR_NAMES = dict(zip(
    [f"{c:02d}" for c in range(1, 98) if c != 77],
    """Live animals|Meat|Fish and seafood|Dairy, eggs and honey|Other animal products|Live plants and flowers
|Vegetables|Fruit and nuts|Coffee, tea and spices|Cereals|Milling products|Oil seeds|Gums and resins
|Other vegetable products|Fats and oils|Meat and fish preparations|Sugar and confectionery|Cocoa
|Cereal and bakery preparations|Vegetable and fruit preparations|Miscellaneous food|Beverages and spirits
|Food residues and animal feed|Tobacco|Salt, stone and cement|Ores|Mineral fuels and oils|Inorganic chemicals
|Organic chemicals|Pharmaceuticals|Fertilisers|Paints and dyes|Perfumery and cosmetics|Soaps and waxes
|Glues and enzymes|Explosives|Photographic goods|Miscellaneous chemicals|Plastics|Rubber|Hides and leather
|Leather goods|Furskins|Wood|Cork|Straw and basketware|Wood pulp|Paper and paperboard|Printed matter|Silk
|Wool|Cotton|Other vegetable fibres|Man-made filaments|Man-made staple fibres|Wadding and rope|Carpets
|Special woven fabrics|Coated textiles|Knitted fabrics|Knitted clothing|Woven clothing|Other made-up textiles
|Footwear|Headgear|Umbrellas|Feathers and artificial flowers|Stone and plaster articles|Ceramics|Glass
|Precious metals and stones|Iron and steel|Iron and steel articles|Copper|Nickel|Aluminium|Lead|Zinc|Tin
|Other base metals|Tools and cutlery|Miscellaneous metal articles|Machinery|Electrical equipment|Railway
|Vehicles|Aircraft|Ships and boats|Optical and medical instruments|Clocks and watches|Musical instruments
|Arms and ammunition|Furniture|Toys and sports goods|Miscellaneous manufactures|Art and antiques"""
    .replace("\n", "").split("|"),
    strict=True,
))
AGREEMENT = {"Any": None, "Yes": True, "No": False}
FILTER_DEFAULTS = {
    "q": "", "product": None, "sector": None, "region": None, "market": None, "type": None,
    "agreement": "Any", "min_gap_m": None, "hide_flagged": False, "hide_hubs": False,
}


def sector_text(code: str) -> str:
    return f"{code} – {SECTOR_NAMES.get(code, '')}"


def reset_filters() -> None:
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value


def filters_ui(where) -> tuple[dict, str | None, list[str]]:
    """Draw the filters into `where`. Returns the API params, the chosen Product and the active-filter phrases."""
    markets = fetch("/markets", client.api_url())
    active = []
    query = where.text_input(
        "Product search", key="q", placeholder="HS6 code or words, e.g. 220830 or whisky",
        help="Replaces the Product → Markets page: pick a Product to rank its Markets.",
    ).strip()
    hs6 = None
    if query:
        found = [p for p in fetch("/products", client.api_url(), q=query) if not p["excluded"]]
        products = {p["hs6"]: p["description"] for p in found}
        if products:
            hs6 = where.selectbox(
                f"Product ({len(products)} match{'es' if len(products) != 1 else ''})", list(products), index=None,
                format_func=lambda code: f"{code} – {products[code]}", placeholder="Choose a Product", key="product",
            )
        else:
            where.caption(f"No Product matches '{query}'.")
    if hs6:
        active.append(f"Product {hs6}")
    sector = None
    if not hs6:
        sector = where.selectbox(
            "Sector", [None, *SECTOR_NAMES], key="sector",
            format_func=lambda s: "All Sectors" if s is None else sector_text(s),
        )
        if sector:
            active.append(f"Sector {sector_text(sector)}")
    regions = sorted({m["region"] for m in markets})
    region = where.selectbox(
        "Market region", [None, *regions], format_func=lambda r: "All regions" if r is None else r, key="region"
    )
    names = {m["iso3"]: m["name"] for m in markets if region is None or m["region"] == region}
    market = where.selectbox(
        "Market", [None, *sorted(names, key=names.get)], key="market",
        format_func=lambda iso3: "All Markets" if iso3 is None else names[iso3],
    )
    kind = where.selectbox(
        "Opportunity Type", [None, *labels()["opportunity_types"]], key="type", help=OPPORTUNITY_TYPE_HELP,
        format_func=lambda k: "All Opportunity Types" if k is None else k,
    )
    agreement = where.selectbox("UK Trade Agreement in force", list(AGREEMENT), key="agreement")
    min_gap_m = where.number_input("Minimum Export Gap ($ millions)", value=None, step=1.0, key="min_gap_m")
    caveat = caveats()
    hide_flagged = where.checkbox("Hide Flagged Products", key="hide_flagged", help=caveat["flagged_product"][1])
    hide_hubs = where.checkbox("Hide Re-export Hubs", key="hide_hubs", help=caveat["reexport_hub"][1])
    orders = rank_orders()
    rank_by = where.radio("Rank by", list(orders), format_func=lambda n: orders[n]["title"], key="rank_by")
    options, min_potential = orders[rank_by].get("min_potential_usd_options"), None
    if options:
        min_potential = where.selectbox(
            "Only pairs with Export Potential of at least", options, format_func=format_usd, key="min_potential",
            index=options.index(orders[rank_by]["default_min_potential_usd"]),
        )
    if region:
        active.append(region)
    if market:
        active.append(names[market])
    if kind:
        active.append(kind)
    if agreement != "Any":
        active.append(f"Trade Agreement: {agreement}")
    if min_gap_m is not None:
        active.append(f"Export Gap ≥ ${min_gap_m:,.1f}m")
    if hide_flagged:
        active.append("Flagged Products hidden")
    if hide_hubs:
        active.append("Re-export Hubs hidden")
    params = {
        "rank_by": rank_by, "min_potential_usd": min_potential, "sector": sector, "region": region,
        "market": market, "opportunity_type": kind, "trade_agreement": AGREEMENT[agreement],
        "min_gap_usd": None if min_gap_m is None else min_gap_m * 1e6,
        "hide_flagged": hide_flagged or None, "hide_hubs": hide_hubs or None,
    }
    return {k: v for k, v in params.items() if v is not None}, hs6, active


def read(params: dict, hs6: str | None) -> dict:
    """The ranking: the Leaderboard read, or (prototype shortcut) the by-Product read reshaped to match."""
    if not hs6:
        return fetch("/opportunities/top", client.api_url(), **params, limit=ROWS_SHOWN)
    market = params.pop("market", None)
    params.pop("sector", None)
    body = dict(fetch(f"/opportunities/by-product/{hs6}", client.api_url(), **params))
    rows = [{**body["product"], **row} for row in body["opportunities"] if market in (None, row["iso3"])]
    body["opportunities"] = rows
    body["summary"] = {
        "pairs": len(rows), "positive_export_gap_usd": sum(max(row["export_gap_usd"], 0) for row in rows)
    }
    return body


def active_line(active: list[str], body: dict, *, reset_here: bool) -> None:
    summary, shown = body["summary"], len(body["opportunities"])
    showing = f"Showing the top {shown:,} of {summary['pairs']:,} pairs" if summary["pairs"] > shown else (
        f"Showing all {shown:,} pairs")
    text = f"**Filters:** {' · '.join(active)}. " if active else "**No filters.** "
    line, button = st.columns([6, 1], vertical_alignment="center")
    line.markdown(
        f"{text}{showing}, with a total positive Export Gap of "
        f"**{format_usd(summary['positive_export_gap_usd'])}**."
    )
    if reset_here and active:
        button.button("Reset filters", on_click=reset_filters, key="reset_inline")


def millions(usd: pd.Series, rule: str):
    value = usd / 1e6
    if rule != "<0.1":
        return value
    return [("<0.1" if 0 < v < 0.05 else f"{v:,.1f}") for v in value]


def ranking_table(body: dict, *, columns: list[str] | None, rule: str, height: int, key: str):
    """The narrow table. Returns the selected row (a dict) or None."""
    rows = body["opportunities"]
    raw = pd.DataFrame(rows)
    averaged = body["gap_period_label"]
    money = {
        "Export Gap": ("export_gap_usd", "Export Potential minus Actual Exports"),
        "Export Potential": ("export_potential_usd", "What the model expects the UK to export, given the fundamentals"),
        "Actual Exports": ("actual_exports_usd", "What the UK exported"),
        "Blocked Opportunity value": (
            "blocked_opportunity_value_usd",
            f"{labels()['illustrative']['badge']}: the Scenario Effect of adding a UK Trade Agreement",
        ),
    }
    table = pd.DataFrame({
        "Product": raw["description"],
        "Market": raw["market"],
        "Export Gap": millions(raw["export_gap_usd"], rule),
        "Export Potential": millions(raw["export_potential_usd"], rule),
        "Actual Exports": millions(raw["actual_exports_usd"], rule),
        "Opportunity Type": raw["opportunity_type"],
        "Caveats": [caveat_badges(row) for row in rows],
        "Relative Export Gap": raw["relative_export_gap"],
        "Potential Range": [
            f"{row['potential_range_low_usd'] / 1e6:,.1f}–{row['potential_range_high_usd'] / 1e6:,.1f}" for row in rows
        ],
        "HS6": raw["hs6"],
        "Sector": raw["sector"].map(sector_text),
        "Region": raw["region"],
        "UK Trade Agreement": raw["trade_agreement"].map({True: "Yes", False: "No"}),
        "Blocked Opportunity value": millions(raw["blocked_opportunity_value_usd"], rule),
        "Blocked Opportunity": raw["blocked_opportunity"],
    })
    number = "%.2f" if rule == "Second decimal" else "%.1f"
    config = {
        "Product": st.column_config.TextColumn(width="large", pinned=True),
        "Opportunity Type": st.column_config.TextColumn("Type", help=OPPORTUNITY_TYPE_HELP),
        "Relative Export Gap": st.column_config.NumberColumn("Relative gap", format="percent"),
        "Potential Range": st.column_config.TextColumn(
            help="Where Actual Exports fall 80% of the time for comparable pairs"
        ),
    }
    for heading, (_, tip) in money.items():
        config[heading] = (
            st.column_config.TextColumn(help=tip) if rule == "<0.1"
            else st.column_config.NumberColumn(help=tip, format=number, width="small")
        )
    st.caption(
        f"Amounts are **$ millions** ({labels()['usd_unit']}), {averaged}. "
        "Select a row to see which factors raised or lowered its Export Potential. "
        "More columns: the table's toolbar, *Show/hide columns*."
    )
    event = st.dataframe(
        table, hide_index=True, height=height, column_order=columns, column_config=config,
        on_select="rerun", selection_mode="single-row", key=key,
    )
    full = raw.drop(columns=["flagged_product", "reexport_hub", "uncertain_opportunity"], errors="ignore")
    st.download_button(
        "Download CSV (full dollars, every column)", full.to_csv(index=False), "leaderboard.csv", "text/csv",
        key=f"csv_{key}",
    )
    picked = event.selection.rows
    return rows[picked[0]] if picked else None


def detail(row: dict) -> None:
    """One pair's Why?, in place."""
    body = fetch(f"/explain/{row['hs6']}/{row['iso3']}", client.api_url())
    st.subheader(f"{row['description']} → {row['market']}")
    st.caption(f"HS6 {row['hs6']} · Sector {sector_text(row['sector'])} · {row['region']} · {body['opportunity_type']}")
    show_caveats(body)
    potential, actual, gap = st.columns(3)
    potential.metric("Export Potential", format_usd(body["export_potential_usd"]))
    actual.metric("Actual Exports", format_usd(body["actual_exports_usd"]))
    gap.metric("Export Gap", format_usd(body["export_gap_usd"]))
    potential_range_bar(body)
    top = body["factors"][:8]
    for title, direction in (("higher", "raised"), ("lower", "lowered")):
        found = [f for f in top if f["direction"] == direction][:3]
        if found:
            st.markdown(
                f"**The model associates these most with {title} Export Potential:** "
                + "; ".join(f"{f['label']} ({f['value_text']})" for f in found)
            )
    st.caption("These are associations the model has learned, not causes.")
    st.markdown(f"[Full Why? page: every factor, and year by year]({why_link(row['hs6'], row['iso3'])})")


def header() -> None:
    st.header("Leaderboard")
    st.markdown(
        "Every UK Product–Market pair, ranked by how far UK goods exports fall short of what the UK's "
        "fundamentals suggest."
    )


def variant_a(columns, rule) -> dict:
    st.sidebar.subheader("Filters")
    params, hs6, active = filters_ui(st.sidebar)
    st.sidebar.button("Reset filters", on_click=reset_filters, disabled=not active)
    body = read(dict(params), hs6)
    header()
    active_line(active, body, reset_here=False)
    if not body["opportunities"]:
        st.info("No Opportunities match these filters.")
        return params
    row = ranking_table(body, columns=columns, rule=rule, height=390, key="table_a")
    with st.container(border=True):
        if row:
            detail(row)
        else:
            st.markdown("**Why?** Select a row above to see which factors raised or lowered its Export Potential.")
    return params


def _close_drawer() -> None:
    st.session_state["drawer_n"] = st.session_state.get("drawer_n", 0) + 1  # a new table key clears the selection


@st.dialog("Why?", width="large", position="right", on_dismiss=_close_drawer)
def drawer(row: dict) -> None:
    detail(row)


def variant_b(columns, rule) -> dict:
    count = sum(
        st.session_state.get(key) not in (default, None) for key, default in FILTER_DEFAULTS.items() if key != "q"
    )
    panel = st.expander(f"Filters and ranking order ({count} active)" if count else "Filters and ranking order")
    left, right = panel.columns(2)
    # One call draws every control, so the panel is crude: all controls on the left, the reset on the right.
    params, hs6, active = filters_ui(left)
    right.button("Reset filters", on_click=reset_filters, disabled=not active)
    right.caption("In the real page these controls would sit in three columns; the prototype draws them in one.")
    body = read(dict(params), hs6)
    header()
    active_line(active, body, reset_here=True)
    if not body["opportunities"]:
        st.info("No Opportunities match these filters.")
        return params
    key = f"table_b_{st.session_state.get('drawer_n', 0)}"
    row = ranking_table(body, columns=columns, rule=rule, height=560, key=key)
    if row:
        drawer(row)
    return params


def variant_c(columns, rule) -> dict:
    del columns  # the split view fixes its own three columns
    st.sidebar.subheader("Filters")
    params, hs6, active = filters_ui(st.sidebar)
    st.sidebar.button("Reset filters", on_click=reset_filters, disabled=not active)
    body = read(dict(params), hs6)
    header()
    active_line(active, body, reset_here=False)
    if not body["opportunities"]:
        st.info("No Opportunities match these filters.")
        return params
    table_side, detail_side = st.columns([5, 4], gap="large")
    with table_side:
        row = ranking_table(body, columns=["Product", "Market", "Export Gap"], rule=rule, height=620, key="table_c")
    with detail_side, st.container(border=True):
        if row is None:
            st.caption("The top pair is shown until you select a row.")
        detail(row or body["opportunities"][0])
    return params


def switcher() -> tuple[str, list[str] | None, str]:
    """The prototype bar, pinned to the bottom of the page. Not part of any design."""
    keys = list(VARIANTS)
    if st.session_state.get("variant") not in keys:
        asked = st.query_params.get("variant")
        st.session_state["variant"] = asked if asked in keys else "A"

    def step(by: int) -> None:
        st.session_state["variant"] = keys[(keys.index(st.session_state["variant"]) + by) % len(keys)]

    with st.bottom, st.container(border=True):
        back, pick, forward, cols, small = st.columns([1, 9, 1, 3, 3], vertical_alignment="bottom")
        back.button("←", on_click=step, args=(-1,), key="proto_back")
        variant = pick.segmented_control(
            "PROTOTYPE: layout", keys, format_func=lambda k: f"{k} · {VARIANTS[k]}" if k != "today" else VARIANTS[k],
            key="variant", required=True,
        )
        forward.button("→", on_click=step, args=(1,), key="proto_forward")
        column_set = cols.selectbox("Default columns (A, B)", list(COLUMN_SETS), index=1, key="proto_columns")
        rule = small.selectbox("A pair under $50k shows", SMALL_RULES, key="proto_small")
    st.query_params["variant"] = variant
    return variant, COLUMN_SETS[column_set], rule


def render() -> bool:
    """Draw the chosen variant. False means: draw today's page instead."""
    if not os.environ.get("EXPORTSPREDICTOR_PROTOTYPE"):  # off unless asked for, so the app tests see today's page
        return False
    variant, columns, rule = switcher()
    if variant == "today":
        return False
    params = {"A": variant_a, "B": variant_b, "C": variant_c}[variant](columns, rule)
    with st.expander("PROTOTYPE state: what this view asked the API for"):
        st.json({"variant": variant, "params": params, "columns": columns, "small_value_rule": rule})
    return True
