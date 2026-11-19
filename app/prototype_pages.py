"""PROTOTYPE, throwaway (wayfinder ticket #100): never merge.

How far the Leaderboard's settled layout carries to the other three pages. The app runs under the
settled flat menu (Leaderboard, Blocked Opportunities, Scenarios, Model Evaluation); each of the three
other pages has three layouts plus today's, switched with `?variant=` or the bar pinned to the bottom.

    Blocked Opportunities   A  Split view, Barrier first          the Leaderboard's layout, ranked by Blocked Opportunity value
                            B  Wide table, Leaderboard explains   no panel; a selected row links to the Leaderboard
                            C  Grouped by Market                  Markets on the left, the chosen Market's Products on the right
    Scenarios               A  Builder in the sidebar, tabs       the controls where the Leaderboard's filters sit
                            B  Builder on top, sections below     today's order; a clicked Market row is the drill-down
                            C  Split view by Market               Markets beside the chosen Market's most affected pairs
    Model Evaluation        A  One long page, contents list       today's order, with links in the sidebar
                            B  Tabs                               the headline above four tabs
                            C  Scorecard and expanders            four figures, then every section folded

Every variant carries the caution line under the heading and the footer under the content (#98).

Run it (PowerShell, from this worktree; the API must be up on :8000):

    $env:EXPORTSPREDICTOR_PROTOTYPE = "1"; ./scripts/wt.ps1 streamlit run app/Home.py

Shortcuts, each an API change in the real thing: the Blocked Opportunities page reads 6,000 rows and
filters them in the page; Sector names are the hard-coded list of #95; "Explain this pair on the
Leaderboard" sets the Leaderboard's filters in this session instead of building a URL.
"""

import json
import re
import runpy
from pathlib import Path

import pandas as pd
import streamlit as st

from app import client, prototype_banner, prototype_logo
from app import prototype_leaderboard as pl
from app.format import format_change, format_usd, years_text
from app.ui import (
    SECTORS,
    caveat_badges,
    caveats,
    fetch,
    illustrative,
    labels,
    run_scenario,
    scenario_note,
)

NAME = "UK Exports Predictor"
CAUTION = prototype_banner.CAUTION
PAGES_DIR = Path(__file__).parent / "pages"
TINT = "background-color: rgba(42, 120, 214, 0.28); font-weight: 600"
PAGES: dict = {}
ROWS = 500


# ---------------------------------------------------------------- shared pieces


def top(heading: str, intro: str) -> dict | None:
    """Title, heading, one-sentence introduction and the caution line. Returns health, or None if the API is down."""
    st.title(NAME)
    st.header(heading)
    st.markdown(intro)
    st.caption(CAUTION)
    try:
        return client.get("/health")
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
        return None


def footer(health: dict) -> None:
    st.divider()
    st.caption(" ".join(prototype_banner.about(health, everything=True)))


def _kept(key: str, allowed: list[str], default: str) -> None:
    """Widget state is dropped when another page is shown, so the bar's choices are kept in a shadow dict."""
    keep = st.session_state.setdefault("_proto_keep", {})
    if st.session_state.get(key) not in allowed:
        st.session_state[key] = keep[key] if keep.get(key) in allowed else default


def bar(page: str, variants: dict[str, str], extras: dict[str, list[str]]) -> tuple[str, dict[str, str]]:
    """The prototype bar, pinned to the bottom of the page. Not part of any design."""
    keys, vkey = list(variants), f"{page}_variant"
    asked = st.query_params.get("variant")
    _kept(vkey, keys, asked if asked in keys else keys[0])
    for label, options in extras.items():
        _kept(f"{page}_{label}", options, options[0])

    def step(by: int) -> None:
        st.session_state[vkey] = keys[(keys.index(st.session_state[vkey]) + by) % len(keys)]

    with st.bottom, st.container(border=True):
        columns = st.columns([1, 7, 1] + [4] * len(extras), vertical_alignment="bottom")
        columns[0].button("←", on_click=step, args=(-1,), key=f"{page}_back")
        variant = columns[1].selectbox(
            f"PROTOTYPE: {page}", keys, key=vkey,
            format_func=lambda k: variants[k] if k == "today" else f"{k} · {variants[k]}",
        )
        columns[2].button("→", on_click=step, args=(1,), key=f"{page}_forward")
        chosen = {
            label: column.selectbox(label, options, key=f"{page}_{label}")
            for column, (label, options) in zip(columns[3:], extras.items(), strict=True)
        }
    st.session_state["_proto_keep"] |= {vkey: variant} | {f"{page}_{label}": v for label, v in chosen.items()}
    st.query_params["variant"] = variant
    return variant, chosen


def state_box(state: dict) -> None:
    with st.expander("PROTOTYPE state"):
        st.json(state)


def money(tip: str | None = None, label: str | None = None):
    return st.column_config.NumberColumn(label, help=tip, format="%.1f", width="small")


def pick_table(
    table: pd.DataFrame, ids: list, *, key: str, formats: dict, config: dict | None = None,
    order: list[str] | None = None, height: int | None = None,
):
    """A table where clicking any value selects its row (tinted here, as on the Leaderboard). The top row is
    selected on open. Returns the position of the selected row, or None for an empty table."""
    cells = (st.session_state.get(key) or {}).get("selection", {}).get("cells", [])
    if cells and cells[0][0] < len(ids):
        st.session_state[f"{key}_id"] = ids[cells[0][0]]
    chosen = st.session_state.get(f"{key}_id")
    at = ids.index(chosen) if chosen in ids else (0 if ids else None)
    styled = table.style.apply(lambda r: [TINT if r.name == at else ""] * len(r), axis=1).format(formats, na_rep="n/a")
    size = {} if height is None else {"height": height}
    st.dataframe(
        styled, hide_index=True, column_order=order, column_config=config,
        on_select="rerun", selection_mode="single-cell", key=key, **size,
    )
    return at


def leaderboard_button(row: dict, key: str) -> None:
    """Prototype shortcut for a link to the Leaderboard filtered to one Product and one Market."""
    if st.button("Explain this pair on the Leaderboard", icon=":material/arrow_forward:", key=key):
        pl.reset_filters()
        st.session_state.update(q=row["hs6"], product=row["hs6"], market=row["iso3"])
        st.switch_page(PAGES["Leaderboard"])


def leaderboard() -> None:
    """The settled Leaderboard (#95 split view, #97 logo, #98 banner D, three steps, tooltips)."""
    st.session_state.update(
        banner="D", app_name=prototype_banner.NAMES[0], guide="Three steps", terms="Tooltips only"
    )
    prototype_banner.bar = lambda: None
    prototype_banner.page()


# ---------------------------------------------------------------- Blocked Opportunities

BLOCKED = {
    "A": "Split view, Barrier first",
    "B": "Wide table, Leaderboard explains",
    "C": "Grouped by Market",
    "today": "Today's page",
}
BLOCKED_EXTRAS = {
    "Panel leads with": ["The Barrier", "The three figures (as Leaderboard)"],
    "Illustrative note": ["Expander under the table", "Open above the table (today)"],
}
B_DEFAULTS = {
    "b_q": "", "b_sector": None, "b_region": None, "b_market": None, "b_min": None,
    "b_flagged": False, "b_hubs": False, "b_all": False,
}
B_WIDE = [
    "Product", "Market", "Blocked Opportunity value", "Applied Tariff", "Export Potential", "Actual Exports", "Export Gap", "Caveats",
]
B_MONEY = ["Blocked Opportunity value", "Export Potential", "Actual Exports", "Export Gap"]


def _reset_blocked() -> None:
    st.session_state.update(B_DEFAULTS)


def blocked_filters(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Sidebar filters, applied in the page (prototype shortcut). Returns the kept rows and the active phrases."""
    side, active = st.sidebar, []
    side.subheader("Filters")
    query = side.text_input("Product search", key="b_q", placeholder="HS6 code or words, e.g. 220830 or whisky")
    words = query.strip().lower().split()
    if words:
        rows = [
            r for r in rows
            if r["hs6"].startswith(words[0]) or all(word in r["description"].lower() for word in words)
        ]
        active.append(f"Product '{query.strip()}'")
    sector = side.selectbox(
        "Sector", [None, *sorted({r["sector"] for r in rows})], key="b_sector",
        format_func=lambda s: "All Sectors" if s is None else pl.sector_text(s),
    )
    region = side.selectbox(
        "Market region", [None, *sorted({r["region"] for r in rows})], key="b_region",
        format_func=lambda r: "All regions" if r is None else r,
    )
    names = {r["iso3"]: r["market"] for r in rows if region is None or r["region"] == region}
    market = side.selectbox(
        "Market", [None, *sorted(names, key=names.get)], key="b_market",
        format_func=lambda iso3: "All Markets" if iso3 is None else names[iso3],
    )
    least = side.number_input("Minimum Blocked Opportunity value ($ millions)", value=None, step=1.0, key="b_min")
    caveat = caveats()
    flagged = side.checkbox("Hide Flagged Products", key="b_flagged", help=caveat["flagged_product"][1])
    hubs = side.checkbox("Hide Re-export Hubs", key="b_hubs", help=caveat["reexport_hub"][1])
    if sector:
        active.append(f"Sector {pl.sector_text(sector)}")
    if region:
        active.append(region)
    if market:
        active.append(names[market])
    if least is not None:
        active.append(f"Blocked Opportunity value ≥ ${least:,.1f}m")
    if flagged:
        active.append("Flagged Products hidden")
    if hubs:
        active.append("Re-export Hubs hidden")
    kept = [
        r for r in rows
        if (not sector or r["sector"] == sector)
        and (not region or r["region"] == region)
        and (not market or r["iso3"] == market)
        and (least is None or r["blocked_opportunity_value_usd"] >= least * 1e6)
        and not (flagged and r["flagged_product"])
        and not (hubs and r["reexport_hub"])
    ]
    side.button("Reset filters", on_click=_reset_blocked, disabled=not active)
    return kept, active


def blocked_frame(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    return pd.DataFrame({
        "Product": raw["description"],
        "Market": raw["market"],
        "Blocked Opportunity value": raw["blocked_opportunity_value_usd"] / 1e6,
        "Applied Tariff": [row.get("applied_tariff_pct") for row in rows],
        "Export Potential": raw["export_potential_usd"] / 1e6,
        "Actual Exports": raw["actual_exports_usd"] / 1e6,
        "Export Gap": raw["export_gap_usd"] / 1e6,
        "Opportunity Type": raw["opportunity_type"],
        "Caveats": [caveat_badges(row) for row in rows],
        "HS6": raw["hs6"],
        "Sector": raw["sector"].map(pl.sector_text),
        "Region": raw["region"],
    })


def blocked_config(gap: str) -> dict:
    word = labels()["illustrative"]["badge"]
    return {
        "Product": st.column_config.TextColumn(width="large", pinned=True),
        "Blocked Opportunity value": money(
            f"{word}: the change in Export Potential the model associates with adding a UK Trade Agreement "
            "and removing the tariff."
        ),
        "Export Potential": money(f"What the model expects the UK to export, given the fundamentals ({gap})."),
        "Actual Exports": money(f"What the UK exported ({gap})."),
        "Export Gap": money(
            "Export Potential minus Actual Exports. A Blocked Opportunity need not show one: the missing "
            "Trade Agreement already holds its Export Potential down."
        ),
        "Opportunity Type": st.column_config.TextColumn("Type"),
        "Applied Tariff": st.column_config.NumberColumn(
            "Applied Tariff %", width="small",
            help="The tariff this Market charges on this Product in the Baseline. A Blocked Opportunity with a "
            "tariff of 0 rests on the Trade Agreement alone.",
        ),
    }


def blocked_table(rows: list[dict], gap: str, *, order: list[str], height: int, key: str) -> dict:
    at = pick_table(
        blocked_frame(rows), [(r["hs6"], r["iso3"]) for r in rows], key=key,
        formats=dict.fromkeys(B_MONEY, "{:,.1f}") | {"Applied Tariff": "{:.1f}"}, config=blocked_config(gap), order=order, height=height,
    )
    return rows[at]


def blocked_page() -> None:
    variant, extra = bar("Blocked Opportunities", BLOCKED, BLOCKED_EXTRAS)
    if variant == "today":
        runpy.run_path(str(PAGES_DIR / "3_Blocked_Opportunities.py"))
        return
    health = top(
        "Blocked Opportunities",
        "UK Product–Market pairs where the model associates adding a UK Trade Agreement, and removing the "
        "tariff, with a material rise in Export Potential. They point to trade-policy work rather than export "
        "promotion.",
    )
    if health is None:
        return
    body = fetch("/opportunities/blocked", client.api_url(), limit=6000)
    thresholds = fetch("/scenarios/thresholds", client.api_url())
    every = [r for r in body["opportunities"] if r["blocked_opportunity"]]
    rows, active = blocked_filters(every)
    word, year, gap = illustrative(), body["baseline_year"], body["gap_period_label"]
    barrier_first = extra["Panel leads with"] == "The Barrier"

    if extra["Illustrative note"].startswith("Open"):
        scenario_note(body["note"])
    show_all = len(rows) > ROWS and variant != "C" and st.session_state.get("b_all", False)
    shown = rows if show_all or variant == "C" else rows[:ROWS]
    count = (
        f"Showing the top {len(shown):,} of {len(rows):,} Blocked Opportunities" if len(rows) > len(shown)
        else f"Showing all {len(rows):,} Blocked Opportunities"
    )
    total = sum(r["blocked_opportunity_value_usd"] for r in rows)
    st.markdown(
        (f"**Filters:** {' · '.join(active)}. " if active else "**No filters.** ")
        + f"{count}, with a total Blocked Opportunity value of **{format_usd(total)}** ({word})."
    )
    if not rows:
        st.info("No Blocked Opportunities match these filters.")
        footer(health)
        return
    unit_line = (
        f"Amounts are **$ millions** ({labels()['usd_unit']}). The Blocked Opportunity value is {word}, on the "
        f"{year} Baseline; the other amounts are {gap}s. A pair is a Blocked Opportunity when that value is at "
        f"least {thresholds['min_effect_share']:.0%} of its {year} Export Potential and at least "
        f"{format_usd(thresholds['min_effect_usd'])}. Try other agreements and thresholds on the Scenarios page."
    )

    if variant == "A":
        view_column, all_column = st.columns([4, 2], vertical_alignment="center")
        view = view_column.segmented_control(
            "View", ["Table and detail", "Wide table"], key="b_view", required=True, default="Table and detail",
            label_visibility="collapsed",
        )
        if len(rows) > ROWS:
            all_column.toggle(f"Show all {len(rows):,}", key="b_all")
        st.caption(unit_line + " Click any value in a row to see its Barrier.")
        if view == "Wide table":
            row = blocked_table(shown, gap, order=B_WIDE, height=390, key="b_table")
            with st.container(border=True):
                pl.detail(row, barrier_first=barrier_first)
        else:
            table_side, detail_side = st.columns([5, 4], gap="large")
            with table_side:
                row = blocked_table(
                    shown, gap, order=["Product", "Market", "Blocked Opportunity value"], height=700, key="b_table"
                )
            with detail_side, st.container(border=True, height=780):
                pl.detail(row, barrier_first=barrier_first)
    elif variant == "B":
        if len(rows) > ROWS:
            st.toggle(f"Show all {len(rows):,}", key="b_all")
        st.caption(unit_line + " Click any value in a row to see its Barrier.")
        row = blocked_table(shown, gap, order=B_WIDE, height=460, key="b_table")
        with st.container(border=True):
            text, button = st.columns([7, 3], vertical_alignment="center")
            with text:
                st.markdown(f"**{row['description']} → {row['market']}**")
                pl._barrier(row, fetch(f"/explain/{row['hs6']}/{row['iso3']}", client.api_url()))
            with button:
                leaderboard_button(row, "b_jump")
                st.caption("Opens the Leaderboard filtered to this Product and Market, with its detail panel.")
    else:  # C: grouped by Market
        st.caption(unit_line + " A Trade Agreement is made with a Market, so the pairs are grouped by Market.")
        frame = pd.DataFrame(rows)
        groups = (
            frame.groupby(["iso3", "market"], as_index=False)
            .agg(count=("hs6", "size"), value=("blocked_opportunity_value_usd", "sum"))
            .sort_values("value", ascending=False, ignore_index=True)
        )
        markets_side, products_side = st.columns([4, 6], gap="large")
        with markets_side:
            st.markdown(f"**Markets** ({len(groups):,})")
            at = pick_table(
                pd.DataFrame({
                    "Market": groups["market"], "Blocked Opportunities": groups["count"],
                    "Total value": groups["value"] / 1e6,
                }),
                list(groups["iso3"]), key="b_markets", formats={"Total value": "{:,.1f}"}, height=700,
                config={"Total value": money(f"Total Blocked Opportunity value, {word}.")},
            )
        group = groups.iloc[at]
        mine = [r for r in rows if r["iso3"] == group["iso3"]]
        with products_side:
            st.markdown(
                f"**{group['market']}**: {group['count']:,} Blocked Opportunities, with a total Blocked "
                f"Opportunity value of **{format_usd(group['value'])}** ({word})"
            )
            row = blocked_table(
                mine, gap, order=["Product", "Blocked Opportunity value", "Applied Tariff", "Export Potential", "Actual Exports"],
                height=250, key=f"b_products_{group['iso3']}",
            )
            with st.container(border=True, height=470, key=f"b_panel_{row['hs6']}_{row['iso3']}"):
                pl.detail(row, barrier_first=barrier_first)

    full = pd.DataFrame(rows).drop(columns=["flagged_product", "reexport_hub", "uncertain_opportunity"])
    st.download_button(
        "Download CSV (full dollars, every column)", full.to_csv(index=False), "blocked_opportunities.csv",
        "text/csv", key="b_csv",
    )
    if extra["Illustrative note"].startswith("Expander"):
        with st.expander(f"About Blocked Opportunity values: {body['note']['label'].lower()}, beside their benchmarks"):
            scenario_note(body["note"])
    footer(health)
    state_box({"variant": variant, **extra, "filters": active, "blocked_opportunities": len(rows), "shown": len(shown)})


# ---------------------------------------------------------------- Scenarios

SCENARIOS = {
    "A": "Builder in the sidebar, results in tabs",
    "B": "Builder on top, sections below",
    "C": "Split view by Market",
    "today": "Today's page",
}
SCENARIO_EXTRAS = {
    "Rows per list": ["100, 50 in the drill-down (today)", "500", "1,000 (the API's maximum)"],
    "On open": ["Empty until Run Scenario", "Runs the first preset"],
    "Scenario caution": ["Page line only", "Page line and Scenario line"],
}
LIMITS = {"100": (100, 50), "500": (500, 500), "1,0": (1000, 1000)}
PRESET, CUSTOM, TARIFF = "Trading Bloc preset", "Custom Trade Agreements", "Tariff change"
S_MONEY = ["Scenario Effect", "Baseline", "Scenario"]


def builder(*, compact: bool) -> tuple[dict, str]:
    """The Scenario controls, drawn where the caller stands. Returns the request body and the Scenario in words."""
    presets = {preset["name"]: preset for preset in fetch("/scenarios/presets", client.api_url())}
    names = {market["iso3"]: market["name"] for market in fetch("/markets", client.api_url())}
    defaults = fetch("/scenarios/thresholds", client.api_url())
    kind = st.radio("Scenario", [PRESET, CUSTOM, TARIFF], key="kind", horizontal=not compact)
    if kind == PRESET:
        preset = st.selectbox("Trading Bloc preset", list(presets), format_func=lambda n: presets[n]["title"], key="preset")
        st.caption(presets[preset]["description"])
        body, words = {"preset": preset}, presets[preset]["title"]
    elif kind == TARIFF:
        st.caption(
            "Set the tariff chosen Markets charge on UK goods. Adding a Trade Agreement removes the tariff, and "
            f"removing one restores the Market's {labels()['last_tariff_year']} MFN rate."
        )
        chosen = st.multiselect("Markets that charge the tariff", list(names), format_func=names.get, key="tariff_markets")
        rate = st.number_input("Applied Tariff, %", min_value=0.0, value=10.0, step=1.0, key="tariff_rate")
        sectors = st.multiselect(
            "Sectors; leave empty for every Sector", SECTORS, format_func=pl.sector_text, key="tariff_sectors"
        )
        body = {"tariff": {"markets": chosen, "rate_pct": rate, "sectors": sectors or None}}
        words = f"an Applied Tariff of {rate:g}% in {len(chosen)} Market{'s' if len(chosen) != 1 else ''}"
    else:
        add = st.multiselect("Add a Trade Agreement with", list(names), format_func=names.get, key="add")
        remove = st.multiselect("Remove the Trade Agreement with", list(names), format_func=names.get, key="remove")
        body = {"add_agreements": add, "remove_agreements": remove}
        words = f"{len(add)} Trade Agreement{'s' if len(add) != 1 else ''} added, {len(remove)} removed"
    st.markdown(
        "**Blocked Opportunity thresholds**", help="A pair counts as a Blocked Opportunity when its Scenario Effect clears both."
    )
    left, right = (st.container(), st.container()) if compact else st.columns(2)
    share = left.number_input(
        "At least this % of Baseline Export Potential", min_value=0.0, value=defaults["min_effect_share"] * 100,
        step=5.0, key="min_share",
    )
    usd = right.number_input(
        "At least this many $ millions", min_value=0.0, value=defaults["min_effect_usd"] / 1e6, step=0.5, key="min_usd"
    )
    return body | {"min_effect_share": share / 100, "min_effect_usd": usd * 1e6}, words


def run_button(body: dict, words: str, *, auto: bool) -> None:
    asked = json.dumps(body, sort_keys=True)
    if st.button("Run Scenario", key="run", type="primary") or (auto and "scenario_body" not in st.session_state):
        st.session_state["scenario_body"], st.session_state["scenario_words"] = asked, words
    elif st.session_state.get("scenario_body") not in (None, asked):
        st.caption("These controls differ from the result shown. Run Scenario to update it.")


def scenario(limit: int, market: str | None = None) -> dict | None:
    """The run Scenario's result (or one Market's drill-down), or None after showing why there is none."""
    if "scenario_body" not in st.session_state:
        st.info("Choose a Scenario and press **Run Scenario** to see what the model associates with it.")
        return None
    body = json.loads(st.session_state["scenario_body"]) | {"limit": limit}
    if market:
        body["market"] = market
    try:
        result = run_scenario(json.dumps(body, sort_keys=True), client.api_url())
    except client.InvalidRequest as error:
        st.warning(f"This Scenario is not valid: {error}")
        return None
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
        return None
    if result["changes_nothing"]:
        st.warning(result["summary"])
        return None
    return result


def result_head(result: dict, *, scenario_line: bool, note_beside: bool) -> None:
    st.markdown(f"**{st.session_state.get('scenario_words', 'Scenario')}.** {result['summary']}")
    if result["clipped_pairs"]:
        st.warning(
            f"{result['clipped_pairs']:,} pairs were given a tariff beyond the training range, the tariffs the "
            "model has seen. It is held at the edge of that range, so the Scenario Effect flattens beyond it: the "
            "model associates no further change with a higher tariff."
        )
    if scenario_line:
        st.caption(
            "Scenario Effects are associations the model has learned from past trade, not causal forecasts: the "
            "model associates these policy conditions with this change in Export Potential, holding everything "
            f"else at the {result['baseline_year']} Baseline."
        )
    total, word = result["total"], illustrative()
    metrics, note = st.columns([1, 1]) if note_beside else (st.container(), None)
    a, b, c = metrics.columns(3)
    a.metric(f"Scenario Effect, {word}", format_change(total["effect_usd"]))
    b.metric("Share of Baseline", f"{total['effect_share']:+.1%}")
    c.metric("Blocked Opportunities", f"{result['blocked_count']:,}")
    if note is not None:
        with note:
            scenario_note(result["note"])
    else:
        with st.expander(f"About Scenario Effects: {result['note']['label'].lower()}, beside their benchmarks"):
            scenario_note(result["note"])
    st.caption(
        f"Amounts in the tables are **$ millions** ({labels()['usd_unit']}) of {result['baseline_year']} Export "
        f"Potential. Scenario Effects are {word}."
    )


def effect_frame(rows: list[dict], lead: dict[str, str]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    table = raw[list(lead)].rename(columns=lead)
    table["Scenario Effect"] = raw["effect_usd"] / 1e6
    table["Effect, % of Baseline"] = raw["effect_share"]
    table["Baseline"] = raw["baseline_potential_usd"] / 1e6
    table["Scenario"] = raw["scenario_potential_usd"] / 1e6
    if "pairs" in raw:
        table["Pairs changed"] = raw["pairs"]
    if "baseline_tariff_pct" in raw:
        table["Tariff, Baseline %"] = raw["baseline_tariff_pct"]
        table["Tariff, Scenario %"] = raw["scenario_tariff_pct"]
    return table


S_FORMATS = {
    **dict.fromkeys(S_MONEY, "{:,.1f}"), "Effect, % of Baseline": "{:+.0%}",
    "Tariff, Baseline %": "{:.1f}", "Tariff, Scenario %": "{:.1f}",
}


def effect_config() -> dict:
    return {
        "Product": st.column_config.TextColumn(width="large", pinned=True),
        "Scenario Effect": money(
            f"{labels()['illustrative']['badge']}: Scenario Export Potential minus Baseline Export Potential."
        ),
        "Effect, % of Baseline": st.column_config.NumberColumn(
            help="The Scenario Effect as a share of Baseline Export Potential.", width="small"
        ),
        "Baseline": money("Baseline Export Potential."),
        "Scenario": money("Scenario Export Potential."),
        "Tariff, Baseline %": st.column_config.NumberColumn(help="Applied Tariff in the Baseline.", width="small"),
        "Tariff, Scenario %": st.column_config.NumberColumn(help="Applied Tariff under the Scenario.", width="small"),
    }


def effect_table(
    rows: list[dict], lead: dict[str, str], *, key: str, ids: list | None = None, order: list[str] | None = None,
    height: int | None = None,
):
    """A Scenario table. With `ids`, clicking any value selects its row; returns the selected position."""
    if not rows:
        st.caption("None.")
        return None
    table = effect_frame(rows, lead)
    formats = {heading: spec for heading, spec in S_FORMATS.items() if heading in table}
    if ids is not None:
        return pick_table(table, ids, key=key, formats=formats, config=effect_config(), order=order, height=height)
    size = {} if height is None else {"height": height}
    st.dataframe(
        table.style.format(formats, na_rep="n/a"), hide_index=True, column_config=effect_config(),
        column_order=order, key=key, **size,
    )
    return None


PAIR_LEAD = {"description": "Product", "market": "Market"}
PRODUCT_LEAD = {"description": "Product"}
MARKET_LEAD = {"market": "Market"}


def pair_card(row: dict, key: str) -> None:
    """One pair under the Scenario, with the way to its explanation."""
    blocked = " It counts as a **Blocked Opportunity** under this Scenario." if row["blocked_opportunity"] else ""
    st.markdown(
        f"**{row['description']} → {row['market']}.** The model associates this Scenario with "
        f"**{format_change(row['effect_usd'])}** ({row['effect_share']:+.0%}) of Export Potential: "
        f"{format_usd(row['baseline_potential_usd'])} in the Baseline, {format_usd(row['scenario_potential_usd'])} "
        f"under the Scenario. Applied Tariff {row['baseline_tariff_pct']:.1f}% in the Baseline, "
        f"{row['scenario_tariff_pct']:.1f}% under the Scenario.{blocked}"
    )
    leaderboard_button(row, key)


def pairs_with_card(rows: list[dict], *, key: str, order: list[str] | None = None, height: int = 320) -> None:
    at = effect_table(rows, PAIR_LEAD, key=key, ids=[(r["hs6"], r["iso3"]) for r in rows], order=order, height=height)
    if at is not None:
        with st.container(border=True):
            pair_card(rows[at], f"{key}_jump")


def scenarios_page() -> None:
    variant, extra = bar("Scenarios", SCENARIOS, SCENARIO_EXTRAS)
    if variant == "today":
        runpy.run_path(str(PAGES_DIR / "4_Scenarios.py"))
        return
    health = top(
        "Scenarios",
        "Re-predict UK Export Potential under a Trading Bloc preset, your own Trade Agreements or a tariff change, "
        "and see which Markets and Products the model associates with the largest change.",
    )
    if health is None:
        return
    limit, drill_limit = LIMITS[extra["Rows per list"][:3]]
    auto = extra["On open"].startswith("Runs")
    scenario_line = extra["Scenario caution"].endswith("Scenario line")

    if variant == "A":
        with st.sidebar:
            st.subheader("Scenario")
            run_button(*builder(compact=True), auto=auto)
        result = scenario(limit)
        if result:
            result_head(result, scenario_line=scenario_line, note_beside=False)
            names = {m["iso3"]: m["market"] for m in result["markets"]}
            market_tab, product_tab, blocked_tab, drill_tab = st.tabs([
                f"Markets ({len(names):,})", "Products gaining", f"Blocked Opportunities ({result['blocked_count']:,})",
                "One Market's pairs",
            ])
            with market_tab:
                effect_table(result["markets"], MARKET_LEAD, key="s_markets")
            with product_tab:
                effect_table(result["products"], PRODUCT_LEAD, key="s_products")
            with blocked_tab:
                st.caption("Click any value in a row to see the pair under this Scenario.")
                pairs_with_card(result["blocked_opportunities"], key="s_blocked")
            with drill_tab:
                market = st.selectbox("Market", list(names), format_func=names.get, key="drill_market")
                drilled = scenario(drill_limit, market)
                if drilled:
                    pairs_with_card(drilled["top_pairs"], key=f"s_drill_{market}")
    elif variant == "B":
        with st.container(border=True):
            run_button(*builder(compact=False), auto=auto)
        result = scenario(limit)
        if result:
            result_head(result, scenario_line=scenario_line, note_beside=True)
            st.subheader("Change per Market")
            st.caption("Click any value in a row to see that Market's most affected pairs.")
            markets = result["markets"]
            at = effect_table(markets, MARKET_LEAD, key="s_markets", ids=[m["iso3"] for m in markets], height=320)
            chosen = markets[at]
            st.subheader(f"The pairs most affected in {chosen['market']}")
            drilled = scenario(drill_limit, chosen["iso3"])
            if drilled:
                pairs_with_card(drilled["top_pairs"], key=f"s_drill_{chosen['iso3']}")
            st.subheader("Top Products gaining")
            effect_table(result["products"], PRODUCT_LEAD, key="s_products", height=320)
            st.subheader(f"Blocked Opportunities ({result['blocked_count']:,})")
            pairs_with_card(result["blocked_opportunities"], key="s_blocked")
    else:  # C: split view by Market
        ran = st.session_state.get("scenario_words")
        with st.expander(f"Scenario: {ran}. Change it" if ran else "Build a Scenario", expanded=not ran):
            run_button(*builder(compact=False), auto=auto)
        result = scenario(limit)
        if result:
            result_head(result, scenario_line=scenario_line, note_beside=False)
            markets = result["markets"]
            markets_side, pairs_side = st.columns([4, 5], gap="large")
            with markets_side:
                st.markdown(f"**Markets** ({len(markets):,}): click one to see its pairs")
                at = effect_table(
                    markets, MARKET_LEAD, key="s_markets", ids=[m["iso3"] for m in markets],
                    order=["Market", "Scenario Effect", "Effect, % of Baseline"], height=560,
                )
            chosen = markets[at]
            with pairs_side, st.container(border=True, height=640):
                st.subheader(chosen["market"])
                a, b, c = st.columns(3)
                a.metric(f"Scenario Effect, {illustrative()}", format_change(chosen["effect_usd"]))
                b.metric("Share of Baseline", f"{chosen['effect_share']:+.1%}")
                c.metric("Pairs changed", f"{chosen['pairs']:,}")
                st.markdown("**The pairs most affected**")
                drilled = scenario(drill_limit, chosen["iso3"])
                if drilled:
                    pairs_with_card(
                        drilled["top_pairs"], key=f"s_drill_{chosen['iso3']}", height=250,
                        order=["Product", "Scenario Effect", "Effect, % of Baseline", "Tariff, Baseline %", "Tariff, Scenario %"],
                    )
            product_tab, blocked_tab = st.tabs(
                ["Products gaining, all Markets", f"Blocked Opportunities ({result['blocked_count']:,})"]
            )
            with product_tab:
                effect_table(result["products"], PRODUCT_LEAD, key="s_products")
            with blocked_tab:
                pairs_with_card(result["blocked_opportunities"], key="s_blocked")
    footer(health)
    state_box({"variant": variant, **extra, "scenario": json.loads(st.session_state.get("scenario_body", "null"))})


# ---------------------------------------------------------------- Model Evaluation

EVALUATION = {
    "A": "One long page, contents list",
    "B": "Tabs",
    "C": "Scorecard and expanders",
    "today": "Today's page",
}
EVALUATION_EXTRAS = {"Excluded Products as": ["Reasons, list in an expander", "The full list"]}
GROUPS = ["Accuracy", "Rankings and ranges", "Scenario checks", "Scope and limits"]


class _Quiet:
    """Streamlit for today's section helpers, minus their own subheaders: the prototype draws the headings."""

    def __getattr__(self, name):
        return getattr(st, name)

    def subheader(self, *args, **kwargs) -> None:
        pass


def _today_helpers() -> dict:
    source = (PAGES_DIR / "6_Model_Evaluation.py").read_text(encoding="utf8").split('st.title("Model Evaluation")')[0]
    helpers: dict = {}
    exec(compile(source, "6_Model_Evaluation.py", "exec"), helpers)  # noqa: S102  throwaway
    helpers["st"] = _Quiet()
    return helpers


def excluded_products(style: str) -> None:
    excluded = pd.DataFrame(fetch("/products/excluded", client.api_url()))
    st.markdown(
        f"{len(excluded)} Products are left out of training, scoring and every ranking, because their recorded "
        "trade follows bullion-market, pipeline or grid geography the model cannot see, or is unreliable. Their "
        "trade still counts in every country and world total."
    )
    full = pd.DataFrame({
        "HS6": excluded["hs6"], "Product": excluded["description"],
        "Sector": excluded["sector"].map(pl.sector_text), "Reason": excluded["reason"],
    })
    if style == "The full list":
        st.dataframe(full, hide_index=True, height=420)
        return
    reasons = full.groupby("Reason", as_index=False).agg(Products=("HS6", "size")).sort_values("Products", ascending=False)
    st.dataframe(reasons[["Products", "Reason"]], hide_index=True)
    with st.expander(f"All {len(full)} Excluded Products"):
        st.dataframe(full, hide_index=True, height=420)


def evaluation_sections(report: dict, ev: dict, excluded_style: str) -> list[tuple[str, str, object]]:
    """(group, heading, draw) for each section, in today's order, with Excluded Products above Known limitations."""
    test_year = report["test_year"]

    def headline() -> None:
        head = report["headline"]
        (st.success if head["beats_naive"] else st.warning)(head["text"])
        st.caption(
            f"Measured on {test_year}, a year the model never saw: the models were trained on "
            f"{years_text(report['test_training_years'])} with settings tuned on {report['validation_year']}. "
            f"The naive baseline is {report['naive_baseline'][0].lower()}{report['naive_baseline'][1:]}"
        )

    def accuracy() -> None:
        for scope, label in ev["SCOPES"].items():
            st.markdown(f"**{label}**")
            st.dataframe(ev["_metric_table"](report["accuracy"][scope]["overall"], report["metrics"]), hide_index=True)
        st.caption(
            "Rank correlation is averaged over Products: 1 means Markets are ranked exactly as the UK's actual "
            "exports rank them, 0 means no better than chance."
        )

    def weakest() -> None:
        scope = st.radio("Rows evaluated", list(ev["SCOPES"]), format_func=ev["SCOPES"].get, key="breakdown_scope", horizontal=True)
        scored = report["accuracy"][scope]
        sector_tab, region_tab = st.tabs(["By Sector", "By region"])
        with sector_tab:
            st.markdown(f"**Weakest Sectors** (rank correlation): {ev['_weakest'](scored['by_sector'])}")
            st.dataframe(ev["_breakdown_table"](scored["by_sector"], "Sector"), hide_index=True)
        with region_tab:
            st.markdown(f"**Weakest regions** (rank correlation): {ev['_weakest'](scored['by_region'])}")
            st.dataframe(ev["_breakdown_table"](scored["by_region"], "Region"), hide_index=True)

    def limitations() -> None:
        for limitation in report["limitations"]:
            st.markdown(f"**{limitation['title']}.** {limitation['text']}")

    def not_run() -> None:
        for part in report["not_run"]:
            st.markdown(part["text"])

    def backtest() -> None:
        names = {market["iso3"]: market["name"] for market in fetch("/markets", client.api_url())}
        ev["_show_event_backtest"](report["event_backtest"], names, report.get("event_backtest_note"))

    sections = [
        ("Accuracy", "How accurate is the model?", headline),
        ("Accuracy", "How the model was chosen", lambda: ev["_show_model_choice"](report["model_choice"])),
        ("Accuracy", f"Accuracy in {test_year}", accuracy),
        ("Accuracy", "Where the model is weakest", weakest),
    ]
    if report.get("tariff_split_text"):
        sections.append(("Accuracy", "Observed against carried-forward tariffs", lambda: ev["_show_tariff_split"](report)))
    sections.append(("Rankings and ranges", "Stability of the rankings", lambda: st.markdown(report["stability"]["text"])))
    if report.get("potential_range"):
        sections.append((
            "Rankings and ranges", "How often Actual Exports fall inside the Potential Range",
            lambda: ev["_show_potential_range"](report["potential_range"]),
        ))
    if report.get("lift"):
        sections.append((
            "Rankings and ranges", "Do large Export Gaps point at growth? (Lift)", lambda: ev["_show_lift"](report["lift"])
        ))
    if report.get("event_backtest"):
        sections.append(
            ("Scenario checks", "Scenario Effects against a real Trade Agreement (Event Backtest)", backtest)
        )
    if report.get("not_run"):
        sections.append(("Scope and limits", "Parts not run", not_run))
    sections += [
        ("Scope and limits", "Excluded Products", lambda: excluded_products(excluded_style)),
        ("Scope and limits", "Known limitations", limitations),
    ]
    return sections


def _slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def evaluation_page() -> None:
    variant, extra = bar("Model Evaluation", EVALUATION, EVALUATION_EXTRAS)
    if variant == "today":
        runpy.run_path(str(PAGES_DIR / "6_Model_Evaluation.py"))
        return
    health = top(
        "Model Evaluation", "How accurate the model is, where it is weak, and what it cannot tell you."
    )
    if health is None:
        return
    report = fetch("/evaluation", client.api_url())
    sections = evaluation_sections(report, _today_helpers(), extra["Excluded Products as"])

    if variant == "A":
        st.sidebar.subheader("On this page")
        st.sidebar.markdown("\n".join(f"- [{heading}](#{_slug(heading)})" for _, heading, _ in sections))
        for _, heading, draw in sections:
            st.subheader(heading, anchor=_slug(heading))
            draw()
    elif variant == "B":
        sections[0][2]()  # the headline stays above the tabs
        for tab, group in zip(st.tabs(GROUPS), GROUPS, strict=True):
            with tab:
                for of, heading, draw in sections[1:]:
                    if of == group:
                        st.subheader(heading)
                        draw()
    else:  # C: scorecard and expanders
        head, tiles = report["headline"], st.columns(4)
        tiles[0].metric(
            "Rank correlation, UK pairs", f"{head['model']:.2f}", f"{head['model'] - head['naive']:+.2f} against naive",
            help=head["text"],
        )
        if report.get("potential_range"):
            covered = report["potential_range"]
            tiles[1].metric(
                "Inside the Potential Range", f"{covered['coverage']:.0%}", f"target {covered['target']:.0%}",
                delta_color="off", help="How often Actual Exports fall inside the Potential Range on held-out rows.",
            )
        if report.get("lift"):
            tiles[2].metric(
                "Size-matched Lift", f"{report['lift']['size_matched']['lift']:.2f}", "1 = no better than size",
                delta_color="off", help="A tendency across many pairs, not a forecast for any one Opportunity.",
            )
        tiles[3].metric(
            "Ranking stability", f"{report['stability']['spearman_potential']:.2f}",
            f"{years_text(report['stability']['years'])}", delta_color="off",
            help="Rank correlation of Export Potential between two years scored by the same models.",
        )
        open_at_start = {"How accurate is the model?", "Known limitations"}
        for _, heading, draw in sections:
            with st.expander(heading, expanded=heading in open_at_start):
                draw()
    footer(health)
    state_box({"variant": variant, **extra, "sections": [heading for _, heading, _ in sections]})


# ---------------------------------------------------------------- the app


SHIP = (
    f"<svg xmlns='{prototype_logo.SVG_NS}' viewBox='0 0 64 64'>"
    "<defs><clipPath id='c'><rect width='64' height='64' rx='14'/></clipPath></defs>"
    "<g clip-path='url(#c)'><rect width='64' height='64' fill='#2a78d6'/><g transform='translate(0 -3)'>"
    # The logo's ship (#97), unchanged, riding up and down and rocking a little.
    "<g><animateTransform attributeName='transform' type='translate' values='0 1.5;0 -3;0 1.5' dur='1.8s' "
    "repeatCount='indefinite'/><g><animateTransform attributeName='transform' type='rotate' "
    "values='-3 32 46;3 32 46;-3 32 46' dur='1.8s' begin='-0.45s' repeatCount='indefinite'/>"
    "<path d='M7 42h50l-7 10H13z' fill='#fff' stroke='#fff' stroke-width='3' stroke-linejoin='round'/>"
    "<path d='M9 38V26.5a1.5 1.5 0 0 1 1.5-1.5h5a1.5 1.5 0 0 1 1.5 1.5V38z' fill='#fff'/>"
    "<g fill='#fff' opacity='.55'><rect x='21' y='30' width='9.5' height='7.5' rx='1.5'/>"
    "<rect x='32.5' y='30' width='9.5' height='7.5' rx='1.5'/><rect x='44' y='30' width='9.5' height='7.5' rx='1.5'/>"
    "<rect x='21' y='20.5' width='9.5' height='7.5' rx='1.5'/></g>"
    "<rect x='32.5' y='20.5' width='9.5' height='7.5' rx='1.5' fill='#fff'/></g></g>"
    # The logo's wave line, lengthened and sliding one wavelength per loop.
    "<path d='M-26.2 60q4-3 8.3 0t8.3 0 8.3 0 8.3 0 8.3 0 8.3 0 8.3 0 8.3 0 8.3 0 8.3 0 8.3 0 8.3 0' fill='none' "
    "stroke='#fff' stroke-linecap='round' stroke-linejoin='round' stroke-width='2.5' opacity='.5'>"
    "<animateTransform attributeName='transform' type='translate' values='0 0;16.6 0' dur='1.2s' "
    "repeatCount='indefinite'/></path></g></g></svg>"
)


def loader() -> None:
    """A container ship riding waves in the middle of the page while the app is working. Plain CSS on
    Streamlit's own running state, so no custom component; it waits 0.4s so quick reruns do not flash it."""
    st.html(
        f"""<style>
        .stApp[data-test-script-state="running"]::after {{
            content: ""; position: fixed; left: 50%; top: 45%; width: 120px; height: 120px; margin: -60px 0 0 -60px;
            background: url("{prototype_logo.data_uri(SHIP)}") center / contain no-repeat; border-radius: 26px;
            box-shadow: 0 6px 28px rgba(0, 0, 0, 0.35); z-index: 999999; pointer-events: none;
            opacity: 0; animation: ship-in 0.2s ease 0.4s forwards;
        }}
        @keyframes ship-in {{ to {{ opacity: 1; }} }}
        </style>"""
    )


def run() -> None:
    tile = prototype_logo.mark_svg("Tile", dark=False)
    st.set_page_config(page_title=NAME, page_icon=prototype_logo.data_uri(tile), layout="wide")
    st.logo(tile, size="large")
    loader()
    PAGES.update({
        "Leaderboard": st.Page(leaderboard, title="Leaderboard", url_path="leaderboard", default=True),
        "Blocked Opportunities": st.Page(blocked_page, title="Blocked Opportunities", url_path="blocked-opportunities"),
        "Scenarios": st.Page(scenarios_page, title="Scenarios", url_path="scenarios"),
        "Model Evaluation": st.Page(evaluation_page, title="Model Evaluation", url_path="model-evaluation"),
    })
    st.navigation(list(PAGES.values())).run()
