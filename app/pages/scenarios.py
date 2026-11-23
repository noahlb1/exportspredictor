"""Scenarios: re-predict UK Export Potential under a Trading Bloc preset, custom Trade Agreements or a tariff change.

The page reads top to bottom (spec #103, "Scenarios"): what a Trade Agreement means in the model (spec #133), the
builder in a box, then, once **Run Scenario** is pressed, the summary, the three figures beside the Illustrative note,
Change per Market, the pairs most affected in the selected Market, Top Products gaining and Blocked Opportunities
(each with room to grow, ADR 0011).

A link with `?add=IND` (ISO3 codes, comma-separated) fills the builder with Custom Trade Agreements adding those
Markets, once, without running it (`prefill`); the Blocked Opportunities page links each Market this way.

Clicking any value selects its row and tints it, as on the other pages (`app.tables.pick_row`). The page remembers
the choice in session state, so a test can set it as a click does: `scenario_market` (ISO3; the top Market when unset
or not in the result) drives the drill-down, and `scenario_drill_pair` / `scenario_blocked_pair` (`{"hs6", "iso3"}`)
open a pair's box under that table; no pair is selected until one is clicked.
"""

import hashlib
import json

import pandas as pd
import streamlit as st

from app import client
from app.format import format_change, format_usd, tariff_text
from app.frame import page
from app.tables import (
    ONE_DECIMAL,
    caveats_column,
    millions,
    millions_unit,
    money_column,
    pick_row,
    pick_row_if_chosen,
    product_column,
    tariff_column,
)
from app.ui import (
    caveat_badges,
    fetch,
    illustrative,
    labels,
    run_scenario,
    scenario_note,
    show_caveats,
)

PRESET = "Trading Bloc preset"
CUSTOM = "Custom Trade Agreements"
TARIFF = "Tariff change"
ROWS = 100  # rows in each list
DRILL_DOWN_ROWS = 50
MARKET = "scenario_market"
DRILL_PAIR = "scenario_drill_pair"
BLOCKED_PAIR = "scenario_blocked_pair"
LEADERBOARD = "pages/leaderboard.py"
ADD_PARAM = "add"  # URL parameter: ISO3 codes, comma-separated, to add a Trade Agreement with (`prefill`)
APPLIED_LINK = "_scenario_add_link"  # the `add` link already applied to the builder
# How each numeric column reads in a click-to-select table (a Styler's formats replace the column config's).
FORMATS = {
    "Scenario Effect": ONE_DECIMAL,
    "Effect, % of Baseline": "{:.1%}",
    "Baseline": ONE_DECIMAL,
    "Scenario": ONE_DECIMAL,
    "Actual Exports": ONE_DECIMAL,
    "Tariff, Baseline %": ONE_DECIMAL,
    "Tariff, Scenario %": ONE_DECIMAL,
}


def builder(presets: list[dict], market_names: dict[str, str], sectors: dict[str, str], defaults: dict) -> dict:
    """The Scenario controls; returns the request body they describe."""
    preset_info = {preset["name"]: preset for preset in presets}
    kind = st.radio("Scenario", [PRESET, CUSTOM, TARIFF], key="kind", horizontal=True)
    if kind == PRESET:
        preset = st.selectbox(
            "Trading Bloc preset",
            list(preset_info),
            format_func=lambda name: preset_info[name]["title"],
            key="preset",
        )
        st.caption(preset_info[preset]["description"])
        body: dict = {"preset": preset}
    elif kind == TARIFF:
        st.caption(
            "Set the tariff chosen Markets charge on UK goods, in every Sector or only in chosen ones. "
            "Adding or removing a Trade Agreement changes the tariff too: adding one removes it, and removing one "
            f"restores the Market's {labels()['last_tariff_year']} MFN rate."
        )
        tariff_markets = st.multiselect(
            "Markets that charge the tariff", list(market_names), format_func=market_names.get, key="tariff_markets"
        )
        rate = st.number_input("Applied Tariff, %", min_value=0.0, value=10.0, step=1.0, key="tariff_rate")
        chosen = st.multiselect(
            "Sectors; leave empty for every Sector",
            list(sectors),
            format_func=lambda code: f"{sectors[code]} ({code})",
            key="tariff_sectors",
        )
        body = {"tariff": {"markets": tariff_markets, "rate_pct": rate, "sectors": chosen or None}}
    else:
        add = st.multiselect(
            "Add a Trade Agreement with", list(market_names), format_func=market_names.get, key="add"
        )
        remove = st.multiselect(
            "Remove the Trade Agreement with", list(market_names), format_func=market_names.get, key="remove"
        )
        body = {"add_agreements": add, "remove_agreements": remove}

    st.markdown("**Blocked Opportunity thresholds**: a pair counts when its Scenario Effect clears both.")
    left, right = st.columns(2)
    min_share = left.number_input(
        "Minimum Scenario Effect, % of Baseline Export Potential",
        min_value=0.0,
        value=defaults["min_effect_share"] * 100,
        step=5.0,
        key="min_share",
    )
    min_usd = right.number_input(
        "Minimum Scenario Effect, USD million",
        min_value=0.0,
        value=defaults["min_effect_usd"] / 1e6,
        step=0.5,
        key="min_usd",
    )
    return body | {"min_effect_share": min_share / 100, "min_effect_usd": min_usd * 1e6, "limit": ROWS}


def prefill(market_names: dict[str, str]) -> None:
    """Fill the builder from the URL once (before its widgets): `add=IND,JPN` chooses Custom Trade Agreements adding
    those Markets (unknown codes are left out). The Blocked Opportunities page links each Market this way. A link
    already applied is not applied again, so the visitor's later changes are kept."""
    link = st.query_params.get(ADD_PARAM)
    if link is None or link == st.session_state.get(APPLIED_LINK):
        return
    st.session_state[APPLIED_LINK] = link
    st.session_state["kind"] = CUSTOM
    st.session_state["add"] = [iso3 for iso3 in link.split(",") if iso3 in market_names]


def run(body_json: str) -> None:
    """Run Scenario: the result shown is now this body's, with the top Market and no pair selected."""
    st.session_state["scenario_body"] = body_json
    for key in (MARKET, DRILL_PAIR, BLOCKED_PAIR):
        st.session_state.pop(key, None)


def columns(year: int) -> dict:
    """Column config of the Scenario tables: short headings, each with a tooltip; money in $ millions."""
    word = illustrative().capitalize()
    return {
        "Product": product_column(),
        "Market": st.column_config.TextColumn("Market", help="The Market the UK exports to."),
        "Scenario Effect": money_column(
            "Scenario Effect",
            f"Scenario Export Potential minus Baseline Export Potential, {year}: the change the model associates "
            f"with the Scenario, holding everything else at the Baseline. {word}.",
        ),
        "Effect, % of Baseline": st.column_config.NumberColumn(
            "Effect, % of Baseline",
            help=f"The Scenario Effect as a share of Baseline Export Potential. {word}.",
            format="percent",
        ),
        "Baseline": money_column("Baseline", f"Baseline Export Potential, {year}."),
        "Scenario": money_column("Scenario", f"Export Potential under the Scenario, {year}."),
        "Actual Exports": money_column(
            "Actual Exports", f"UK Actual Exports in the Baseline year {year}; 0 where there was no Flow that year."
        ),
        "Pairs changed": st.column_config.NumberColumn(
            "Pairs changed", help="Product–Market pairs whose Export Potential the Scenario changes."
        ),
        "Tariff, Baseline %": tariff_column(year, label="Tariff, Baseline %"),
        "Tariff, Scenario %": tariff_column(
            year,
            label="Tariff, Scenario %",
            help="The Applied Tariff under the Scenario, in percent, held within the range the model was trained on. "
            "Blank where it is not known.",
        ),
        "Caveats": caveats_column(),
    }


def effects(rows: list[dict], names: dict[str, str]) -> pd.DataFrame:
    """A Scenario table: the naming columns `names` (row field -> heading), then the effects in $ millions."""
    raw = pd.DataFrame(rows)
    frame = pd.DataFrame({heading: raw[field] for field, heading in names.items()})
    frame["Scenario Effect"] = millions(raw["effect_usd"])
    frame["Effect, % of Baseline"] = raw["effect_share"]
    frame["Baseline"] = millions(raw["baseline_potential_usd"])
    frame["Scenario"] = millions(raw["scenario_potential_usd"])
    return frame


def pairs_frame(rows: list[dict], *, with_market: bool, with_exports: bool) -> pd.DataFrame:
    """A table of Product–Market pairs, with both Applied Tariffs (and Baseline-year Actual Exports if asked), then
    each pair's caveat badges."""
    frame = effects(rows, {"description": "Product", **({"market": "Market"} if with_market else {})})
    raw = pd.DataFrame(rows)
    if with_exports:
        frame["Actual Exports"] = millions(raw["baseline_actual_exports_usd"])
    frame["Tariff, Baseline %"] = pd.Series(raw["baseline_tariff_pct"], dtype="float64")
    frame["Tariff, Scenario %"] = pd.Series(raw["scenario_tariff_pct"], dtype="float64")
    frame["Caveats"] = [caveat_badges(row) for row in rows]
    return frame


def pair_table(rows: list[dict], key: str, table_key: str, year: int, *, with_market: bool, with_exports: bool) -> None:
    """A table of pairs where clicking any value chooses a pair (remembered under `key` as {"hs6", "iso3"}), and the
    chosen pair's box under it. No pair is chosen until one is clicked."""
    if not rows:
        st.info("No pairs to list.")
        return
    chosen = pick_row_if_chosen(
        pairs_frame(rows, with_market=with_market, with_exports=with_exports),
        [{"hs6": row["hs6"], "iso3": row["iso3"]} for row in rows],
        key=table_key,
        remembered=key,
        formats=FORMATS,
        column_config=columns(year),
    )
    if chosen is not None:
        pair_box(rows[chosen])


def pair_box(row: dict) -> None:
    """The chosen pair: its caveats, Baseline and Scenario Export Potential, the Scenario Effect and both Applied
    Tariffs, with a link that opens the Leaderboard on that Product and Market (its URL parameters `product` and
    `market`)."""
    with st.container(border=True):
        st.markdown(f"**{row['description']}** (HS6 {row['hs6']}) to **{row['market']}**")
        show_caveats(row)
        baseline, scenario, effect, tariff_before, tariff_after = st.columns(5)
        baseline.metric("Baseline Export Potential", format_usd(row["baseline_potential_usd"]))
        scenario.metric("Scenario Export Potential", format_usd(row["scenario_potential_usd"]))
        effect.metric(
            f"Scenario Effect, {illustrative()}",
            format_change(row["effect_usd"]),
            f"{row['effect_share']:+.1%} of Baseline",
            delta_color="off",
            delta_arrow="off",
        )
        tariff_before.metric("Tariff, Baseline", tariff_text(row["baseline_tariff_pct"]))
        tariff_after.metric("Tariff, Scenario", tariff_text(row["scenario_tariff_pct"]))
        st.page_link(
            LEADERBOARD,
            label="Open this pair on the Leaderboard",
            icon=":material/arrow_forward:",
            query_params={"product": row["hs6"], "market": row["iso3"]},
        )


def _pairs(count: int) -> str:
    return f"{count:,} pair" if count == 1 else f"{count:,} pairs"


def tariff_directions(result: dict) -> str | None:
    """For a tariff Scenario, one line on how many pairs it sets to the rate and how many of those it raises and
    lowers (a pair with an unknown Baseline tariff counts in neither); None for any other Scenario."""
    tariff = result["scenario"].get("tariff")
    if tariff is None or result.get("pairs_raised") is None:
        return None
    changed = sum(market["pairs"] for market in result["markets"])
    return (
        f"This sets {_pairs(changed)} to {tariff['rate_pct']:g}%: it raises {result['pairs_raised']:,} "
        f"and lowers {result['pairs_lowered']:,}."
    )


def show_result(result: dict, shown: str, market_names: dict[str, str]) -> None:
    """A Scenario's result, in order: summary, figures and note, Change per Market, the selected Market's pairs, Top
    Products gaining, Blocked Opportunities. Table keys carry the run, so a new run starts with fresh selections."""
    run_key = hashlib.sha1(shown.encode()).hexdigest()[:12]
    year, word = result["baseline_year"], illustrative()
    config = columns(year)
    st.markdown(result["summary"])
    if result["clipped_pairs"]:
        st.warning(
            f"{result['clipped_pairs']:,} pairs were given a tariff beyond the training range, the "
            "tariffs the model has seen. It is held at the edge of that range, so the Scenario Effect "
            "flattens beyond it: the model associates no further change with a higher tariff."
        )
    directions = tariff_directions(result)
    if directions is not None:
        st.markdown(directions)
    st.caption(
        "Scenario Effects are associations the model has learned from past trade, not causal "
        "forecasts: the model associates these policy conditions with this change in Export "
        f"Potential, holding everything else at the {year} Baseline."
    )
    total = result["total"]
    figures, note = st.columns([1, 1])
    a, b, c = figures.columns(3)
    a.metric(f"Scenario Effect, {word}", format_change(total["effect_usd"]))
    b.metric("Share of Baseline", f"{total['effect_share']:+.1%}")
    c.metric("Blocked Opportunities", f"{result['blocked_count']:,}")
    with note:
        scenario_note(result["note"])

    st.caption(
        f"Amounts in {millions_unit()}, for the Baseline year {year}. Scenario Effects are {word}. "
        f"Click any value to select its row. {labels()['totals_note']['text']}"
    )

    st.subheader("Change per Market")
    changed = result["markets"]
    isos = [market["iso3"] for market in changed]
    market_table = effects(changed, {"market": "Market"})
    market_table["Pairs changed"] = [market["pairs"] for market in changed]
    at = pick_row(
        market_table, list(isos), key=f"markets_{run_key}", remembered=MARKET, formats=FORMATS, column_config=config
    )
    iso3 = isos[at]

    st.subheader(f"Pairs most affected in {market_names.get(iso3, iso3)}")
    # Ask for this Market's own most affected pairs: those of every Market together can leave a smaller Market out.
    drill_body = {**json.loads(shown), "market": iso3, "limit": DRILL_DOWN_ROWS}
    try:
        drilled = run_scenario(json.dumps(drill_body, sort_keys=True), client.api_url())
    except (client.InvalidRequest, client.ApiUnavailable) as error:
        st.error(f"The drill-down failed: {error}")
    else:
        drill_key = f"drill_{run_key}_{iso3}"
        pair_table(drilled["top_pairs"], DRILL_PAIR, drill_key, year, with_market=False, with_exports=False)

    st.subheader("Top Products gaining")
    product_table = effects(result["products"], {"description": "Product"})
    product_table["Pairs changed"] = [product["pairs"] for product in result["products"]]
    st.dataframe(product_table, hide_index=True, column_config=config)

    st.subheader("Blocked Opportunities")
    st.caption(
        f"{result['blocked_count']:,} pairs have a Scenario Effect of at least "
        f"{result['thresholds']['min_effect_share']:.0%} of Baseline Export Potential and at least "
        f"{format_usd(result['thresholds']['min_effect_usd'])}, and room to grow: {year} Actual Exports below their "
        f"Scenario Export Potential. The largest {ROWS} are listed."
    )
    blocked_key = f"blocked_{run_key}"
    pair_table(result["blocked_opportunities"], BLOCKED_PAIR, blocked_key, year, with_market=True, with_exports=True)


with page(
    "Scenarios",
    "Re-predict UK Export Potential under a Trading Bloc preset, custom Trade Agreements or a tariff change.",
):
    try:
        presets = fetch("/scenarios/presets", client.api_url())
        markets = fetch("/markets", client.api_url())
        sectors = {sector["code"]: sector["name"] for sector in fetch("/sectors", client.api_url())}
        defaults = fetch("/scenarios/thresholds", client.api_url())
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
        st.stop()
    # Only the Markets a Scenario accepts, in every picker and in the `?add=` link.
    market_names = {market["iso3"]: market["name"] for market in markets if market["scenario"]}
    prefill(market_names)
    st.caption(labels()["trade_agreement_meaning"])

    with st.container(border=True):
        body_json = json.dumps(builder(presets, market_names, sectors, defaults), sort_keys=True)
        st.button("Run Scenario", key="run", type="primary", on_click=run, args=(body_json,))

    shown = st.session_state.get("scenario_body")
    if shown is None:
        st.caption("Choose a Scenario above and press **Run Scenario** to see the change the model associates with it.")
    else:
        if shown != body_json:
            st.info("The controls differ from the result shown. Press **Run Scenario** to update it.")
        try:
            result = run_scenario(shown, client.api_url())
        except client.InvalidRequest as error:
            st.warning(f"This Scenario is not valid: {error}")
        except client.ApiUnavailable as error:
            st.error(f"The API is unavailable: {error}")
        else:
            if result["changes_nothing"]:
                st.warning(result["summary"])
            else:
                show_result(result, shown, market_names)
