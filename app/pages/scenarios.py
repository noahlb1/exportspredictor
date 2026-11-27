"""The Scenario Builder: the Advisor builds a Scenario (a Trading Bloc preset, Trade Agreements added or removed, or
a tariff change) and sees the change in UK Export Potential the model associates with it.

The page reads top to bottom: what a Trade Agreement means in the model, the
builder in a box, then, once **Run Scenario** is pressed, the result in the before/after style of the Blocked
Opportunities page: the summary (framing only; each figure is stated once, in the metrics), the clipped-tariff
note when it applies, a tariff Scenario's raised/lowered line, the metrics, the Illustrative note closed in the Blocked
Opportunities page's expander, Change per Market, the pairs most affected in the selected Market, Top Products gaining
and, last, Blocked Opportunities. Only an invalid Scenario is a warning; "changes nothing" and the clipped-tariff note
are neutral boxes.

Blocked Opportunities have one source (ADR 0012). Only a Scenario that adds a Trade Agreement with a Market that has
none in force (`adds_agreement`) shows them: the Blocked Opportunity metric and a section listing those Markets'
Blocked Opportunities as the Blocked Opportunities page does, in its Products table (with a Market column) and its
panel (`app.panel.blocked_opportunity`). Any other Scenario shows one neutral line instead, with a link to that page.

A link with `?add=IND` (ISO3 codes, comma-separated) fills the builder with Custom Trade Agreements adding those
Markets, once, without running it (`prefill`); the Blocked Opportunities page links each Market this way.

Clicking any value in a Market or pair table selects its row and tints it, as on the other pages
(`app.tables.pick_row`); Top Products gaining cannot be clicked. The page remembers the choice in session state, so a
test can set it as a click does: `scenario_market` (ISO3; the top Market when unset or not in the result) drives the
drill-down, `scenario_drill_pair` (`{"hs6", "iso3"}`) and `scenario_blocked_pair` ((HS6, ISO3), as on the Blocked
Opportunities page) open a pair's box under that table; no pair is selected until one is clicked.

The drill-down lists the Market's top pairs only, so a Product search above it (`app.filters.product_search`, as in
the ranking pages' sidebars) asks for one Product instead: its pair's box replaces the table, or a line says the
Scenario does not change that pair.

The page words nothing about a result: the summary, the raised/lowered line (`tariff_directions`) and the Blocked
Opportunity prose (`blocked_note`: the section's caption or the neutral line, each stating the fixed rule) come from
the API, worded in `exportspredictor.scenarios`.
"""

import hashlib
import json

import pandas as pd
import streamlit as st

from app import client, panel
from app.filters import product_search
from app.format import format_change, format_usd, tariff_text
from app.frame import page
from app.tables import (
    ONE_DECIMAL,
    caveats_column,
    market_column,
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
    caveat_tiers,
    fetch,
    illustrative,
    labels,
    run_scenario,
    scenario_note,
    show_caveats,
    term,
    terms_help,
)

PRESET = "Trading Bloc preset"
CUSTOM = "Custom Trade Agreements"
TARIFF = "Tariff change"
ROWS = 100  # rows in each list
DRILL_DOWN_ROWS = 50
MARKET = "scenario_market"
DRILL_PAIR = "scenario_drill_pair"
BLOCKED_PAIR = "scenario_blocked_pair"
PRODUCT_KEY_PREFIX = "scenario_"  # the drill-down's Product search: `scenario_product_query`, `scenario_product`
LEADERBOARD = "pages/leaderboard.py"
BLOCKED_PAGE = "pages/blocked_opportunities.py"
ADD_PARAM = "add"  # URL parameter: ISO3 codes, comma-separated, to add a Trade Agreement with (`prefill`)
APPLIED_LINK = "_scenario_add_link"  # the `add` link already applied to the builder
# How each numeric column reads in a click-to-select table (a Styler's formats replace the column config's).
FORMATS = {
    "Scenario Effect": ONE_DECIMAL,
    "Effect, % of Baseline": "{:.1%}",
    "Baseline": ONE_DECIMAL,
    "Scenario": ONE_DECIMAL,
    "Tariff, Baseline %": ONE_DECIMAL,
    "Tariff, Scenario %": ONE_DECIMAL,
}


def builder(presets: list[dict], market_names: dict[str, str], sectors: dict[str, str]) -> dict:
    """The Scenario controls; returns the request body they describe. There are no threshold controls: the Blocked
    Opportunity thresholds are fixed (ADR 0012)."""
    preset_info = {preset["name"]: preset for preset in presets}
    kind = st.radio("Scenario", [PRESET, CUSTOM, TARIFF], key="kind", persist_state="session", horizontal=True)
    if kind == PRESET:
        preset = st.selectbox(
            "Trading Bloc preset",
            list(preset_info),
            format_func=lambda name: preset_info[name]["title"],
            key="preset", persist_state="session",
        )
        st.caption(preset_info[preset]["description"])
        body: dict = {"preset": preset}
    elif kind == TARIFF:
        st.caption(
            "Set the tariff chosen Markets charge on UK goods, in every Sector or only in chosen ones. "
            "Adding or removing a Trade Agreement changes the tariff too: adding one removes it, and removing one "
            f"restores the Market's {labels()['last_tariff_year']} standard (most-favoured-nation, MFN) rate."
        )
        tariff_markets = st.multiselect(
            "Markets that charge the tariff", list(market_names), format_func=market_names.get, key="tariff_markets", persist_state="session"
        )
        rate = st.number_input("Applied Tariff, %", min_value=0.0, value=10.0, step=1.0, key="tariff_rate", persist_state="session")
        chosen = st.multiselect(
            "Sectors; leave empty for every Sector",
            list(sectors),
            format_func=lambda code: f"{sectors[code]} ({code})",
            key="tariff_sectors", persist_state="session",
        )
        body = {"tariff": {"markets": tariff_markets, "rate_pct": rate, "sectors": chosen or None}}
    else:
        add = st.multiselect(
            "Add a Trade Agreement with", list(market_names), format_func=market_names.get, key="add", persist_state="session"
        )
        remove = st.multiselect(
            "Remove the Trade Agreement with", list(market_names), format_func=market_names.get, key="remove", persist_state="session"
        )
        body = {"add_agreements": add, "remove_agreements": remove}
    return body | {"limit": ROWS}


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
        "Market": market_column(width=None),
        "Scenario Effect": money_column("Scenario Effect", term("scenario_effect")),
        "Effect, % of Baseline": st.column_config.NumberColumn(
            "Effect, % of Baseline",
            help=f"The Scenario Effect as a share of Baseline Export Potential. {word}.",
            format="percent",
        ),
        "Baseline": money_column("Baseline", f"Baseline Export Potential, {year}."),
        "Scenario": money_column("Scenario", f"Export Potential under the Scenario, {year}."),
        "Pairs changed": st.column_config.NumberColumn(
            "Pairs changed", help="Product–Market pairs whose Export Potential the Scenario changes."
        ),
        "Tariff, Baseline %": tariff_column(label="Tariff, Baseline %"),
        "Tariff, Scenario %": tariff_column(
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


def pairs_frame(rows: list[dict]) -> pd.DataFrame:
    """One Market's most affected pairs, with both Applied Tariffs, then each pair's caveat badges."""
    frame = effects(rows, {"description": "Product"})
    raw = pd.DataFrame(rows)
    frame["Tariff, Baseline %"] = pd.Series(raw["baseline_tariff_pct"], dtype="float64")
    frame["Tariff, Scenario %"] = pd.Series(raw["scenario_tariff_pct"], dtype="float64")
    frame["Caveats"] = [caveat_badges(row) for row in rows]
    return frame


def pair_table(rows: list[dict], key: str, table_key: str, year: int) -> None:
    """A table of pairs where clicking any value chooses a pair (remembered under `key` as {"hs6", "iso3"}), and the
    chosen pair's box under it. No pair is chosen until one is clicked."""
    if not rows:
        st.info("No pairs to list.")
        return
    chosen = pick_row_if_chosen(
        pairs_frame(rows),
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


def show_result(result: dict, shown: str, market_names: dict[str, str]) -> None:
    """A Scenario's result, in order: summary, clipped-tariff note, raised/lowered line, metrics, the
    Illustrative expander, Change per Market, the selected Market's pairs, Top Products gaining, then Blocked
    Opportunities or the neutral line. Table keys carry the run, so a new run starts with fresh selections."""
    run_key = hashlib.sha1(shown.encode()).hexdigest()[:12]
    year, word = result["baseline_year"], illustrative()
    config = columns(year)
    st.markdown(result["summary"])
    if result["clipped_pairs"]:
        st.info(
            f"{result['clipped_pairs']:,} pairs were given a tariff higher than any the model was trained on. It is "
            "read as the highest one, so the model associates no further change with a higher tariff."
        )
    if result["tariff_directions"] is not None:
        st.markdown(result["tariff_directions"])
    total = result["total"]
    figures = st.columns([1, 1])[0]  # the metrics keep to half the width, as before the note moved under them
    metrics = figures.columns(3 if result["adds_agreement"] else 2)
    metrics[0].metric(
        f"Scenario Effect, {word}", format_change(total["effect_usd"]),
        help=terms_help(("Scenario Effect", "scenario_effect"), ("Illustrative", "illustrative")),
    )
    metrics[1].metric("Share of Baseline", f"{total['effect_share']:+.1%}", help=term("baseline"))
    if result["adds_agreement"]:
        metrics[2].metric("Blocked Opportunities", f"{result['blocked_count']:,}")
    scenario_note(result["note"])

    st.caption(
        f"Amounts in {millions_unit()}, for the Baseline year {year}. Scenario Effects are {word}. "
        f"Click a Market or a pair to select its row. {labels()['totals_note']['text']}"
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
    st.caption(f"The table lists at most {DRILL_DOWN_ROWS} pairs. Search for a Product to see its pair, however small.")
    with st.columns([1, 1])[0]:
        hs6, _ = product_search(PRODUCT_KEY_PREFIX)
    drill_body = {**json.loads(shown), "market": iso3, "limit": DRILL_DOWN_ROWS} | ({"product": hs6} if hs6 else {})
    try:
        drilled = run_scenario(json.dumps(drill_body, sort_keys=True), client.api_url())
    except (client.InvalidRequest, client.ApiUnavailable) as error:
        if hs6 and isinstance(error, client.InvalidRequest):
            st.info(str(error))  # an Excluded Product, with its reason
        else:
            st.error(f"This Market's pairs did not load: {error}")
    else:
        if hs6 is None:
            pair_table(drilled["top_pairs"], DRILL_PAIR, f"drill_{run_key}_{iso3}", year)
        elif drilled["top_pairs"]:
            pair_box(drilled["top_pairs"][0])  # a Product and a Market are one pair
        else:
            st.info(f"This Scenario does not change Product {hs6} in {market_names.get(iso3, iso3)}.")

    st.subheader("Top Products gaining")
    product_table = effects(result["products"], {"description": "Product"})
    product_table["Pairs changed"] = [product["pairs"] for product in result["products"]]
    st.dataframe(product_table, hide_index=True, column_config=config)

    if result["adds_agreement"]:
        blocked_section(result, run_key)
    else:
        st.markdown(result["blocked_note"])  # this Scenario adds none
        st.page_link(BLOCKED_PAGE, label="Open the Blocked Opportunities page", icon=":material/arrow_forward:")


def blocked_section(result: dict, run_key: str) -> None:
    """The Blocked Opportunities of the Markets the Scenario adds a Trade Agreement with, as the API reads them from
    the Blocked Opportunities page's own table (ADR 0012): the caption, a link to that page with the default-hidden
    Caveat Tiers shown, its Products table with a Market column and, once a pair is clicked, its shared panel."""
    rows = result["blocked_opportunities"]
    hidden = [tier for tier in caveat_tiers() if tier["hidden_by_default"]]
    st.subheader("Blocked Opportunities", help=term("blocked_opportunity"))
    st.caption(result["blocked_note"])
    if hidden:
        st.page_link(
            BLOCKED_PAGE,
            label="See the hidden pairs on the Blocked Opportunities page",
            icon=":material/arrow_forward:",
            query_params={tier["hide_param"]: "false" for tier in hidden},
        )
    if not rows:
        st.info("No pair in these Markets is a Blocked Opportunity.")
        return
    panel.blocked_opportunities(
        rows, result, key=f"blocked_{run_key}", remembered=BLOCKED_PAIR, noun="pair", with_market=True
    )


with page(
    "Scenario Builder",
    "Build a Scenario (a Trading Bloc preset, Trade Agreements added or removed, or a tariff change) and see the "
    "change in UK Export Potential the model associates with it.",
    intro_help=terms_help(("Scenario", "scenario"), ("Trading Bloc", "trading_bloc"), ("Customs Union", "customs_union")),
):
    try:
        presets = fetch("/scenarios/presets", client.api_url())
        markets = fetch("/markets", client.api_url())
        sectors = {sector["code"]: sector["name"] for sector in fetch("/sectors", client.api_url())}
    except client.ApiUnavailable as error:
        st.error(f"The API is unavailable: {error}")
        st.stop()
    # Only the Markets a Scenario accepts, in every picker and in the `?add=` link.
    market_names = {market["iso3"]: market["name"] for market in markets if market["scenario"]}
    prefill(market_names)
    st.caption(labels()["trade_agreement_meaning"])

    with st.container(border=True):
        body_json = json.dumps(builder(presets, market_names, sectors), sort_keys=True)
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
                st.info(result["summary"])  # nothing is wrong: only an invalid Scenario is a warning
            else:
                show_result(result, shown, market_names)
