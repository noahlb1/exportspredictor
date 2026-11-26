"""The detail panel: one Product–Market pair explained in place, built from its ranking row
and its explanation read (`/explain/{hs6}/{iso3}`).

The Leaderboard shows the whole panel with `show(row)`, in order:

1. `opening(row, body)`: the pair, its caveats and the tariff-carried-forward warning.
2. `figures(body)`: Export Potential, Actual Exports and Export Gap, each with its term tooltip.
3. `from_facts(row, body)`: the rest, from the facts table on: the facts, the Market's Market Reach, the Barrier
   (only for a pair with no UK Trade Agreement), the factors and the year-by-year history.

A Blocked Opportunity has its own panel, `blocked_opportunity(row)`, shared by the Blocked Opportunities page, which
draws it under its tables through `app.views`, and the Scenario Builder, which reaches it through
`blocked_opportunities` (the Products table, then the chosen pair's panel), so a pair reads the same on both. It leads with a box naming the pair and its Barrier, then
`before_after`, the three-bar chart, and a sentence on the split, then `opening(row, body, titled=False,
with_tariff_caveat=False)` (the page states the carried-forward tariff once, above its tables), then
`from_facts(row, body, with_barrier=False, with_gap=False)`: without the facts its box and chart already give, and
without the Export Gap figures, which a Blocked Opportunity is not read by.

A row is a ranking row (`/opportunities/top`, `/opportunities/blocked`): it carries the pair's names, caveats, Trade
Agreement, Applied Tariff and Blocked Opportunity value. The explanation carries the Gap Period figures, the factors
and the history. Every figure is an association the model has learned, worded so.
"""

import pandas as pd
import streamlit as st

from app import client
from app.format import format_usd, tariff_text
from app.tables import (
    BLOCKED_FORMATS,
    MILLION,
    blocked_products_columns,
    blocked_products_frame,
    millions,
    millions_unit,
    money_column,
    pick_row_if_chosen,
    scroll_hint,
)
from app.ui import fetch, illustrative, labels, show_caveats, term, terms_help

TOP_FACTORS = 8
RAISED, LOWERED = "#2a78d6", "#e34948"  # the factor bars that raise and lower the estimate
TEXT = {"light": "#1b1915", "dark": "#f5f2ec"}  # chart value labels take the mode's text colour
# The Factor column has a fixed width, so the value and association beside it stay in view in a narrow panel.
FACTOR_WIDTHS = {"Factor": st.column_config.TextColumn("Factor", width="medium")}
FACTOR_COLUMNS = {
    "label": "Factor",
    "value_text": "Value for this pair",
    "effect": "Association with Export Potential",
}


def explanation(row: dict) -> dict:
    """The pair's explanation read: its Gap Period figures, factors and history."""
    return fetch(f"/explain/{row['hs6']}/{row['iso3']}", client.api_url())


def show(row: dict) -> None:
    """The whole panel for one ranking row, as the Leaderboard shows it."""
    body = explanation(row)
    opening(row, body)
    figures(body)
    from_facts(row, body)


def blocked_opportunity(row: dict) -> None:
    """One Blocked Opportunity's panel, for a row of `/opportunities/blocked`: a neutral box naming the pair and its
    Barrier, the before and after, the split, the caveats, then the rest of the panel from its facts on, without the
    Export Gap or the facts given above."""
    body = explanation(row)
    year, word = body["baseline_year"], illustrative()
    st.info(f"**{row['description']} → {row['market']}**\n\n**Barrier:** {row['barrier']}.")
    before_after(row, year)
    st.caption(f"Each is for {year}; Export Potential with a Trade Agreement is {word}.")
    st.markdown(
        f"{row['value_split']} {word.capitalize()}: a tendency the model has learned, not a forecast.",
        help=terms_help(("Tariff part", "tariff_part"), ("Agreement part", "agreement_part")),
    )
    if row["above_export_potential"]:
        st.caption(
            f"UK Actual Exports in {year} are already above Export Potential, though below the level the model "
            "associates with an agreement: the model already rates current trade here as strong."
        )
    opening(row, body, titled=False, with_tariff_caveat=False)
    from_facts(row, body, with_barrier=False, with_gap=False)


def blocked_opportunities(
    rows: list[dict], body: dict, *, key: str, remembered: str, noun: str, with_market: bool = False
) -> None:
    """Blocked Opportunities as the Scenario Builder lists them: the Products table (`app.tables.blocked_products_frame`, with a Market column `with_market`), where clicking any value
    chooses a pair, remembered as (HS6, ISO3) under `remembered`, and under it the chosen pair's panel in a box, or,
    until one is chosen, a line saying to click a `noun` ("Product" or "pair"). `body` is the response holding the
    rows (it gives the Baseline year); `key` names the table."""
    chosen = pick_row_if_chosen(
        blocked_products_frame(rows, with_market=with_market),
        [(row["hs6"], row["iso3"]) for row in rows],
        key=key,
        remembered=remembered,
        formats=BLOCKED_FORMATS,
        column_config=blocked_products_columns(body),
    )
    if chosen is None:
        st.caption(
            f"Click a {noun} to see its before and after, and how much of the rise the model associates with the tariff."
        )
        return
    row = rows[chosen]
    # A new key for each pair makes the panel a new element, so it starts from its top on a new selection.
    with st.container(border=True, key=f"{key}_panel_{row['hs6']}_{row['iso3']}"):
        blocked_opportunity(row)


def opening(row: dict, body: dict, *, titled: bool = True, with_tariff_caveat: bool = True) -> None:
    """The pair (unless `titled` is False: a box above already names it), what it is, its caveats and, when the
    Baseline year's Applied Tariff is carried forward, that warning (unless `with_tariff_caveat` is False: the page
    states it once itself)."""
    if titled:
        st.subheader(_title(row))
    st.caption(f"HS6 {row['hs6']} · {row['sector_name']} · {row['region']}")
    show_caveats(row)
    if with_tariff_caveat and body["tariff_caveat"]:
        st.warning(body["tariff_caveat"])


def _title(row: dict) -> str:
    return f"{row['description']} → {row['market']}"


def figures(body: dict) -> None:
    """Export Potential, Actual Exports and Export Gap in the adaptive money form, each with the API's term wording."""
    potential, actual, gap = st.columns(3)
    potential.metric("Export Potential", format_usd(body["export_potential_usd"]), help=term("export_potential"))
    actual.metric("Actual Exports", format_usd(body["actual_exports_usd"]), help=term("actual_exports"))
    gap.metric("Export Gap", format_usd(body["export_gap_usd"]), help=term("export_gap"))
    st.caption(f"Each is a {body['gap_period_label']}.")


def from_facts(row: dict, body: dict, *, with_barrier: bool = True, with_gap: bool = True) -> None:
    """The panel from its facts table on: the facts, the Barrier (unless the page led with it), the factors and the
    year-by-year history. `with_barrier=False` also leaves out the facts the page's own Barrier box and figures give
    (Applied Tariff, UK Trade Agreement, Blocked Opportunity value). `with_gap=False` leaves out the Export Gap figures
    (the Relative Export Gap and each year's Export Gap), for a page that shows Baseline-year figures only."""
    facts(row, body, with_gap=with_gap, with_policy=with_barrier)
    market_reach(body)
    if with_barrier:
        barrier(row, body)
    factors(body)
    history(body, with_gap=with_gap)


POLICY_FACTS = ("Applied Tariff", "UK Trade Agreement in force", "Blocked Opportunity value")


def facts(row: dict, body: dict, *, with_gap: bool = True, with_policy: bool = True) -> None:
    """The facts table, the Barrier's figures first: one line per fact, with a tooltip where a fact needs one.
    `with_gap=False` leaves out the Relative Export Gap; `with_policy=False` the Barrier's figures (`POLICY_FACTS`)."""
    year = body["baseline_year"]
    kind = body["opportunity_type"]
    kinds = term("opportunity_type")
    meaning = next((sentence for sentence in kinds.split(". ") if sentence.startswith(f"{kind}:")), kinds)
    meaning = meaning.removeprefix(f"{kind}:").strip().rstrip(".")
    relative = body["relative_export_gap"]
    value = format_usd(row["blocked_opportunity_value_usd"])
    blocked = ", a Blocked Opportunity" if row["blocked_opportunity"] else ""
    lines = [
        (
            f"Applied Tariff, {year}",
            tariff_text(row["applied_tariff_pct"]),
            term("applied_tariff"),
        ),
        ("UK Trade Agreement in force", "Yes" if row["trade_agreement"] else "No", term("trade_agreement")),
        (
            "Blocked Opportunity value",
            f"{value}{blocked} ({labels()['illustrative']['badge'].lower()})",
            term("blocked_opportunity_value"),
        ),
        ("Opportunity Type", f"{kind}: {meaning[:1].lower()}{meaning[1:]}", kinds),
        (
            "Relative Export Gap",
            "n/a" if relative is None else f"{relative:.0%}",
            term("relative_export_gap"),
        ),
        (
            "Potential Range",
            f"{format_usd(body['potential_range_low_usd'])} to {format_usd(body['potential_range_high_usd'])}",
            body["potential_range_text"],
        ),
    ]
    if not with_gap:
        lines = [line for line in lines if line[0] != "Relative Export Gap"]
    if not with_policy:
        lines = [line for line in lines if not line[0].startswith(POLICY_FACTS)]
    with st.container(border=True):
        for name, text, tip in lines:
            st.markdown(f"**{name}:** {text}", help=tip)


def market_reach(body: dict) -> None:
    """The Market's Market Reach in one sentence (ADR 0010), where the API serves one."""
    if body.get("market_reach"):
        st.markdown(f"**Market Reach.** {body['market_reach']['text']}", help=term("market_reach"))


def barrier(row: dict, body: dict) -> None:
    """The Barrier, only for a pair with no UK Trade Agreement: its Applied Tariff and the Illustrative amount the
    model associates with adding an agreement. Only a known tariff above 0 is said to go to 0 with it (ADR 0011). A
    warning when the pair is a Blocked Opportunity."""
    if row["trade_agreement"]:
        return
    badge = labels()["illustrative"]["badge"]
    year = body["baseline_year"]
    amount = f"**{format_usd(row['blocked_opportunity_value_usd'])}** more {year} Export Potential"
    tariff = row["applied_tariff_pct"]
    if tariff:
        association = f"The model associates adding a Trade Agreement, which sets that tariff to 0, with {amount}."
    else:
        association = (
            f"The model associates adding a Trade Agreement with {amount}, from the agreement as a whole, as no tariff "
            "is priced here."
        )
    text = (
        f"**The Barrier.** No UK Trade Agreement is in force with {row['market']}, and the Applied Tariff is "
        f"{tariff_text(tariff)}. {association} {badge}: a tendency the model has learned, not a forecast."
    )
    if row["blocked_opportunity"]:
        st.warning(text)
    else:
        st.info(
            text + " It is not a Blocked Opportunity: the rise is below the thresholds, or Actual Exports are already "
            "above the level with an agreement (no room to grow)."
        )


def _factor_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.rename(columns=FACTOR_COLUMNS)[list(FACTOR_COLUMNS.values())]


def factors(body: dict) -> None:
    """The top factors (the largest effects) as a bar chart and a table, every factor in an expander, each ordered
    from the highest multiplier to the lowest, then how to read them."""
    st.markdown("**What raised or lowered its Export Potential**")
    st.markdown(
        "Each factor is something the model associates with higher or lower UK exports of this Product to this "
        "Market. These are associations the model has learned, not causes."
    )
    every = pd.DataFrame(body["factors"])
    every["effect"] = [
        f"{row['direction']} ×{row['multiplier']:,.2f}" if row["direction"] != "no effect" else "no effect"
        for row in every.to_dict("records")
    ]
    # The API orders factors by size of effect: keep the largest, then show each list highest multiplier first.
    top = every.head(TOP_FACTORS).sort_values("multiplier", ascending=False, kind="stable")
    every = every.sort_values("multiplier", ascending=False, kind="stable")
    shown = top[top["direction"] != "no effect"].assign(times=lambda t: [f"×{m:,.2f}" for m in t["multiplier"]])
    st.vega_lite_chart(
        {
            "data": {"values": shown[["label", "value_text", "multiplier", "direction", "times"]].to_dict("records")},
            "height": {"step": 26},
            "encoding": {
                "y": {"field": "label", "type": "nominal", "sort": None, "title": None, "axis": {"labelLimit": 300}},
                "x": {
                    "field": "multiplier",
                    "type": "quantitative",
                    "scale": {"type": "log"},
                    "title": "Association with Export Potential (×1 = no effect)",
                    "axis": {"titleAnchor": "end"},  # a centred title runs off the right of a narrow panel
                },
                "tooltip": [
                    {"field": "label", "title": "Factor"},
                    {"field": "value_text", "title": "Value for this pair"},
                    {"field": "times", "title": "Association"},
                ],
            },
            "layer": [
                {
                    "mark": "bar",
                    "encoding": {
                        "x2": {"datum": 1},
                        "color": {
                            "field": "direction",
                            "type": "nominal",
                            "title": None,
                            "scale": {"domain": ["raised", "lowered"], "range": [RAISED, LOWERED]},
                            "legend": {"orient": "top"},
                        },
                    },
                },
                {
                    "mark": {"type": "text", "align": "left", "dx": 4, "color": _text_colour()},
                    "encoding": {"text": {"field": "times"}},
                },
            ],
        },
        width="stretch",
    )
    st.dataframe(_factor_table(top), hide_index=True, column_config=FACTOR_WIDTHS)
    with st.expander(f"All {len(every)} factors"):
        st.dataframe(_factor_table(every), hide_index=True, column_config=FACTOR_WIDTHS)
        scroll_hint(len(every))
    st.caption(
        "×1.50 means the model associates this factor with 1.5 times the Export Potential it expects on average "
        "; ×0.70 means 30% lower. The factors multiply together into the Export Potential. They come from the model "
        f"that scored this pair's Baseline, {format_usd(body['baseline_export_potential_usd'])} in "
        f"{body['baseline_year']}."
    )


BEFORE_AFTER = ("Actual Exports", "Export Potential", "Export Potential with a Trade Agreement")
BEFORE_AFTER_COLOURS = ("#8a8378", RAISED, "#9cc3ee")  # grey; the model's estimate; the hypothetical, lighter


def before_after(row: dict, year: int) -> None:
    """One Blocked Opportunity's before and after as three bars in $ millions: its Baseline-year Actual
    Exports, Export Potential and Export Potential with a Trade Agreement, each the API's figure."""
    fields = ("baseline_actual_exports_usd", "baseline_export_potential_usd", "potential_with_agreement_usd")
    values = [
        {"figure": figure, "value": row[field] / MILLION, "text": format_usd(row[field])}
        for figure, field in zip(BEFORE_AFTER, fields, strict=True)
    ]
    st.vega_lite_chart(
        {
            "data": {"values": values},
            "height": {"step": 30},
            "encoding": {
                "y": {"field": "figure", "type": "nominal", "sort": None, "title": None, "axis": {"labelLimit": 300}},
                "x": {"field": "value", "type": "quantitative", "title": f"{millions_unit()}, {year}"},
                "tooltip": [{"field": "figure", "title": "Figure"}, {"field": "text", "title": str(year)}],
            },
            "layer": [
                {
                    "mark": "bar",
                    "encoding": {
                        "color": {
                            "field": "figure",
                            "type": "nominal",
                            "legend": None,
                            "scale": {"domain": list(BEFORE_AFTER), "range": list(BEFORE_AFTER_COLOURS)},
                        }
                    },
                },
                {
                    "mark": {"type": "text", "align": "left", "dx": 4, "color": _text_colour()},
                    "encoding": {"text": {"field": "text"}},
                },
            ],
        },
        width="stretch",
    )


def _text_colour() -> str:
    """The text colour of the mode the visitor's browser shows (Vega draws text black by default)."""
    return TEXT["dark"] if st.context.theme.type == "dark" else TEXT["light"]


def history(body: dict, *, with_gap: bool = True) -> None:
    """Actual Exports against Export Potential year by year, then the yearly figures with each year's Applied Tariff
    mark in an expander (with each year's Export Gap unless `with_gap` is False)."""
    st.markdown("**Year by year**")
    year = body["baseline_year"]
    points = {point["year"]: point for point in body["history"]}
    years = list(range(year - 5, year + 1))
    empty = {"actual_exports_usd": 0.0, "export_potential_usd": None, "export_gap_usd": None, "tariff_imputed": None}
    yearly = [points.get(each, empty) for each in years]
    lines = [
        {"year": each, "series": series, "value": point[field] / MILLION}
        for each, point in zip(years, yearly, strict=True)
        for series, field in (("Actual Exports", "actual_exports_usd"), ("Export Potential", "export_potential_usd"))
        if point[field] is not None
    ]
    st.vega_lite_chart(
        {
            "data": {"values": lines},
            "height": 220,
            "mark": {"type": "line", "point": True, "tooltip": True},
            "encoding": {
                "x": {"field": "year", "type": "ordinal", "title": None, "axis": {"labelAngle": 0}},
                "y": {"field": "value", "type": "quantitative", "title": millions_unit()},
                "color": {"field": "series", "type": "nominal", "title": None, "legend": {"orient": "top"}},
            },
        },
        width="stretch",
    )
    averaged = body["gap_period_label"]
    with st.expander("Figures by year"):
        st.caption(f"Amounts in {millions_unit()}.")
        # Each year's Applied Tariff mark (ADR 0006); blank for a year with no row.
        marks = {True: f"Carried forward from {labels()['last_tariff_year']}", False: "From tariff data"}
        table = pd.DataFrame(
            {
                "Year": years,
                "Export Potential": millions(p["export_potential_usd"] for p in yearly),
                "Actual Exports": millions(p["actual_exports_usd"] for p in yearly),
                "Export Gap": millions(p["export_gap_usd"] for p in yearly),
                "Applied Tariff": [marks.get(p["tariff_imputed"], "") for p in yearly],
            }
        )
        st.dataframe(
            table if with_gap else table.drop(columns=["Export Gap"]),
            hide_index=True,
            column_config={
                "Export Potential": money_column("Export Potential", term("export_potential")),
                "Actual Exports": money_column("Actual Exports", "What the UK actually exported that year."),
                **(
                    {"Export Gap": money_column("Export Gap", "Export Potential minus Actual Exports that year.")}
                    if with_gap
                    else {}
                ),
            },
        )
        gap = f"The Export Gap above is the {averaged} of the yearly Export Gaps of the Gap Period. " if with_gap else ""
        st.caption(
            f"{gap}Each year's Export Potential is scored from that year's conditions. "
            f"Earlier years show UK Actual Exports only. {year} figures are provisional. A year with no recorded UK "
            "exports shows 0."
        )
