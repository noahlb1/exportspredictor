"""The detail panel (spec #103, "Detail panel"): one Product–Market pair explained in place, built from its ranking row
and its explanation read (`/explain/{hs6}/{iso3}`).

The Leaderboard shows the whole panel with `show(row)`, in order:

1. `opening(row, body)`: the pair, its caveats and the tariff-carried-forward warning.
2. `figures(body)`: Export Potential, Actual Exports and Export Gap, each with its term tooltip.
3. `from_facts(row, body)`: the rest, from the facts table on: the facts, the Market's Market Reach, the Barrier
   (only for a pair with no UK Trade Agreement), the factors and the year-by-year history.

A page with another opening (Blocked Opportunities leads with the Barrier, then its own three figures) reads the
explanation with `explanation(row)`, draws `barrier(row, body, note=..., titled=True)` (the box headed by the pair),
`opening(row, body, titled=False)` and its figures itself, then
`from_facts(row, body, with_barrier=False, with_gap=False)`: without the Export Gap figures, which that page does not
show.

A row is a ranking row (`/opportunities/top`, `/opportunities/blocked`): it carries the pair's names, caveats, Trade
Agreement, Applied Tariff and Blocked Opportunity value. The explanation carries the Gap Period figures, the factors
and the history. Every figure is an association the model has learned, worded so.
"""

import pandas as pd
import streamlit as st

from app import client
from app.format import format_usd, tariff_text
from app.tables import MILLION, millions, millions_unit, money_column
from app.ui import fetch, labels, show_caveats, term

TOP_FACTORS = 8
RAISED, LOWERED = "#2a78d6", "#e34948"  # the factor bars (spec #103, "Theme and logo")
TEXT = {"light": "#1b1915", "dark": "#f5f2ec"}  # chart value labels take the mode's text colour
# The Factor column has a fixed width, so the value and association beside it stay in view in a narrow panel.
FACTOR_WIDTHS = {"Factor": st.column_config.TextColumn("Factor", width="medium")}
MARKET_REACH_HELP = (
    "How widely the UK sends its Products to this Market, against how widely it does across Markets. The model "
    "associates broad UK reach into a Market with higher Export Potential there, for every Product alike, so where "
    "the UK sends fewer Products to a whole Market than expected, it shows here, not in any one Product's figures."
)
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


def opening(row: dict, body: dict, *, titled: bool = True) -> None:
    """The pair (unless `titled` is False: a box above already names it), what it is, its caveats and, when the
    Baseline year's Applied Tariff is carried forward, that warning."""
    if titled:
        st.subheader(_title(row))
    st.caption(f"HS6 {row['hs6']} · {row['sector_name']} · {row['region']}")
    show_caveats(row)
    if body["tariff_caveat"]:
        st.warning(body["tariff_caveat"])


def _title(row: dict) -> str:
    return f"{row['description']} → {row['market']}"


def uplift_text(row: dict) -> str | None:
    """A Blocked Opportunities row's value as an uplift on Export Potential (e.g. "+25%"), only where Actual Exports
    exceed Export Potential (`blocked_opportunity_uplift`, spec #113); None elsewhere and on other pages' rows."""
    uplift = row.get("blocked_opportunity_uplift")
    return None if uplift is None else f"{uplift:+.0%}"


def _value_text(row: dict, year: int) -> str:
    """The Blocked Opportunity value as the panel words it: the uplift on Export Potential where there is one (the
    dollars after it), the dollars elsewhere."""
    dollars = format_usd(row["blocked_opportunity_value_usd"])
    uplift = uplift_text(row)
    return dollars if uplift is None else f"{uplift} on its {year} Export Potential ({dollars})"


def figures(body: dict) -> None:
    """Export Potential, Actual Exports and Export Gap in the adaptive money form, each with the API's term wording."""
    potential, actual, gap = st.columns(3)
    potential.metric("Export Potential", format_usd(body["export_potential_usd"]), help=term("export_potential"))
    actual.metric("Actual Exports", format_usd(body["actual_exports_usd"]), help=term("actual_exports"))
    gap.metric("Export Gap", format_usd(body["export_gap_usd"]), help=term("export_gap"))
    st.caption(f"Each is a {body['gap_period_label']}.")


def from_facts(row: dict, body: dict, *, with_barrier: bool = True, with_gap: bool = True) -> None:
    """The panel from its facts table on: the facts, the Barrier (unless the page led with it), the factors and the
    year-by-year history. `with_gap=False` leaves out the Export Gap figures (the Relative Export Gap and each year's
    Export Gap), for a page that shows Baseline-year figures only."""
    facts(row, body, with_gap=with_gap)
    market_reach(body)
    if with_barrier:
        barrier(row, body)
    factors(body)
    history(body, with_gap=with_gap)


def facts(row: dict, body: dict, *, with_gap: bool = True) -> None:
    """The facts table, the Barrier's figures first: one line per fact, with a tooltip where a fact needs one.
    `with_gap=False` leaves out the Relative Export Gap."""
    year = body["baseline_year"]
    kind = body["opportunity_type"]
    kinds = term("opportunity_type")
    meaning = next((sentence for sentence in kinds.split(". ") if sentence.startswith(f"{kind}:")), kinds)
    meaning = meaning.removeprefix(f"{kind}:").strip().rstrip(".")
    relative = body["relative_export_gap"]
    value = _value_text(row, year)
    blocked = ", a Blocked Opportunity" if row["blocked_opportunity"] else ""
    lines = [
        (
            f"Applied Tariff, {year}",
            tariff_text(row["applied_tariff_pct"]),
            f"The tariff the Market charges on this Product from the UK in the Baseline year {year}.",
        ),
        ("UK Trade Agreement in force", "Yes" if row["trade_agreement"] else "No", None),
        (
            "Blocked Opportunity value",
            f"{value}{blocked} ({labels()['illustrative']['badge'].lower()})",
            (
                f"The rise in {year} Export Potential the model associates with adding a UK Trade Agreement and "
                "removing the tariff where none is in force; 0 where one is."
            ),
        ),
        ("Opportunity Type", f"{kind}: {meaning[:1].lower()}{meaning[1:]}", kinds),
        (
            "Relative Export Gap",
            "n/a" if relative is None else f"{relative:.0%}",
            "Export Gap divided by Export Potential: the share of its Export Potential the UK does not yet realise.",
        ),
        (
            "Potential Range",
            f"{format_usd(body['potential_range_low_usd'])} to {format_usd(body['potential_range_high_usd'])}",
            body["potential_range_text"],
        ),
    ]
    if not with_gap:
        lines = [line for line in lines if line[0] != "Relative Export Gap"]
    with st.container(border=True):
        for name, text, tip in lines:
            st.markdown(f"**{name}:** {text}", help=tip)


def market_reach(body: dict) -> None:
    """The Market's Market Reach in one sentence (ADR 0010), where the API serves one."""
    if body.get("market_reach"):
        st.markdown(f"**Market Reach.** {body['market_reach']['text']}", help=MARKET_REACH_HELP)


def barrier(row: dict, body: dict, note: str | None = None, *, titled: bool = False) -> None:
    """The Barrier, only for a pair with no UK Trade Agreement: its Applied Tariff and the Illustrative amount the
    model associates with adding an agreement. A warning when the pair is a Blocked Opportunity. `note` adds a
    sentence of the calling page's own; `titled` heads the box with the pair, for a panel that opens with it."""
    if row["trade_agreement"]:
        return
    badge = labels()["illustrative"]["badge"]
    year, uplift = body["baseline_year"], uplift_text(row)
    amount = (
        f"**{format_usd(row['blocked_opportunity_value_usd'])}** more {year} Export Potential" if uplift is None
        else f"**{uplift}** on its {year} Export Potential ({format_usd(row['blocked_opportunity_value_usd'])})"
    )
    text = (
        f"**The Barrier.** No UK Trade Agreement is in force with {row['market']}, and the Applied Tariff is "
        f"{tariff_text(row['applied_tariff_pct'])}. The model associates adding a Trade Agreement, and removing the "
        f"tariff, with {amount}. {badge}: a tendency the model has learned, not a forecast."
    )
    if note:
        text += f" {note}"
    if titled:
        text = f"**{_title(row)}**\n\n{text}"
    if row["blocked_opportunity"]:
        st.warning(text)
    else:
        st.info(text + " That is too small to count as a Blocked Opportunity.")


def _factor_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.rename(columns=FACTOR_COLUMNS)[list(FACTOR_COLUMNS.values())]


def factors(body: dict) -> None:
    """The top factors as a bar chart and a table, every factor in an expander, then how to read them."""
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
    top = every.head(TOP_FACTORS)
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
    with st.expander("All factors"):
        st.dataframe(_factor_table(every), hide_index=True, column_config=FACTOR_WIDTHS)
    st.caption(
        "×1.50 means the model associates this factor with 1.5 times the Export Potential it expects on average "
        "(SHAP values); ×0.70 means 30% lower. The factors multiply together into the Export Potential. "
        f"Computed with SHAP from fold model {body['fold']}, the same model that scored this pair's Baseline, "
        f"{format_usd(body['baseline_export_potential_usd'])} in {body['baseline_year']}."
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
