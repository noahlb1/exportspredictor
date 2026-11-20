"""Table pieces every page reuses (spec #103, "Money"): $ millions columns, short headings with tooltips, the term
definitions the API serves, the Applied Tariff column, and the Leaderboard's table and CSV.

The term definitions for tooltips are `app.ui.term`.

Money in a table is $ millions to one decimal: Streamlit has no number format that divides, so `millions` builds a
column already divided by 1,000,000 and `money_column` formats it `%.1f` (a value under $50,000 shows `0.0`). The
columns stay numeric, so they sort as numbers. The unit is said once above the table (`millions_unit`), never in a
heading. A CSV keeps full dollars and every column: build it from the undivided rows (`leaderboard_csv`).
Summary figures, sentences and panel metrics keep the adaptive form of `app.format.format_usd`.
"""

from collections.abc import Iterable

import pandas as pd
import streamlit as st

from app.ui import caveat_badges, illustrative, term, usd_unit, why_column, why_link

MILLION = 1_000_000


def millions(values: Iterable[float | None]) -> pd.Series:
    """Dollar amounts as $ millions, still numeric (missing stays missing), for a `money_column`."""
    return pd.Series(list(values), dtype="float64") / MILLION


def money_column(label: str, help: str) -> "st.column_config.NumberColumn":
    """The column config of a $ millions column (`millions`): one decimal, with a tooltip."""
    return st.column_config.NumberColumn(label, help=help, format="%.1f")


def millions_unit() -> str:
    """The unit, said once above a table of `money_column`s, e.g. "$ millions (USD, nominal)"."""
    return f"$ millions ({usd_unit()})"


def tariff_column(baseline_year: int, label: str = "Applied Tariff %", help: str | None = None) -> (
    "st.column_config.NumberColumn"
):
    """The column config of a ranking row's `applied_tariff_pct` (the Baseline Applied Tariff, percent; blank where
    unknown)."""
    return st.column_config.NumberColumn(
        label,
        help=help
        or f"The tariff the Market charges on this Product from the UK in the Baseline year {baseline_year}, in percent. "
        "Blank where it is not known.",
        format="%.1f",
    )


# The Leaderboard's wide table: the columns shown by default, in order, then those hidden by default (offered by the
# table's own Show/hide columns menu). No column is pinned.
LEADERBOARD_SHOWN = [
    "Product",
    "Market",
    "Export Gap",
    "Export Potential",
    "Actual Exports",
    "Applied Tariff %",
    "Relative Export Gap",
    "UK Trade Agreement",
    "Blocked Opportunity value",
    "Blocked Opportunity",
    "Caveats",
]
LEADERBOARD_HIDDEN = ["Opportunity Type", "Potential Range", "HS6", "Sector", "Region"]
WHY = "Why?"  # the link column, until the detail panel replaces it


def leaderboard_frame(rows: list[dict]) -> pd.DataFrame:
    """The Leaderboard's table of ranking rows: short headings, money in $ millions, every column (shown and hidden),
    and the Why? link."""
    raw = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "Product": raw["description"],
            "Market": raw["market"],
            "Export Gap": millions(raw["export_gap_usd"]),
            "Export Potential": millions(raw["export_potential_usd"]),
            "Actual Exports": millions(raw["actual_exports_usd"]),
            "Applied Tariff %": pd.Series(raw["applied_tariff_pct"], dtype="float64"),
            "Relative Export Gap": raw["relative_export_gap"],
            "UK Trade Agreement": raw["trade_agreement"].map({True: "Yes", False: "No"}),
            "Blocked Opportunity value": millions(raw["blocked_opportunity_value_usd"]),
            "Blocked Opportunity": raw["blocked_opportunity"],
            "Caveats": [caveat_badges(row) for row in rows],
            "Opportunity Type": raw["opportunity_type"],
            "Potential Range": [
                f"{row['potential_range_low_usd'] / MILLION:.1f}–{row['potential_range_high_usd'] / MILLION:.1f}"
                for row in rows
            ],
            "HS6": raw["hs6"],
            "Sector": raw["sector_name"],
            "Region": raw["region"],
            WHY: [why_link(row["hs6"], row["iso3"]) for row in rows],
        }
    )


def leaderboard_columns(body: dict) -> dict:
    """The column config of `leaderboard_frame`: a tooltip on every heading, the four terms in the API's wording."""
    year, averaged = body["baseline_year"], body["gap_period_label"]
    return {
        "Product": st.column_config.TextColumn("Product", help="The Product (HS6, 2012 edition)."),
        "Market": st.column_config.TextColumn("Market", help="The Market the UK exports to."),
        "Export Gap": money_column("Export Gap", term("export_gap")),
        "Export Potential": money_column("Export Potential", term("export_potential")),
        "Actual Exports": money_column("Actual Exports", term("actual_exports")),
        "Applied Tariff %": tariff_column(year),
        "Relative Export Gap": st.column_config.NumberColumn(
            "Relative Export Gap",
            help=f"Export Gap divided by Export Potential ({averaged}): the share of its Export Potential the UK "
            "does not yet realise.",
            format="percent",
        ),
        "UK Trade Agreement": st.column_config.TextColumn(
            "UK Trade Agreement", help=f"Whether a UK Trade Agreement with the Market is in force in {year}."
        ),
        "Blocked Opportunity value": money_column(
            "Blocked Opportunity value",
            f"The rise in {year} Export Potential the model associates with adding a UK Trade Agreement and removing "
            f"the tariff where none is in force; 0 where one is. {illustrative().capitalize()}.",
        ),
        "Blocked Opportunity": st.column_config.CheckboxColumn(
            "Blocked Opportunity", help="The Blocked Opportunity value clears the default thresholds."
        ),
        "Caveats": st.column_config.TextColumn(
            "Caveats", help="Flagged Product or Re-export Hub: read this pair's figures with care."
        ),
        "Opportunity Type": st.column_config.TextColumn("Opportunity Type", help=term("opportunity_type")),
        "Potential Range": st.column_config.TextColumn(
            "Potential Range",
            help=f"Where Actual Exports usually fall for comparable pairs, low–high in $ millions ({averaged}).",
        ),
        "HS6": st.column_config.TextColumn("HS6", help="The Product's HS6 code (2012 edition)."),
        "Sector": st.column_config.TextColumn("Sector", help="The Product's Sector (HS2 chapter)."),
        "Region": st.column_config.TextColumn("Region", help="The Market's region."),
        WHY: why_column(),
    }


def leaderboard_unit(body: dict) -> str:
    """The line above the Leaderboard's table that states the unit once."""
    return (
        f"Amounts in {millions_unit()}. Export Potential, Actual Exports and Export Gap are "
        f"{body['gap_period_label']}s; the Blocked Opportunity value is for {body['baseline_year']} and "
        f"{illustrative()}."
    )


def leaderboard_csv(body: dict, rows: list[dict]) -> pd.DataFrame:
    """The Leaderboard's CSV: every row given, in full dollars, with every column of the table (money headings name
    their period and unit, as the CSV has no line above it)."""
    table = leaderboard_frame(rows).drop(columns=[WHY, "Potential Range"])
    averaged, year, unit = body["gap_period_label"], body["baseline_year"], usd_unit()
    raw = pd.DataFrame(rows)
    dollars = {
        f"Export Gap, {averaged} ({unit})": raw["export_gap_usd"],
        f"Export Potential, {averaged} ({unit})": raw["export_potential_usd"],
        f"Actual Exports, {averaged} ({unit})": raw["actual_exports_usd"],
        f"Blocked Opportunity value, {year}, {illustrative()} ({unit})": raw["blocked_opportunity_value_usd"],
        f"Potential Range low, {averaged} ({unit})": raw["potential_range_low_usd"],
        f"Potential Range high, {averaged} ({unit})": raw["potential_range_high_usd"],
    }
    money = ["Export Gap", "Export Potential", "Actual Exports", "Blocked Opportunity value"]
    columns = ["HS6", "Product", "Sector", "Market", "Region"]
    rest = [column for column in table.columns if column not in {*columns, *money}]
    return pd.concat([table[columns], pd.DataFrame(dollars), table[rest]], axis=1)
