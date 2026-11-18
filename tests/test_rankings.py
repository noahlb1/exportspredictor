"""The one ranking query both Opportunity views use: filters, order, limit and flags."""

import polars as pl
import pytest

from exportspredictor import rankings

FIXTURE_HS6 = "220830"


def test_rank_orders_name_a_column_a_title_and_a_description():
    # The Leaderboard and ranking pages offer gap and relative; Blocked Opportunities have their own page.
    assert list(rankings.RANK_BY) == ["gap", "relative"]
    assert rankings.RANK_BY["relative"].title == "Relative Export Gap"
    assert rankings.RANK_BY["gap"].title == "Export Gap"
    assert rankings.BLOCKED.title == "Blocked Opportunity value"
    # The description states the default Blocked Opportunity thresholds (20% and $1M).
    assert "20%" in rankings.BLOCKED.description
    assert "$1M" in rankings.BLOCKED.description
    # Adding the agreement removes the tariff too (ADR 0006).
    assert "and removing the tariff" in rankings.BLOCKED.description


def test_ranks_by_export_gap_by_default(fixture_artefacts):
    rows = rankings.rank_opportunities(fixture_artefacts, iso3="JPN")

    gaps = rows["export_gap_usd"].to_list()
    assert gaps == sorted(gaps, reverse=True)


def test_ranks_by_blocked_opportunity_value_then_limits(fixture_artefacts):
    everything = rankings.rank_opportunities(fixture_artefacts, hs6=FIXTURE_HS6, rank_by="blocked")
    top = rankings.rank_opportunities(fixture_artefacts, hs6=FIXTURE_HS6, rank_by="blocked", limit=1)

    values = everything["blocked_effect_usd"].to_list()
    assert values == sorted(values, reverse=True)
    assert top["iso3"].to_list() == ["IND"]  # the larger of the two fixture Markets without a UK Trade Agreement


def test_rows_carry_the_blocked_opportunity_flag(fixture_artefacts):
    rows = rankings.rank_opportunities(fixture_artefacts, iso3="JPN")

    # Japan has a UK Trade Agreement, so no pair has a Blocked Opportunity value.
    assert rows["blocked_opportunity"].to_list() == [False] * rows.height


def test_blocked_summary_counts_blocked_opportunities_and_totals_their_value(fixture_artefacts):
    ranked = rankings.rank_opportunities(fixture_artefacts, rank_by="blocked")
    # The fixture's values are too small for the $1M threshold, so flag the two largest as Blocked.
    rows = ranked.with_columns(blocked_opportunity=pl.int_range(pl.len()) < 2)

    summary = rankings.summarise_blocked(rows)

    assert summary["pairs"] == 20
    blocked = rows.filter("blocked_opportunity")
    unblocked = rows.filter(~pl.col("blocked_opportunity"))
    # Only Blocked Opportunities count towards the total, not every pair with a value.
    assert (unblocked["blocked_effect_usd"] > 0).any()
    assert summary["blocked_effect_usd"] == pytest.approx(blocked["blocked_effect_usd"].sum())
    assert summary["blocked_effect_usd"] > 0
    assert summary["blocked_opportunities"] == blocked.height
