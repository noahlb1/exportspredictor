"""The one ranking query both Opportunity views use: filters, order, limit and flags."""

from exportspredictor import rankings

FIXTURE_HS6 = "220830"


def test_rank_orders_name_a_column_a_title_and_a_description():
    assert list(rankings.RANK_BY) == ["gap", "blocked"]
    assert rankings.RANK_BY["gap"].title == "Export Gap"
    assert rankings.RANK_BY["blocked"].title == "Blocked Opportunity value"
    # The description states the default Blocked Opportunity thresholds (20% and $1M).
    assert "20%" in rankings.RANK_BY["blocked"].description
    assert "$1M" in rankings.RANK_BY["blocked"].description


def test_ranks_by_export_gap_by_default(fixture_artefacts):
    rows = rankings.rank_opportunities(fixture_artefacts, iso3="JPN")

    gaps = rows["export_gap_usd"].to_list()
    assert gaps == sorted(gaps, reverse=True)


def test_ranks_by_blocked_opportunity_value_then_limits(fixture_artefacts):
    everything = rankings.rank_opportunities(fixture_artefacts, hs6=FIXTURE_HS6, rank_by="blocked")
    top = rankings.rank_opportunities(fixture_artefacts, hs6=FIXTURE_HS6, rank_by="blocked", limit=1)

    values = everything["blocked_effect_usd"].to_list()
    assert values == sorted(values, reverse=True)
    assert top["iso3"].to_list() == ["IND"]  # the only fixture Market without a UK Trade Agreement


def test_rows_carry_the_blocked_opportunity_flag(fixture_artefacts):
    rows = rankings.rank_opportunities(fixture_artefacts, iso3="JPN")

    # Japan has a UK Trade Agreement, so no pair has a Blocked Opportunity value.
    assert rows["blocked_opportunity"].to_list() == [False] * rows.height
