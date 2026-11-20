"""The one ranking query both Opportunity views use: filters, order, limit and flags."""


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

    values = everything["blocked_opportunity_value_usd"].to_list()
    assert values == sorted(values, reverse=True)
    assert top["iso3"].to_list() == ["IND"]  # the larger of the two fixture Markets without a UK Trade Agreement


def test_rows_carry_the_blocked_opportunity_flag(fixture_artefacts):
    rows = rankings.rank_opportunities(fixture_artefacts, iso3="JPN")

    # Japan has a UK Trade Agreement, so no pair has a Blocked Opportunity value.
    assert rows["blocked_opportunity"].to_list() == [False] * rows.height


def test_a_ranking_summarises_every_filtered_pair_before_its_limit(fixture_artefacts):
    everything = rankings.rank_opportunities(fixture_artefacts)

    top = rankings.ranking(fixture_artefacts, limit=3)

    assert top.rows["hs6"].to_list() == everything.head(3)["hs6"].to_list()
    assert top.summary == rankings.summarise(everything)


def test_hide_drops_the_rows_of_each_named_caveat_tier(fixture_artefacts):
    everything = rankings.rank_opportunities(fixture_artefacts)

    shown = rankings.rank_opportunities(fixture_artefacts, hide=["flagged_product", "reexport_hub"])

    expected = everything.filter(~everything["flagged_product"] & ~everything["reexport_hub"])
    assert shown.equals(expected) and 0 < shown.height < everything.height
