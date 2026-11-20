"""Sanctioned Markets and Restricted Pairs (ADR 0005): two Caveat Tiers read from hand-kept, dated tables."""

from datetime import date

import polars as pl

from exportspredictor.caveats import (
    RESTRICTED_PAIR,
    SANCTIONED_MARKET,
    CaveatContext,
    restricted_pair_tier,
    sanctioned_market_tier,
    with_caveat_columns,
)
from exportspredictor.restrictions import (
    PROHIBITIONS,
    SANCTIONS_REGIMES,
    VERIFIED,
    Prohibition,
    SanctionsRegime,
)

WHISKY, BEER, ETHANOL, CARS = "220830", "220300", "220710", "870323"


def _marked(rows: dict, year: int, *tiers) -> pl.DataFrame:
    return with_caveat_columns(pl.DataFrame(rows), CaveatContext(baseline_year=year), tiers=tiers)


def _sanctioned(iso3s: list[str], year: int) -> dict[str, str | None]:
    """Each Market's Sanctioned Market reason in `year` (None where it is not one)."""
    marked = _marked({"hs6": [CARS] * len(iso3s), "iso3": iso3s}, year, SANCTIONED_MARKET)
    return {
        row["iso3"]: row["sanctioned_market_reason"] if row["sanctioned_market"] else None
        for row in marked.iter_rows(named=True)
    }


def test_russia_and_belarus_are_sanctioned_markets_from_2022_and_not_before():
    assert _sanctioned(["RUS", "BLR", "FRA"], 2021) == {"RUS": None, "BLR": None, "FRA": None}
    assert _sanctioned(["RUS", "BLR", "FRA"], 2024) == {
        "RUS": "the UK's Russia sanctions regime, broad since 2022",
        "BLR": "the UK's Republic of Belarus sanctions regime, broad since 2022",
        "FRA": None,
    }


def test_north_korea_is_a_sanctioned_market_and_iran_is_not_as_its_regimes_are_targeted():
    assert _sanctioned(["PRK", "IRN"], 2024) == {
        "PRK": "the UK's Democratic People's Republic of Korea sanctions regime, broad since 2007",
        "IRN": None,
    }


def test_syria_is_a_sanctioned_market_until_its_regime_narrowed_in_2025():
    assert _sanctioned(["SYR"], 2024)["SYR"] == "the UK's Syria sanctions regime, broad since 2011"
    assert _sanctioned(["SYR"], 2025)["SYR"] is None


def test_alcoholic_beverages_into_markets_that_ban_them_are_restricted_pairs_and_ethyl_alcohol_is_not():
    rows = {
        "hs6": [WHISKY, BEER, ETHANOL, CARS, WHISKY, WHISKY, WHISKY],
        "iso3": ["SAU", "KWT", "SAU", "SAU", "LBY", "IRN", "JPN"],
    }
    marked = _marked(rows, 2024, RESTRICTED_PAIR)

    assert marked["restricted_pair"].to_list() == [True, True, False, False, True, True, False]
    assert marked["restricted_pair_reason"].to_list() == [
        "Saudi Arabia prohibits imports of alcoholic beverages",
        "Kuwait prohibits imports of alcoholic beverages",
        None,
        None,
        "Libya prohibits imports of alcoholic beverages",
        "Iran prohibits imports of alcoholic beverages",
        None,
    ]
    assert RESTRICTED_PAIR.caveat_text(marked["restricted_pair_reason"][0]) == (
        "Restricted Pair: Saudi Arabia prohibits imports of alcoholic beverages. It is scored and ranked as normal, "
        "but it is not an open lead."
    )


def test_a_regime_flags_only_the_baseline_years_it_covers():
    ended = SanctionsRegime("JPN", "Ended", 2015, 2020, "https://www.gov.uk/x", "https://www.legislation.gov.uk/x")
    later = SanctionsRegime("CAN", "Later", 2030, None, "https://www.gov.uk/y", "https://www.legislation.gov.uk/y")
    tier = sanctioned_market_tier((ended, later))
    rows = {"hs6": [CARS, CARS], "iso3": ["JPN", "CAN"]}

    assert _marked(rows, 2014, tier)["sanctioned_market"].to_list() == [False, False]
    assert _marked(rows, 2015, tier)["sanctioned_market"].to_list() == [True, False]
    assert _marked(rows, 2020, tier)["sanctioned_market"].to_list() == [True, False]
    assert _marked(rows, 2024, tier)["sanctioned_market"].to_list() == [False, False]
    assert _marked(rows, 2030, tier)["sanctioned_market"].to_list() == [False, True]


def test_a_prohibition_covers_every_product_under_its_codes():
    tier = restricted_pair_tier((Prohibition("IND", "India", "test goods", ("22", "870323"), "https://example.org"),))
    rows = {"hs6": [WHISKY, CARS, "870324"], "iso3": ["IND", "IND", "IND"]}

    assert _marked(rows, 2024, tier)["restricted_pair"].to_list() == [True, True, False]


def test_every_entry_carries_a_source_and_the_tables_a_verification_date():
    assert VERIFIED == date(2026, 11, 20)
    for regime in SANCTIONS_REGIMES:
        assert regime.source.startswith("https://www.gov.uk/government/publications/")
        assert regime.evidence.startswith("https://www.legislation.gov.uk/")
    for prohibition in PROHIBITIONS:
        assert prohibition.source.startswith("https://")
    assert {regime.iso3 for regime in SANCTIONS_REGIMES} == {"RUS", "BLR", "PRK", "SYR"}
    assert {prohibition.iso3 for prohibition in PROHIBITIONS} == {"SAU", "KWT", "LBY", "IRN"}
