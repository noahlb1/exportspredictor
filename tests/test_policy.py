"""Trade Agreement, Customs Union and Trading Bloc features from the WTO RTA-IS export.

Expected values are known in-force dates from official sources
(docs/research/trade-agreement-coverage.md), mapped to years by the 1 July rule.
Each test runs on a committed extract of real WTO records and, when it has been
downloaded, on the full export in the data directory.
"""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from exportspredictor import policy

EXTRACT = Path(__file__).parent / "data" / "wto_rta_extract.csv"
YEARS = list(range(2012, 2028))


@pytest.fixture(scope="module", params=["extract", "full export"])
def rtas(request):
    if request.param == "extract":
        return policy.read_rta_export(EXTRACT)
    full = policy.rta_export_path()
    if not full.exists():
        pytest.skip(f"WTO export not downloaded to {full}")
    return policy.read_rta_export(full)


@pytest.fixture(scope="module")
def agreements(rtas):
    return policy.pair_agreements(rtas, YEARS)


def _flag(agreements, exporter, importer, year, column="trade_agreement"):
    row = agreements.filter(
        (pl.col("exporter") == exporter) & (pl.col("importer") == importer) & (pl.col("year") == year)
    )
    return int(row[column].item()) if row.height else 0


def test_eu_japan_in_force_from_2019(agreements):
    for eu_member in ["FRA", "DEU", "HRV"]:
        assert _flag(agreements, eu_member, "JPN", 2018) == 0
        assert _flag(agreements, eu_member, "JPN", 2019) == 1
        assert _flag(agreements, "JPN", eu_member, 2019) == 1


def test_uk_stays_in_the_eu_customs_union_for_2020_and_moves_to_an_agreement_in_2021(agreements):
    for year, customs_union in [(2019, 1), (2020, 1), (2021, 0), (2024, 0)]:
        assert _flag(agreements, "GBR", "FRA", year) == 1
        assert _flag(agreements, "GBR", "FRA", year, "customs_union") == customs_union
        assert _flag(agreements, "DEU", "GBR", year, "customs_union") == customs_union


def test_uk_eu_agreement_covers_members_that_join_by_bloc(agreements):
    # EU - United Kingdom lists only "European Union" as the EU party.
    assert _flag(agreements, "GBR", "HRV", 2021) == 1
    assert _flag(agreements, "HRV", "GBR", 2024) == 1


def test_uk_australia_and_uk_new_zealand_in_force_from_2023(agreements):
    for partner in ["AUS", "NZL"]:
        assert _flag(agreements, "GBR", partner, 2022) == 0
        assert _flag(agreements, "GBR", partner, 2023) == 1
        assert _flag(agreements, partner, "GBR", 2023) == 1
        assert _flag(agreements, "GBR", partner, 2023, "customs_union") == 0


@pytest.fixture(scope="module")
def blocs(rtas):
    return policy.bloc_membership(rtas, YEARS)


def _member(blocs, iso3, year, bloc):
    row = blocs.filter((pl.col("iso3") == iso3) & (pl.col("year") == year))
    return int(row[bloc].item()) if row.height else 0


def test_trading_bloc_membership_per_country_per_year(blocs):
    assert set(policy.TRADING_BLOCS) <= set(blocs.columns)
    assert blocs.select("iso3", "year").is_duplicated().sum() == 0
    assert _member(blocs, "FRA", 2012, "eu") == 1
    assert _member(blocs, "HRV", 2013, "eu") == 0  # joined 1 July 2013, not before it
    assert _member(blocs, "HRV", 2014, "eu") == 1
    assert _member(blocs, "CHE", 2020, "efta") == 1
    assert _member(blocs, "JPN", 2019, "cptpp") == 1
    assert _member(blocs, "MEX", 2026, "cptpp") == 1


def test_uk_leaves_the_eu_bloc_in_2020(blocs):
    assert _member(blocs, "GBR", 2019, "eu") == 1
    assert _member(blocs, "GBR", 2020, "eu") == 0
    assert _member(blocs, "GBR", 2024, "efta") == 0


def test_uk_cptpp_accession_counts_from_2025_not_2024(blocs, agreements):
    assert _member(blocs, "GBR", 2024, "cptpp") == 0
    assert _member(blocs, "GBR", 2025, "cptpp") == 1
    # Canada ratified on 1 Sep 2026, after 1 July, so the UK-Canada pair joins in 2027
    # (the full export also has the UK-Canada continuity agreement, so check bloc only).
    assert _flag(agreements, "GBR", "MEX", 2026) == 1


def test_attach_policy_fills_every_policy_feature_for_any_rows(rtas):
    rows = pl.DataFrame(
        {"exporter": ["GBR", "GBR", "GBR"], "importer": ["FRA", "IND", "JPN"], "year": [2020, 2020, 2025]},
        schema={"exporter": pl.String, "importer": pl.String, "year": pl.Int32},
    )
    out = policy.attach_policy(rows, policy.pair_agreements(rtas, YEARS), policy.bloc_membership(rtas, YEARS))
    assert out.columns == [*rows.columns, *policy.POLICY_FEATURES]
    by_importer = {r["importer"]: r for r in out.iter_rows(named=True)}
    assert (by_importer["FRA"]["trade_agreement"], by_importer["FRA"]["customs_union"]) == (1, 1)
    assert (by_importer["FRA"]["eu_o"], by_importer["FRA"]["eu_d"]) == (0, 1)
    assert (by_importer["IND"]["trade_agreement"], by_importer["IND"]["eu_d"]) == (0, 0)
    assert (by_importer["JPN"]["cptpp_o"], by_importer["JPN"]["cptpp_d"]) == (1, 1)


def test_build_writes_reproducible_policy_tables(tmp_path):
    written = policy.build(tmp_path, years=YEARS, source=EXTRACT)
    agreements = pl.read_parquet(written["agreements"])
    blocs = pl.read_parquet(written["trading_blocs"])
    policy.build(tmp_path, years=YEARS, source=EXTRACT)
    assert_frame_equal(pl.read_parquet(written["agreements"]), agreements)
    assert_frame_equal(pl.read_parquet(written["trading_blocs"]), blocs)
    assert _flag(agreements, "GBR", "AUS", 2023) == 1
    assert _member(blocs, "GBR", 2019, "eu") == 1
    assert all(path.is_relative_to(tmp_path) for path in written.values())
