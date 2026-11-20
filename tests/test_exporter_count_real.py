"""The real `exporter_count` feature set (#126): the Exporter count and the gravity fill on real data.

Skips unless `python -m exportspredictor.features --set exporter_count` has been run.
"""

import json

import polars as pl
import pytest

from exportspredictor import baci, features, gravity

SET = "exporter_count"
YEAR = 2022  # the validation year: the set is built up to it, so the test year is never read


@pytest.fixture(scope="module")
def built():
    folder = features.features_dir(feature_set_name=SET)
    if not (folder / features.SETTINGS_FILE).exists():
        pytest.skip(f"The {SET} feature set is not built under {folder}")
    return folder


def test_the_set_records_its_features_constraints_gravity_and_stops_before_the_test_year(built):
    settings = json.loads((built / features.SETTINGS_FILE).read_text(encoding="utf-8"))

    assert settings["feature_set"] == SET
    assert settings["candidate_features"] == ["log_exporters_loo"]
    assert "log_exporters_loo" not in settings["monotone_constraints"]
    assert max(settings["years"]) == YEAR
    assert settings["gravity"]["country_year"] == gravity.COUNTRY_YEAR_FILLED
    assert not (built / "training_2023.parquet").exists()


def test_every_scored_market_has_a_gdp_except_those_no_source_covers(built):
    grid = pl.scan_parquet(built / f"uk_grid_{YEAR}.parquet")
    missing = grid.filter(pl.col("log_gdp_d").is_null()).select(pl.col("importer").cast(pl.String).unique()).collect()

    assert set(missing["importer"]) == set(gravity.GDP_GAPS)
    assert grid.select(pl.col("log_distance").null_count()).collect().item() == 0


def test_no_uk_row_counts_the_uk_and_the_exporter_count_matches_the_flows(built):
    """On real data: each UK row's Exporter count is the Market's suppliers of the Product, the UK left out."""
    grid = pl.scan_parquet(built / f"uk_grid_{YEAR}.parquet").select(
        importer=pl.col("importer").cast(pl.String), hs6=pl.col("hs6").cast(pl.String), log_exporters_loo="log_exporters_loo"
    )
    flows = (
        pl.scan_parquet(baci.flows_dir() / f"flows_{YEAR}.parquet")
        .filter(pl.col("value_usd") > 0, pl.col("exporter").cast(pl.String) != baci.UK)
        .group_by(importer=pl.col("importer").cast(pl.String), hs6=pl.col("hs6").cast(pl.String))
        .agg(others=pl.len())
    )
    to_iso3 = dict(pl.read_parquet(baci.flows_dir().parent / "country_map.parquet").select("baci_iso3", "iso3").iter_rows())
    flows = flows.with_columns(pl.col("importer").replace(to_iso3))
    joined = grid.join(flows, on=["importer", "hs6"], how="left").with_columns(pl.col("others").fill_null(0)).collect()

    assert joined["log_exporters_loo"].null_count() == 0
    assert (joined["log_exporters_loo"].exp() - 1).round(3).cast(pl.Int64).equals(joined["others"].cast(pl.Int64),
                                                                                    check_names=False)
