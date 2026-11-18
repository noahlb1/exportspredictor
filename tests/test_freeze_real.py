"""The real v1 freeze, checked on disk; skips until `python -m exportspredictor.freeze` has run."""

import polars as pl
import pytest

from exportspredictor import backtests, freeze
from exportspredictor.folds import assign_folds
from exportspredictor.paths import artefacts_dir


@pytest.fixture(scope="module")
def frozen():
    folder = artefacts_dir() / freeze.V1_IDS["artefact"] / freeze.FOLDER
    if not (folder / freeze.MANIFEST_FILE).exists():
        pytest.skip(f"v1 has not been frozen into {folder}")
    return folder


@pytest.fixture(scope="module")
def manifest(frozen):
    return freeze.verify(frozen)


def _scan(frozen, name):
    return pl.scan_parquet(frozen / freeze.TABLES[name])


def test_the_manifest_names_the_v1_tag_and_v1s_source_models(manifest):
    assert manifest["tag"] == "v1"
    assert len(manifest["commit"]) == 40 and manifest["dirty"] is False
    assert manifest["source_models"] == freeze.V1_IDS


# After the freeze, v1's feature files are rebuilt for v2 and v1 is never re-loaded, so the
# frozen tables are checked against their own manifest, never against the live feature files.


def test_the_uk_grid_covers_the_2023_grid_out_of_fold(frozen, manifest):
    grid = _scan(frozen, "uk_grid").select("hs6", "iso3", "fold").collect()

    assert grid.height == manifest["tables"]["uk_grid"]["rows"] > 1_000_000  # about 5,200 Products x 225 Markets
    assert grid.select("hs6", "iso3").is_duplicated().sum() == 0
    assert (grid["fold"] == assign_folds(grid.select("hs6", "iso3"))["fold"]).all()


def test_the_test_rows_are_v1s_2023_rows_with_their_weights(frozen, manifest):
    rows = _scan(frozen, "test_rows").select("year", "value_usd", "weight").collect()

    assert rows.height == manifest["tables"]["test_rows"]["rows"]
    assert set(rows["year"]) == {manifest["test_year"]} == {2023}
    # every positive Flow at weight 1 and v1's seeded 2% sample of Zero Cells at weight 50
    assert set(rows.filter(pl.col("value_usd") > 0)["weight"]) == {1.0}
    assert set(rows.filter(pl.col("value_usd") == 0)["weight"]) == {50.0}


def test_lift_and_the_event_backtest_cover_their_years_and_directions(frozen):
    lift_years = _scan(frozen, "lift").select(pl.col("year").unique()).collect()["year"]
    held = _scan(frozen, "lift_held_out_exporters").collect()["exporter"]
    event = _scan(frozen, "event_backtest").select("direction", "year").unique().collect()

    assert sorted(lift_years) == list(freeze.V1_LIFT_YEARS)
    assert set(_scan(frozen, "lift").select(pl.col("exporter").unique()).collect()["exporter"]) <= set(held)
    assert event.height == len(backtests.EU_JAPAN.directions) * len(backtests.EU_JAPAN.post_years)
