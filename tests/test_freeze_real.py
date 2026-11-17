"""The real v1 freeze, checked on disk; skips until `python -m exportspredictor.freeze` has run."""

import json

import polars as pl
import pytest

from exportspredictor import backtests, freeze
from exportspredictor.features import features_dir
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


@pytest.fixture(scope="module")
def v1_features():
    """v1's feature files, which the frozen tables were drawn from; skips once they are rebuilt for v2."""
    settings = features_dir() / "features.json"
    if not settings.exists() or "exporter_scope" in json.loads(settings.read_text(encoding="utf-8")):
        pytest.skip("The feature files on disk are no longer v1's (rebuilt for v2), so there is no source to compare")
    return features_dir()


def _scan(frozen, name):
    return pl.scan_parquet(frozen / freeze.TABLES[name])


def test_the_manifest_names_the_v1_tag_and_v1s_source_models(manifest):
    assert manifest["tag"] == "v1"
    assert len(manifest["commit"]) == 40 and manifest["dirty"] is False
    assert manifest["source_models"] == freeze.V1_IDS


def test_the_uk_grid_covers_the_2023_grid_out_of_fold(frozen, v1_features):
    grid = _scan(frozen, "uk_grid").select("hs6", "iso3", "fold").collect()
    source = pl.scan_parquet(v1_features / "uk_grid_2023.parquet").select(pl.len()).collect().item()

    assert grid.height == source
    assert grid.select("hs6", "iso3").is_duplicated().sum() == 0
    assert (grid["fold"] == assign_folds(grid.select("hs6", "iso3"))["fold"]).all()


def test_the_test_rows_are_v1s_2023_rows_with_their_weights(frozen, v1_features):
    totals = [pl.len().alias("rows"), pl.col("weight").sum(), pl.col("value_usd").sum()]
    rows = _scan(frozen, "test_rows").select(*totals).collect()
    source = pl.scan_parquet(v1_features / "training_2023.parquet").select(*totals).collect()

    assert rows["rows"].item() == source["rows"].item()
    assert rows["weight"].item() == pytest.approx(source["weight"].item())
    assert rows["value_usd"].item() == pytest.approx(source["value_usd"].item())


def test_lift_and_the_event_backtest_cover_their_years_and_directions(frozen):
    lift_years = _scan(frozen, "lift").select(pl.col("year").unique()).collect()["year"]
    held = _scan(frozen, "lift_held_out_exporters").collect()["exporter"]
    event = _scan(frozen, "event_backtest").select("direction", "year").unique().collect()

    assert sorted(lift_years) == list(freeze.V1_LIFT_YEARS)
    assert set(_scan(frozen, "lift").select(pl.col("exporter").unique()).collect()["exporter"]) <= set(held)
    assert event.height == len(backtests.EU_JAPAN.directions) * len(backtests.EU_JAPAN.post_years)
