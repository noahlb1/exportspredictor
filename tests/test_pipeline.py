"""Training through the package: fixture world in, a Baseline table out-of-fold for every UK pair."""

import polars as pl
import pytest

from exportspredictor.fixture import UK
from exportspredictor.folds import assign_folds


@pytest.fixture
def world(fixture_world):
    return fixture_world


@pytest.fixture
def artefacts(fixture_artefacts):
    return fixture_artefacts


def test_baseline_has_one_row_per_uk_product_market_pair(artefacts):
    baseline = artefacts.baseline

    assert baseline.height == 5 * 3
    assert baseline.select("hs6", "iso3").is_duplicated().sum() == 0


def test_baseline_carries_actual_exports_from_the_baseline_year(world, artefacts):
    uk_2024 = world.flows.filter(pl.col("exporter") == UK, pl.col("year") == 2024).select(
        "hs6", iso3="importer", expected="value_usd"
    )
    joined = artefacts.baseline.join(uk_2024, on=["hs6", "iso3"])

    assert joined.height == 15
    assert (joined["actual_exports_usd"] == joined["expected"]).all()


def test_export_gap_is_potential_minus_actual(artefacts):
    row = artefacts.baseline.row(0, named=True)

    assert row["export_potential_usd"] > 0
    assert row["export_gap_usd"] == pytest.approx(
        row["export_potential_usd"] - row["actual_exports_usd"]
    )


def test_metadata_records_version_baseline_year_and_folds(artefacts):
    assert artefacts.metadata["version"] == "fixture-test"
    assert artefacts.metadata["baseline_year"] == 2024
    assert len(artefacts.fold_models) == artefacts.metadata["n_folds"] == 5


def test_fold_assignment_is_deterministic_and_keeps_a_pair_in_one_fold(world):
    uk = world.flows.filter(pl.col("exporter") == UK).select("hs6", iso3="importer")

    first, second = assign_folds(uk), assign_folds(uk)

    assert first.equals(second)
    assert first.group_by("hs6", "iso3").agg(pl.col("fold").n_unique())["fold"].max() == 1
