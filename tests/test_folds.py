"""Fold guards (ADR 0001): UK pairs are assigned to folds by pair, and scored out-of-fold.

Fold models are trained through the package on the fixture world's feature table, then a
held-out pair's training labels are distorted to show which models could have seen it.
"""

import polars as pl
import pytest

from exportspredictor.features import build_features, training_rows, uk_grid
from exportspredictor.fixture import UK
from exportspredictor.folds import assign_folds, score_out_of_fold, train_fold_models

PARAMS = {"max_depth": 3, "eta": 0.3}
ROUNDS = 20


@pytest.fixture(scope="module")
def training(fixture_world):
    flows = fixture_world.flows.filter(pl.col("year") < 2024)
    return build_features(training_rows(flows, rate=1.0), flows, fixture_world.sources)


@pytest.fixture(scope="module")
def grid(fixture_world):
    flows = fixture_world.flows.filter(pl.col("year") == 2024)
    return build_features(uk_grid(flows, UK), flows, fixture_world.sources)


def _train(chunks):
    return train_fold_models(chunks, uk=UK, params=PARAMS, rounds=ROUNDS, device="cpu")


def test_fold_assignment_is_deterministic_and_keeps_all_years_of_a_pair_together(training):
    uk_rows = training.filter(pl.col("exporter") == UK).select("hs6", "year", iso3="importer")

    first, second = assign_folds(uk_rows), assign_folds(uk_rows.reverse())

    assert first.sort("hs6", "iso3", "year").equals(second.sort("hs6", "iso3", "year"))
    assert uk_rows["year"].n_unique() > 1
    assert first.group_by("hs6", "iso3").agg(pl.col("fold").n_unique())["fold"].max() == 1
    assert first["fold"].n_unique() > 1


def test_no_uk_pair_is_scored_by_a_model_that_trained_on_it(training, grid):
    held_out = assign_folds(grid.select("hs6", iso3="importer")).filter(pl.col("fold") == 0)
    in_fold_0 = (pl.col("exporter") == UK) & pl.struct("hs6", "importer").is_in(
        held_out.select("hs6", importer="iso3").to_struct().implode()
    )
    distorted = training.with_columns(
        value_usd=pl.when(in_fold_0).then(pl.col("value_usd") * 1000 + 1e9).otherwise("value_usd")
    )
    assert training.filter(in_fold_0).height > 0

    before = score_out_of_fold(grid, _train([training]), uk=UK)
    after = score_out_of_fold(grid, _train([distorted]), uk=UK)

    fold_0 = pl.col("fold") == 0
    # Fold 0's model never saw the distorted labels, so its pairs score exactly as before ...
    assert before.filter(fold_0)["export_potential_usd"].equals(after.filter(fold_0)["export_potential_usd"])
    # ... while every other fold model trained on them and moved.
    assert (before.filter(~fold_0)["export_potential_usd"] != after.filter(~fold_0)["export_potential_usd"]).all()


def test_scoring_records_each_pairs_fold_and_covers_every_uk_row(training, grid):
    scored = score_out_of_fold(grid, _train([training]), uk=UK)
    expected = assign_folds(grid.select("hs6", iso3="importer"))

    assert scored.height == grid.height
    assert scored.select("hs6", "importer", "fold").sort("hs6", "importer").equals(
        expected.select("hs6", importer="iso3", fold="fold").sort("hs6", "importer")
    )


def test_training_from_yearly_chunks_matches_training_from_one_table(training, grid, tmp_path):
    paths = []
    for (year,), part in training.partition_by("year", as_dict=True).items():
        paths.append(tmp_path / f"training_{year}.parquet")
        part.write_parquet(paths[-1])

    whole = score_out_of_fold(grid, _train([training]), uk=UK)
    chunked = score_out_of_fold(grid, _train(paths), uk=UK)

    assert chunked["export_potential_usd"].to_list() == pytest.approx(whole["export_potential_usd"].to_list(), rel=1e-5)
