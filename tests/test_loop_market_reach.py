"""The experiment loop with Market Reach and a base retrained on rebuilt files."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from exportspredictor import baci, loop, market_reach
from exportspredictor.experiments import uk_grid
from exportspredictor.features import build_features
from exportspredictor.folds import assign_folds

# Two folds, so fold 0's offsets come from the other fold's pairs (one fold would leave none).
REACH_LOOP = replace(loop.FIXTURE_LOOP, n_folds=2)
BETA = 4.0


@pytest.fixture(scope="module")
def data(fixture_world):
    return loop.fixture_data(fixture_world)


def _no_training(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a model was trained")

    monkeypatch.setattr(xgb, "train", refuse)


def _fold_0_offsets(world, markets: pl.Series) -> np.ndarray:
    """Fold 0's offsets for 2022, from the 2021 UK grid built here from the world's Flows."""
    flows = world.flows.filter(pl.col("year") <= 2021)
    grid = build_features(uk_grid(flows.filter(pl.col("year") == 2021), baci.UK), flows, world.sources)
    rows = assign_folds(grid.rename({"importer": "iso3"}), 2).rename({"iso3": "importer"})
    table = market_reach.reach_table(rows.with_columns(year=pl.lit(2021)), 2022, n_folds=2)
    offsets = dict(table.filter(pl.col("fold") == 0).select("importer", "offset").iter_rows())
    return np.array([offsets.get(m, 0.0) for m in markets.cast(pl.String)])


def test_market_reach_moves_each_held_out_markets_potential_by_its_lagged_offset_and_trains_nothing(
    data, fixture_world, monkeypatch
):
    lp = loop.Loop(data, REACH_LOOP)
    plain = lp.export_potential(lp.plan.start, lp.grid).to_numpy()
    _no_training(monkeypatch)

    reached = lp.export_potential(replace(lp.plan.start, market_reach_beta=BETA), lp.grid).to_numpy()

    offsets = _fold_0_offsets(fixture_world, lp.grid["importer"])
    assert np.ptp(offsets) > 0.05  # the Markets differ, so the test can tell
    constant = reached / (plain * np.exp(BETA * offsets))
    assert constant == pytest.approx(np.full(len(constant), constant[0]))
    assert reached.sum() == pytest.approx(plain.sum())  # the held-out grid's summed Export Potential is kept


def test_an_arm_scored_with_market_reach_also_records_its_measures_without_it(data):
    lp = loop.Loop(data, REACH_LOOP)
    without = lp.run_arm(loop.Arm("without"), lp.plan.start)

    row = lp.run_arm(loop.Arm("with Market Reach", {"market_reach_beta": BETA}), lp.plan.start)

    assert row["settings"]["market_reach_beta"] == BETA
    assert row["training_key"] == without["training_key"]
    assert row["without_market_reach"]["selection_measure"] == without["selection_measure"]
    assert row["without_market_reach"]["tweedie_deviance"] == pytest.approx(without["tweedie_deviance"])


def test_an_experiment_with_a_base_arm_retrains_the_best_so_far_on_its_own_files_and_decides_against_that(
    data, tmp_path
):
    plan = replace(loop.FIXTURE_LOOP, min_win_threshold=2.0)  # nothing can clear it
    loop.Loop(data, plan, tmp_path).noise_floor()
    rebuilt = replace(data, feature_set="exporter_count")
    base = loop.Arm("base on rebuilt files", {"market_reach_beta": BETA})
    candidate = loop.Arm("+ depth 3", {"params": {"max_depth": 3}, "market_reach_beta": BETA})

    results = loop.Loop(rebuilt, plan, tmp_path).run_experiment("rebuilt", [candidate], base_arm=base)

    section = results["experiments"]["rebuilt"]
    assert section["feature_set"] == "exporter_count"
    base_row = section["base_arm"]
    assert base_row["trained"] is True  # a fresh model on the rebuilt files
    assert base_row["settings"]["market_reach_beta"] == BETA
    assert section["base"]["arm"] == "base on rebuilt files"
    assert section["base"]["selection_measure"] == base_row["selection_measure"]
    assert section["previous_best"]["experiment"] == "noise_floor"
    assert section["kept"] is None
    assert results["best"]["experiment"] == "noise_floor"  # dropped: the best so far is unchanged


def test_an_arm_that_beats_the_retrained_base_becomes_the_best_so_far(data, tmp_path):
    plan = replace(loop.FIXTURE_LOOP, min_win_threshold=0.0)  # the seeds agree, so any better arm wins
    loop.Loop(data, plan, tmp_path).noise_floor()
    base = loop.Arm("base", {"market_reach_beta": BETA})
    candidate = loop.Arm("three quarters naive", {"naive_weight": 0.75, "market_reach_beta": BETA})

    results = loop.Loop(data, plan, tmp_path).run_experiment("reach", [candidate], base_arm=base)

    assert results["experiments"]["reach"]["kept"] == "three quarters naive"
    assert results["best"]["arm"] == "three quarters naive"
    assert results["best"]["settings"]["market_reach_beta"] == BETA


def test_the_exporter_count_experiment_adds_the_feature_to_the_retrained_base_both_with_market_reach(data, tmp_path):
    lp = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path)

    results = loop.run_exporter_count(lp)

    section = results["experiments"][loop.EXPORTER_COUNT_EXPERIMENT]
    base, (arm,) = section["base_arm"], section["arms"]
    assert base["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA
    assert base["settings"]["naive_weight"] == results["noise_floor"]["arms"][0]["settings"]["naive_weight"]
    assert arm["settings"]["features"] == [*base["settings"]["features"], "log_exporters_loo"]
    assert arm["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA
    assert [v["naive_weight"] for v in arm["grid"]] == list(loop.BLEND_WEIGHTS)  # its own blend weight
    assert arm["trained"] is True
