"""A cap on Market Reach's upside, scored by the experiment loop."""

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
MEASURES = {
    "selection_measure",
    "non_zero_spearman",
    "zero_vs_non_zero_spearman",
    "value_weighted_spearman",
    "top_ten_share",
    "tweedie_deviance",
}


@pytest.fixture(scope="module")
def data(fixture_world):
    return loop.fixture_data(fixture_world)


def _no_training(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a model was trained")

    monkeypatch.setattr(xgb, "train", refuse)


def test_with_no_cap_the_scores_are_exactly_the_uncapped_scores(data):
    lp = loop.Loop(data, REACH_LOOP)
    reach = replace(lp.plan.start, market_reach_beta=BETA)
    uncapped = lp.export_potential(reach, lp.grid).to_numpy()

    assert reach.market_reach_cap is None  # the default is no cap
    assert "market_reach_cap" not in reach.to_dict()  # so settings saved before it existed compare equal
    # A cap above every Market's multiplier caps nothing.
    roomy = lp.export_potential(replace(reach, market_reach_cap=1e12), lp.grid).to_numpy()
    np.testing.assert_array_equal(roomy, uncapped)


def _fold_0_offsets(world, markets: pl.Series) -> np.ndarray:
    """Fold 0's offsets for 2022, from the 2021 UK grid built here from the world's Flows."""
    flows = world.flows.filter(pl.col("year") <= 2021)
    grid = build_features(uk_grid(flows.filter(pl.col("year") == 2021), baci.UK), flows, world.sources)
    rows = assign_folds(grid.rename({"importer": "iso3"}), 2).rename({"iso3": "importer"})
    table = market_reach.reach_table(rows.with_columns(year=pl.lit(2021)), 2022, n_folds=2)
    offsets = dict(table.filter(pl.col("fold") == 0).select("importer", "offset").iter_rows())
    return np.array([offsets.get(m, 0.0) for m in markets.cast(pl.String)])


def test_a_cap_holds_every_markets_multiplier_at_or_below_it_and_keeps_the_summed_export_potential(
    data, fixture_world
):
    lp = loop.Loop(data, REACH_LOOP)
    plain = lp.export_potential(lp.plan.start, lp.grid).to_numpy()
    offsets = _fold_0_offsets(fixture_world, lp.grid["importer"])
    cap = 0.4  # between the fixture's multipliers (all below 1: every offset is negative)

    capped = lp.export_potential(replace(lp.plan.start, market_reach_beta=BETA, market_reach_cap=cap), lp.grid).to_numpy()

    assert np.exp(BETA * offsets).min() < cap < np.exp(BETA * offsets).max()  # the cap bites on some Markets only
    constant = capped / (plain * np.minimum(np.exp(BETA * offsets), cap))
    assert constant == pytest.approx(np.full(len(constant), constant[0]))
    assert capped.sum() == pytest.approx(plain.sum())


def test_a_cap_arm_trains_no_model_and_returns_one_row_with_every_measure(data, monkeypatch):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    reach = replace(lp.plan.start, market_reach_beta=BETA)
    base = lp.run_arm(loop.Arm("uncapped"), reach)
    _no_training(monkeypatch)

    row = lp.run_arm(loop.market_reach_cap_arm(), reach)

    assert row["trained"] is False
    assert row["training_key"] == base["training_key"]
    assert MEASURES | {"rounds", "without_market_reach", "grid"} <= set(row)
    assert [v["market_reach_cap"] for v in row["grid"]] == list(loop.MARKET_REACH_CAP_GRID)
    chosen = max(row["grid"], key=lambda v: v["selection_measure"])
    assert row["settings"]["market_reach_cap"] == chosen["market_reach_cap"]
    assert row["selection_measure"] == chosen["selection_measure"]
    assert row["settings"]["market_reach_beta"] == BETA


def test_the_served_base_adds_market_reach_only_to_a_best_so_far_without_it():
    plain = loop.Settings(params={})
    reached = replace(plain, market_reach_beta=BETA, market_reach_cap=8.0)

    assert loop.served_base_arm(plain).apply(plain).market_reach_beta == market_reach.CHOSEN_BETA
    assert loop.served_base_arm(reached).apply(reached) == reached


def test_the_cap_experiment_trains_nothing_and_decides_against_the_best_so_far_as_served(data, tmp_path, monkeypatch):
    plan = replace(loop.FIXTURE_LOOP, min_win_threshold=2.0)  # nothing can clear it
    loop.Loop(data, plan, tmp_path).noise_floor()
    _no_training(monkeypatch)

    results = loop.run_market_reach_cap(loop.Loop(data, plan, tmp_path))

    section = results["experiments"][loop.MARKET_REACH_CAP_EXPERIMENT]
    base, (arm,) = section["base_arm"], section["arms"]
    assert base["trained"] is False and arm["trained"] is False
    assert base["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA
    assert "market_reach_cap" not in base["settings"]
    assert section["base"]["selection_measure"] == base["selection_measure"]
    assert section["kept"] is None
    assert results["best"]["experiment"] == "noise_floor"  # dropped: the best so far is unchanged


def test_a_kept_cap_becomes_part_of_the_best_so_far(data, tmp_path):
    loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).noise_floor()
    saved = loop.read_results(tmp_path)
    saved["noise_floor"]["win_threshold"] = -1.0  # any Arm wins, so the test can see a kept cap
    loop.save_results(tmp_path, saved)

    results = loop.run_market_reach_cap(loop.Loop(data, loop.FIXTURE_LOOP, tmp_path))

    section = results["experiments"][loop.MARKET_REACH_CAP_EXPERIMENT]
    assert section["kept"] == loop.MARKET_REACH_CAP
    best = loop.Settings.from_dict(results["best"]["settings"])
    assert best.market_reach_beta == market_reach.CHOSEN_BETA
    assert best.market_reach_cap in loop.MARKET_REACH_CAP_GRID
    # A later idea stacks on it: its served base changes nothing.
    assert loop.served_base_arm(best).apply(best) == best
