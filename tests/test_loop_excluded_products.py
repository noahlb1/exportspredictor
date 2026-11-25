"""Training on the Excluded Products' rows, tried by the experiment loop (spec #155, ticket #171)."""

from dataclasses import replace

import pytest

from exportspredictor import loop, market_reach
from exportspredictor.loop.served import kept_feature_set


@pytest.fixture(scope="module")
def data(fixture_world):
    """The fixture world's rows with the Excluded Product's training rows kept, as the `excluded_products` set."""
    return replace(loop.fixture_data(fixture_world, keep_excluded=True), feature_set="excluded_products")


def test_the_excluded_products_rows_train_only_an_arm_that_asks_for_them(data):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    dropped = lp.run_arm(loop.Arm("as always"), loop.FIXTURE_LOOP.start)
    kept = lp.run_arm(loop.Arm("train on them", {"train_excluded_products": True}), loop.FIXTURE_LOOP.start)

    # One training year (2021); 3 Exporters in scope (the UK held out), 4 Markets each: 60 rows of the five Products
    # (20 per Exporter), plus 12 Flows of the Excluded Product (270900), which every Exporter sells to every Market.
    assert dropped["training_rows"] == 60
    assert kept["training_rows"] == 72
    assert kept["settings"]["train_excluded_products"] is True
    assert kept["training_key"] != dropped["training_key"]


def test_dropping_them_is_the_default_and_keeps_every_cached_key():
    assert "train_excluded_products" not in loop.Settings(params={}).to_dict()


def test_the_excluded_products_arm_runs_against_the_best_so_far_as_served_on_the_same_files(data, tmp_path):
    lp = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path)

    results = loop.run_excluded_products(lp)

    section = results["experiments"][loop.EXCLUDED_PRODUCTS_EXPERIMENT]
    base, (arm,) = section["base_arm"], section["arms"]
    assert section["feature_set"] == "excluded_products"
    assert section["files_kept_if_dropped"] is False
    assert "train_excluded_products" not in base["settings"]
    assert arm["settings"]["train_excluded_products"] is True
    assert base["settings"]["market_reach_beta"] == arm["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA
    assert base["settings"]["non_uk_rate"] == arm["settings"]["non_uk_rate"]  # one rate: only the field differs
    assert [v["naive_weight"] for v in arm["grid"]] == list(loop.BLEND_WEIGHTS)
    assert arm["training_rows"] > base["training_rows"]


def test_a_dropped_excluded_products_arm_leaves_the_served_files_unchanged(data, fixture_world, tmp_path):
    plan = replace(loop.FIXTURE_LOOP, min_win_threshold=2.0)  # nothing can clear it
    served = loop.Loop(replace(loop.fixture_data(fixture_world), feature_set="exporter_count"), plan, tmp_path)
    loop.run_market_reach_cap(served)

    results = loop.run_excluded_products(loop.Loop(data, plan, tmp_path))

    assert results["experiments"][loop.EXCLUDED_PRODUCTS_EXPERIMENT]["kept"] is None
    assert kept_feature_set(results) == "exporter_count"
