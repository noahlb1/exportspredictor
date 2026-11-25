"""The wider Exporter scope tried by the experiment loop."""

from dataclasses import replace

import pytest

from exportspredictor import loop, market_reach
from exportspredictor.loop.served import kept_feature_set
from exportspredictor.training import RowBudgetExceeded

# On the fixture, the top 2 Exporters by mean GDP are DEU and USA; with the UK, FRA is the one left out.
TOP = 2


@pytest.fixture(scope="module")
def data(fixture_world):
    return loop.fixture_data(fixture_world)


def test_an_arm_with_an_exporter_scope_trains_only_on_the_top_exporters_rows(data):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    every = lp.run_arm(loop.Arm("every Exporter"), loop.FIXTURE_LOOP.start)
    top = lp.run_arm(loop.Arm("top 2", {"exporter_scope": TOP}), loop.FIXTURE_LOOP.start)

    # 80 fixture training rows, 20 per Exporter; the UK's 20 are held out, and FRA's 20 are out of scope.
    assert every["training_rows"] == 60
    assert top["training_rows"] == 40
    assert top["settings"]["exporter_scope"] == TOP
    assert top["training_key"] != every["training_key"]


def test_every_exporter_is_the_default_and_keeps_every_cached_key():
    assert "exporter_scope" not in loop.Settings(params={}).to_dict()


def test_the_wider_scope_arm_runs_against_the_best_so_far_as_served_on_the_top_exporters_of_the_same_files(data, tmp_path):
    lp = loop.Loop(replace(data, feature_set="all_exporters"), loop.FIXTURE_LOOP, tmp_path)

    results = loop.run_wider_scope(lp, top=TOP, non_uk_rate=1.0)

    section = results["experiments"][loop.WIDER_SCOPE_EXPERIMENT]
    base, (arm,) = section["base_arm"], section["arms"]
    assert section["feature_set"] == "all_exporters"
    assert section["files_kept_if_dropped"] is False
    assert base["settings"]["exporter_scope"] == TOP
    assert base["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA
    assert "exporter_scope" not in arm["settings"]  # every Exporter
    assert arm["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA
    assert base["settings"]["non_uk_rate"] == arm["settings"]["non_uk_rate"] == 1.0  # one rate: only the scope differs
    assert [v["naive_weight"] for v in arm["grid"]] == list(loop.BLEND_WEIGHTS)
    assert arm["training_rows"] > base["training_rows"]


def test_a_dropped_wider_scope_leaves_the_served_files_unchanged(data, tmp_path):
    plan = replace(loop.FIXTURE_LOOP, min_win_threshold=2.0)  # nothing can clear it
    served = loop.Loop(replace(data, feature_set="exporter_count"), plan, tmp_path)
    loop.run_market_reach_cap(served)
    lp = loop.Loop(replace(data, feature_set="all_exporters"), plan, tmp_path)

    results = loop.run_wider_scope(lp, top=TOP, non_uk_rate=1.0)

    assert results["experiments"][loop.WIDER_SCOPE_EXPERIMENT]["kept"] is None
    assert kept_feature_set(results) == "exporter_count"


def test_an_experiment_whose_rows_would_not_fit_the_row_budget_is_refused_before_training_anything(data, tmp_path):
    loop.Loop(data, loop.FIXTURE_LOOP, tmp_path).noise_floor()
    before = {p.name for p in (tmp_path / loop.MODELS_DIR).iterdir()}
    lp = loop.Loop(data, replace(loop.FIXTURE_LOOP, row_budget=150), tmp_path)

    with pytest.raises(RowBudgetExceeded, match="largest rate that fits"):
        lp.run_experiment("too many rows", [loop.Arm("every row", {"non_uk_rate": 1.0, "params": {"max_depth": 3}})])

    assert {p.name for p in (tmp_path / loop.MODELS_DIR).iterdir()} == before
