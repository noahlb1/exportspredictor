"""Sibling Demand tried as a model feature by the experiment loop (spec #155, ticket #164)."""

from dataclasses import replace

import pytest

from exportspredictor import loop, market_reach
from exportspredictor.loop.served import kept_feature_set

FEATURE = "log_sibling_demand_loo"


@pytest.fixture(scope="module")
def data(fixture_world):
    return loop.fixture_data(fixture_world)


def test_the_sibling_demand_arm_runs_against_the_best_so_far_as_served_retrained_on_its_files(data, tmp_path):
    lp = loop.Loop(replace(data, feature_set="sibling_demand"), loop.FIXTURE_LOOP, tmp_path)

    results = loop.run_sibling_demand(lp)

    section = results["experiments"][loop.SIBLING_DEMAND_EXPERIMENT]
    base, (arm,) = section["base_arm"], section["arms"]
    assert section["feature_set"] == "sibling_demand"
    assert base["arm"] == loop.SERVED_BASE
    assert base["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA  # the noise floor's model, Reach added
    assert arm["settings"]["features"] == [*base["settings"]["features"], FEATURE]
    assert arm["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA
    assert [v["naive_weight"] for v in arm["grid"]] == list(loop.BLEND_WEIGHTS)  # its own blend weight
    assert arm["trained"] is True


def test_a_model_cached_on_the_served_files_is_never_reused_on_the_sibling_demand_files(data, tmp_path):
    served = loop.Loop(replace(data, feature_set="exporter_count"), loop.FIXTURE_LOOP, tmp_path)
    served.noise_floor()
    best = loop.Settings.from_dict(served.results["best"]["settings"])
    on_served = served.run_arm(loop.served_base_arm(best), best)

    lp = loop.Loop(replace(data, feature_set="sibling_demand"), loop.FIXTURE_LOOP, tmp_path)
    results = loop.run_sibling_demand(lp)

    base = results["experiments"][loop.SIBLING_DEMAND_EXPERIMENT]["base_arm"]
    assert base["settings"] == on_served["settings"]  # the same Arm...
    assert base["training_key"] != on_served["training_key"]  # ...keyed by its files


def test_a_dropped_sibling_demand_experiment_leaves_the_served_files_unchanged(data, tmp_path):
    plan = replace(loop.FIXTURE_LOOP, min_win_threshold=2.0)  # nothing can clear it
    served = loop.Loop(replace(data, feature_set="exporter_count"), plan, tmp_path)
    loop.run_market_reach_cap(served)  # an experiment of the round on the served files, dropped
    assert kept_feature_set(served.results) == "exporter_count"

    lp = loop.Loop(replace(data, feature_set="sibling_demand"), plan, tmp_path)
    results = loop.run_sibling_demand(lp)

    assert results["experiments"][loop.SIBLING_DEMAND_EXPERIMENT]["kept"] is None
    assert kept_feature_set(results) == "exporter_count"


def test_a_kept_experiment_on_files_kept_only_if_it_wins_serves_its_files():
    def section(feature_set: str, kept: str | None, **extra) -> dict:
        return {"feature_set": feature_set, "kept": kept, **extra}

    dropped = {"experiments": {
        "exporter count": section("exporter_count", None),  # #127: the gravity fill kept whatever the decision
        "sibling demand": section("sibling_demand", None, files_kept_if_dropped=False),
    }}
    kept = {"experiments": {**dropped["experiments"], "sibling demand": section(
        "sibling_demand", "+ log_sibling_demand_loo", files_kept_if_dropped=False
    )}}

    assert kept_feature_set(dropped) == "exporter_count"
    assert kept_feature_set(kept) == "sibling_demand"
