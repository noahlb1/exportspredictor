"""More training rows through XGBoost external memory, tried by the experiment loop."""

from dataclasses import replace

import pytest
import xgboost as xgb

from exportspredictor import loop, market_reach
from exportspredictor.training import RowBudgetExceeded

# Keys computed before the `external_memory` field existed (as in test_loop_trainers): models cached then must
# still be found.
FINGERPRINT = {
    "training_years": [2020, 2021], "validation_year": 2022, "built_zero_rate": 0.05, "zero_seed": 0,
    "fold": 0, "n_folds": 5, "device": "cuda",
}
ONE_MODEL = loop.Settings(params={"max_depth": 8, "eta": 0.05}, non_uk_rate=0.3, zero_rate=0.05)


@pytest.fixture(scope="module")
def data(fixture_world):
    return loop.fixture_data(fixture_world)


def test_external_memory_off_is_the_default_and_keeps_every_existing_training_key():
    assert "external_memory" not in ONE_MODEL.to_dict()
    named = loop.Arm("in-core", {"external_memory": False}).apply(ONE_MODEL)
    assert loop.training_key(named, FINGERPRINT) == "51d1f778f2df1f91"
    switched = loop.Arm("external memory", {"external_memory": True}).apply(ONE_MODEL)
    assert loop.training_key(switched, FINGERPRINT) != "51d1f778f2df1f91"


def test_an_external_memory_arm_trains_on_the_same_rows_through_an_external_memory_matrix(data, monkeypatch):
    built = []

    class Spy(xgb.ExtMemQuantileDMatrix):
        def __init__(self, *args, **kwargs):
            built.append(args[0])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(xgb, "ExtMemQuantileDMatrix", Spy)
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    in_core = lp.run_arm(loop.Arm("in-core"), loop.FIXTURE_LOOP.start)
    assert not built
    external = lp.run_arm(loop.Arm("external memory", {"external_memory": True}), loop.FIXTURE_LOOP.start)

    assert len(built) == 1
    assert external["training_rows"] == in_core["training_rows"] == 60
    assert external["settings"]["external_memory"] is True
    assert external["training_key"] != in_core["training_key"]
    assert external[loop.SELECTION_MEASURE] == pytest.approx(in_core[loop.SELECTION_MEASURE], abs=0.02)


@pytest.mark.parametrize("family", [loop.LIGHTGBM, loop.CATBOOST])
def test_another_family_with_external_memory_is_refused_before_training_anything(data, family, tmp_path):
    lp = loop.Loop(data, loop.FIXTURE_LOOP, tmp_path)
    lp.noise_floor()
    before = {p.name for p in (tmp_path / loop.MODELS_DIR).iterdir()}

    with pytest.raises(ValueError, match="only XGBoost"):
        lp.run_experiment("external memory", [loop.Arm(family, {"family": family, "external_memory": True})])

    assert {p.name for p in (tmp_path / loop.MODELS_DIR).iterdir()} == before


def test_an_external_memory_arm_is_exempt_from_the_row_budget_and_an_in_core_one_is_not(data, tmp_path):
    lp = loop.Loop(data, replace(loop.FIXTURE_LOOP, row_budget=150), tmp_path)
    changes = {"non_uk_rate": 1.0, "params": {"max_depth": 3}}

    with pytest.raises(RowBudgetExceeded):
        lp.run_experiment("in-core", [loop.Arm("every row", changes)])
    results = lp.run_experiment("external memory", [loop.Arm("every row", {**changes, "external_memory": True})])

    (row,) = results["experiments"]["external memory"]["arms"]
    assert row["trained"] is True
    assert row["settings"]["non_uk_rate"] == 1.0


def test_the_external_memory_arm_runs_every_non_uk_row_against_the_best_so_far_as_served(data, tmp_path):
    lp = loop.Loop(replace(data, feature_set="exporter_count"), loop.FIXTURE_LOOP, tmp_path)

    results = loop.run_external_memory(lp)

    section = results["experiments"][loop.EXTERNAL_MEMORY_EXPERIMENT]
    base, (arm,) = section["base_arm"], section["arms"]
    assert "external_memory" not in base["settings"]
    assert base["settings"]["market_reach_beta"] == arm["settings"]["market_reach_beta"] == market_reach.CHOSEN_BETA
    assert arm["arm"] == loop.EXTERNAL_MEMORY
    assert arm["settings"]["external_memory"] is True
    assert arm["settings"]["non_uk_rate"] == 1.0
    assert [v["naive_weight"] for v in arm["grid"]] == list(loop.BLEND_WEIGHTS)
    assert section["feature_set"] == "exporter_count"
    assert "decided" in section
