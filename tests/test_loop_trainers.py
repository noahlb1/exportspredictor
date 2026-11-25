"""The loop's trainer seam: the model family is a training field, XGBoost by default."""

import numpy as np
import pytest
import xgboost as xgb

from exportspredictor import loop

# Keys computed before the `family` field existed: models cached then must still be found.
FINGERPRINT = {
    "training_years": [2020, 2021], "validation_year": 2022, "built_zero_rate": 0.05, "zero_seed": 0,
    "fold": 0, "n_folds": 5, "device": "cuda",
}
ONE_MODEL = loop.Settings(params={"max_depth": 8, "eta": 0.05}, non_uk_rate=0.3, zero_rate=0.05)
TWO_PART_UNCONSTRAINED = loop.Settings(
    params={"max_depth": 8, "eta": 0.05}, non_uk_rate=0.3, zero_rate=0.05, two_part=True, monotone=False,
    market_reach_beta=0.5,
)


@pytest.fixture(scope="module")
def data(fixture_world):
    return loop.fixture_data(fixture_world)


@pytest.mark.parametrize(
    ("settings", "key"), [(ONE_MODEL, "51d1f778f2df1f91"), (TWO_PART_UNCONSTRAINED, "842e49a9ce43d317")]
)
def test_xgboost_whether_named_or_left_at_the_default_keeps_every_existing_training_key(settings, key):
    assert loop.training_key(settings, FINGERPRINT) == key
    named = loop.Arm("xgboost", {"family": loop.XGBOOST}).apply(settings)
    assert loop.training_key(named, FINGERPRINT) == key


def test_settings_saved_before_the_family_field_existed_read_back_unchanged():
    saved = ONE_MODEL.to_dict()

    assert "family" not in saved
    assert loop.Settings.from_dict(saved).family == loop.XGBOOST


def test_an_arm_naming_xgboost_reuses_the_best_so_fars_cached_model(data):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    results = lp.run_experiment("family", [loop.Arm("xgboost", {"family": loop.XGBOOST})])

    row = results["experiments"]["family"]["arms"][0]
    assert row["trained"] is False
    assert row["training_key"] == results["noise_floor"]["arms"][0]["training_key"]


def test_an_unknown_family_is_refused_before_training_anything(data, monkeypatch):
    lp = loop.Loop(data, loop.FIXTURE_LOOP)
    lp.noise_floor()

    def refuse(*args, **kwargs):
        raise AssertionError("a model was trained")

    monkeypatch.setattr(xgb, "train", refuse)
    arms = [loop.Arm("another seed", {"seed": 7}), loop.Arm("unknown", {"family": "unknown"})]

    with pytest.raises(ValueError, match="no trainer for family 'unknown'"):
        lp.run_experiment("family", arms)


def _xgboost_model(x, y):
    return xgb.train({"objective": "reg:squarederror", "device": "cpu"}, xgb.DMatrix(x, label=y), num_boost_round=3)


def _catboost_model(x, y):
    import catboost as cb

    return cb.CatBoost({"iterations": 3, "verbose": False, "thread_count": 1, "allow_writing_files": False}).fit(x, y)


@pytest.mark.parametrize(("family", "fit"), [(loop.XGBOOST, _xgboost_model), (loop.CATBOOST, _catboost_model)])
def test_a_saved_model_loads_back_predicting_the_same(family, fit, tmp_path):
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = x[:, 0] * 2.0
    trainer = loop.trainer_for(family)
    model = fit(x, y)
    path = tmp_path / f"model{trainer.suffix}"

    trainer.save(model, path)
    loaded = trainer.load(path)

    matrix = xgb.DMatrix(x) if family == loop.XGBOOST else x
    assert np.allclose(loaded.predict(matrix), model.predict(matrix))
