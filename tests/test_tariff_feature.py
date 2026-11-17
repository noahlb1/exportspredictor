"""The Applied Tariff as a model input (ADR 0006, ADR 0002): ln(1 + t), a policy feature
constrained to never raise Export Potential, clipped to the training-year range, with the
carried-forward mark kept as row metadata and never fed to the model.
"""

import math

import polars as pl
import pytest
import xgboost as xgb

from exportspredictor.features import (
    FEATURES,
    FeatureTransform,
    build_features,
    training_rows,
)
from exportspredictor.fixture import UK, build_world
from exportspredictor.training import model_params

WHISKY = "220830"


@pytest.fixture(scope="module")
def world():
    return build_world()


def _rows(keys: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        [(*key, 1.0) for key in keys],
        schema={"exporter": pl.String, "importer": pl.String, "hs6": pl.String, "year": pl.Int16, "value_usd": pl.Float64},
        orient="row",
    )


def test_the_tariff_feature_is_the_log_of_one_plus_the_applied_tariff(world):
    rows = _rows([
        (UK, "IND", WHISKY, 2021),  # India's 150% MFN rate on whisky
        (UK, "JPN", WHISKY, 2021),  # Japan's 2% MFN rate, before the UK–Japan agreement
        (UK, "CAN", WHISKY, 2021),  # Canada's 0% preference for the UK
    ])

    built = build_features(rows, world.flows, world.sources)

    assert built["log_applied_tariff"].to_list() == pytest.approx([math.log(2.5), math.log(1.02), 0.0])


def test_carried_forward_tariffs_are_marked_as_row_metadata_never_as_a_model_input(world):
    rows = _rows([
        (UK, "IND", WHISKY, 2021),  # observed
        (UK, "IND", WHISKY, 2023),  # carried forward from 2021
        (UK, "JPN", WHISKY, 2023),  # the UK–Japan agreement entered force in 2022: set to 0
    ])

    built = build_features(rows, world.flows, world.sources)

    assert built["tariff_imputed"].to_list() == [False, True, True]
    assert built["log_applied_tariff"].to_list() == pytest.approx([math.log(2.5), math.log(2.5), 0.0])
    # Japan's 2021 MFN rate is kept for Scenarios that remove the agreement.
    assert built["mfn_tariff"][2] == pytest.approx(2.0)
    assert "tariff_imputed" not in FEATURES
    assert "mfn_tariff" not in FEATURES


TRAINING_YEARS = [2021, 2022]


def _all_rows(world, flows=None) -> pl.DataFrame:
    flows = world.flows if flows is None else flows
    return build_features(training_rows(flows, rate=1.0), flows, world.sources)


def test_the_tariff_clip_range_comes_from_training_years_only(world):
    table = _all_rows(world)
    hiked = table.with_columns(
        log_applied_tariff=pl.when(pl.col("year") >= 2023).then(pl.lit(math.log(5.0))).otherwise("log_applied_tariff")
    )

    fitted = FeatureTransform.fit(table, TRAINING_YEARS)

    assert FeatureTransform.fit(hiked, TRAINING_YEARS) == fitted
    assert fitted.bounds["log_applied_tariff"] == pytest.approx((0.0, math.log(2.5)))


def test_a_tariff_beyond_the_training_range_is_clipped_at_serving(world):
    table = _all_rows(world)
    fitted = FeatureTransform.fit(table, TRAINING_YEARS)
    serving = table.filter(pl.col("year") == 2024).with_columns(log_applied_tariff=pl.lit(math.log(5.0), pl.Float32))

    applied = fitted.apply(serving)

    assert applied["log_applied_tariff"].max() == pytest.approx(math.log(2.5))


def test_a_model_trained_where_tariffs_go_with_more_trade_still_never_rises_with_the_tariff(world):
    """The constraint holds against the data: the Flows are rigged so that higher tariffs come
    with higher Flows, and the trained model still never predicts more at a higher tariff."""
    rows = _all_rows(world).filter(pl.col("log_applied_tariff").is_not_null())
    rigged = rows.with_columns(value_usd=pl.col("value_usd") * (5 * pl.col("log_applied_tariff")).exp())
    matrix = xgb.DMatrix(rigged.select(FEATURES).to_numpy(), label=rigged["value_usd"].to_numpy(), feature_names=FEATURES)
    model = xgb.train(model_params({"max_depth": 3, "eta": 0.3}), matrix, num_boost_round=30)

    low = rows.with_columns(log_applied_tariff=pl.lit(0.0, pl.Float32)).select(FEATURES).to_numpy()
    high = rows.with_columns(log_applied_tariff=pl.lit(math.log(2.5), pl.Float32)).select(FEATURES).to_numpy()

    assert (model.inplace_predict(high) <= model.inplace_predict(low) + 1e-6).all()
