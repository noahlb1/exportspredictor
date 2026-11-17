"""Scenarios at the package seam: leakage guards and monotonicity, on fixture artefacts.

The fixture Baseline (2024) has UK Trade Agreements with Japan and Canada, none with
India; Japan and Canada are CPTPP members, and the UK is in no Trading Bloc.
"""

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from exportspredictor import scenarios
from exportspredictor.features import FEATURES, POLICY_FEATURES

NON_POLICY = [f for f in FEATURES if f not in POLICY_FEATURES]


@pytest.mark.parametrize(
    "scenario",
    [
        scenarios.Scenario(add_agreements=("IND",)),
        scenarios.Scenario(remove_agreements=("JPN",)),
        scenarios.Scenario(preset="cptpp"),
    ],
)
def test_a_scenario_changes_only_policy_columns(fixture_artefacts, scenario):
    before = fixture_artefacts.baseline_features
    after = scenarios.apply(before, scenario)

    assert after.height > 0
    assert after.columns == before.columns
    original = before.join(after.select("hs6", "iso3"), on=["hs6", "iso3"]).sort("hs6", "iso3")
    after = after.sort("hs6", "iso3")
    assert_frame_equal(after.select("hs6", "iso3", "fold", *NON_POLICY), original.select("hs6", "iso3", "fold", *NON_POLICY))
    assert (after.select(POLICY_FEATURES) != original.select(POLICY_FEATURES)).to_numpy().any(axis=1).all()


def test_the_package_check_rejects_a_change_outside_the_policy_columns(fixture_artefacts):
    before = fixture_artefacts.baseline_features
    tampered = before.with_columns(log_distance=before["log_distance"] + 1)

    with pytest.raises(scenarios.LeakageError):
        scenarios.check_only_policy_changed(before, tampered)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    row=st.integers(min_value=0, max_value=14),
    other_policy=st.lists(st.integers(min_value=0, max_value=1), min_size=7, max_size=7),
)
def test_switching_a_trade_agreement_on_never_lowers_a_prediction(fixture_artefacts, row, other_policy):
    pair = fixture_artefacts.baseline_features[row].with_columns(
        pl.lit(v, pl.Int8).alias(c) for c, v in zip(POLICY_FEATURES[1:], other_policy, strict=True)
    )
    without = pair.with_columns(trade_agreement=pl.lit(0, pl.Int8))
    with_agreement = pair.with_columns(trade_agreement=pl.lit(1, pl.Int8))

    off = scenarios.predict(without, fixture_artefacts.fold_models)
    on = scenarios.predict(with_agreement, fixture_artefacts.fold_models)

    assert on[0] >= off[0]


def test_predict_reproduces_the_out_of_fold_baseline(fixture_artefacts):
    features = fixture_artefacts.baseline_features
    baseline = fixture_artefacts.baseline.join(features.select("hs6", "iso3"), on=["hs6", "iso3"], how="right")

    predicted = scenarios.predict(features, fixture_artefacts.fold_models)

    assert predicted.to_list() == pytest.approx(baseline["export_potential_usd"].to_list(), rel=1e-6)
