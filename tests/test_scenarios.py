"""Scenarios at the package seam: leakage guards and monotonicity, on fixture artefacts.

The fixture Baseline (2024) has UK Trade Agreements with Japan and Canada, none with
India; Japan and Canada are CPTPP members, and the UK is in no Trading Bloc.
"""

from dataclasses import replace

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from exportspredictor import artefacts as store
from exportspredictor import policy, scenarios
from exportspredictor.features import FEATURES, POLICY_FEATURES
from exportspredictor.prediction import export_potential

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
        pl.lit(v, pl.Int8).alias(c) for c, v in zip(policy.POLICY_FEATURES[1:], other_policy, strict=True)
    )
    without = pair.with_columns(trade_agreement=pl.lit(0, pl.Int8))
    with_agreement = pair.with_columns(trade_agreement=pl.lit(1, pl.Int8))

    form = fixture_artefacts.model_form
    off = export_potential(without, fixture_artefacts.fold_models, form=form)
    on = export_potential(with_agreement, fixture_artefacts.fold_models, form=form)

    assert on[0] >= off[0]


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    row=st.integers(min_value=0, max_value=19),
    tariffs=st.lists(st.floats(min_value=0.0, max_value=2.0), min_size=2, max_size=2),
)
def test_raising_the_applied_tariff_never_raises_a_prediction(fixture_artefacts, row, tariffs):
    low, high = sorted(tariffs)
    pair = fixture_artefacts.baseline_features[row]
    models = fixture_artefacts.fold_models

    form = fixture_artefacts.model_form
    lower = export_potential(pair.with_columns(log_applied_tariff=pl.lit(low, pl.Float32)), models, form=form)
    higher = export_potential(pair.with_columns(log_applied_tariff=pl.lit(high, pl.Float32)), models, form=form)

    assert higher[0] <= lower[0]


def test_scenario_effects_start_from_the_baseline_year_not_the_gap_period_mean(fixture_artefacts):
    result = scenarios.run(
        fixture_artefacts.baseline_features,
        fixture_artefacts.baseline,
        fixture_artefacts.fold_models,
        scenarios.Scenario(add_agreements=("IND",)),
        form=fixture_artefacts.model_form,
    )
    joined = result.pairs.join(fixture_artefacts.baseline, on=["hs6", "iso3"])

    assert joined.height == 5
    assert joined["baseline_potential_usd"].to_list() == pytest.approx(
        joined["baseline_export_potential_usd"].to_list()
    )
    assert not joined["baseline_potential_usd"].equals(joined["export_potential_usd"])
    blocked = joined["effect_usd"].to_list()
    assert joined["blocked_opportunity_value_usd"].to_list() == pytest.approx(blocked)


def test_blocked_values_can_be_rebuilt_from_saved_fold_models(fixture_artefacts, tmp_path):
    older = replace(
        fixture_artefacts,
        metadata={**fixture_artefacts.metadata, "version": "20260101T000000Z"},
        baseline=fixture_artefacts.baseline.drop("blocked_opportunity_value_usd"),
    )
    store.save(older, tmp_path)

    folder = scenarios.rebuild_blocked_values(tmp_path, version="20260102T000000Z")
    rebuilt = store.load_latest(tmp_path)

    assert folder.name == rebuilt.version == "20260102T000000Z"
    assert rebuilt.metadata["blocked_values_from"] == "20260101T000000Z"
    assert_frame_equal(rebuilt.baseline, fixture_artefacts.baseline)
    assert_frame_equal(rebuilt.baseline_features, fixture_artefacts.baseline_features)


def test_blocked_values_are_zero_where_a_trade_agreement_exists(fixture_artefacts):
    values = fixture_artefacts.baseline.join(
        fixture_artefacts.baseline_features.select("hs6", "iso3", "trade_agreement"), on=["hs6", "iso3"]
    )

    assert (values.filter(pl.col("trade_agreement") == 1)["blocked_opportunity_value_usd"] == 0).all()
    assert (values.filter(pl.col("trade_agreement") == 0)["blocked_opportunity_value_usd"] >= 0).all()


# --- Tariffs in Scenarios (ADR 0006) ------------------------------------------------------


def test_removing_a_trade_agreement_restores_the_importers_2021_mfn_tariff(fixture_artefacts):
    before = fixture_artefacts.baseline_features
    after = scenarios.apply(before, scenarios.Scenario(remove_agreements=("JPN",))).sort("hs6")
    mfn = before.filter(pl.col("iso3") == "JPN").sort("hs6")["mfn_tariff"]

    assert (mfn > 0).any()
    assert after["log_applied_tariff"].to_list() == pytest.approx((mfn / 100).log1p().to_list(), rel=1e-6)


def test_adding_a_trade_agreement_removes_the_tariff(fixture_artefacts):
    before = fixture_artefacts.baseline_features
    after = scenarios.apply(before, scenarios.Scenario(add_agreements=("IND",)))

    assert before.filter(pl.col("iso3") == "IND")["log_applied_tariff"].min() > 0
    assert after.height == 5 and (after["log_applied_tariff"] == 0).all()


def _unknown_tariffs(features: pl.DataFrame, hs6: tuple[str, ...]) -> pl.DataFrame:
    """The Baseline feature rows with the Applied Tariff unknown (null) for these Products in India and HKG,
    which are made EU and CPTPP members so the bloc presets add an agreement there too."""
    unknown = pl.col("hs6").is_in(list(hs6)) & pl.col("iso3").is_in(["IND", "HKG"])
    members = pl.col("iso3").is_in(["IND", "HKG"])
    return features.with_columns(
        pl.when(unknown).then(None).otherwise(pl.col("log_applied_tariff")).alias("log_applied_tariff"),
        *(pl.when(members).then(1).otherwise(pl.col(c)).cast(features.schema[c]).alias(c) for c in ("eu_d", "cptpp_d")),
    )


@pytest.mark.parametrize(
    "scenario",
    [
        scenarios.Scenario(add_agreements=("IND", "HKG")),
        scenarios.Scenario(preset="eu_rejoin"),
        scenarios.Scenario(preset="eu_customs_union"),
        scenarios.Scenario(preset="cptpp"),
    ],
)
def test_adding_a_trade_agreement_leaves_an_unknown_tariff_unknown(fixture_artefacts, scenario):
    # XGBoost's monotone constraints do not cover the missing-value branch, so zeroing an
    # unknown tariff could lower a prediction: a missing tariff stays missing (ADR 0006).
    unknown = ("220830", "851712")
    before = _unknown_tariffs(fixture_artefacts.baseline_features, unknown)
    after = scenarios.apply(before, scenario).filter(pl.col("iso3").is_in(["IND", "HKG"]))
    known = after.filter(~pl.col("hs6").is_in(list(unknown)))

    assert after.filter(pl.col("hs6").is_in(list(unknown)))["log_applied_tariff"].null_count() == 4
    assert known.height > 0 and (known["log_applied_tariff"] == 0).all()


def test_blocked_values_leave_an_unknown_tariff_unknown(fixture_artefacts):
    unknown = ("220830", "851712")
    features = _unknown_tariffs(fixture_artefacts.baseline_features, unknown)
    rows = features.filter(pl.col("iso3") == "IND").with_columns(
        trade_agreement=pl.lit(1, features.schema["trade_agreement"]),
        log_applied_tariff=pl.when(pl.col("hs6").is_in(list(unknown)))
        .then(None)
        .otherwise(0.0)
        .cast(features.schema["log_applied_tariff"]),
    )
    expected = (
        rows.select("hs6", "iso3")
        .with_columns(scenario=export_potential(rows, fixture_artefacts.fold_models, form=fixture_artefacts.model_form))
        .join(fixture_artefacts.baseline, on=["hs6", "iso3"])
        .select("hs6", effect=pl.col("scenario") - pl.col("baseline_export_potential_usd"))
        .sort("hs6")
    )

    values = scenarios.blocked_opportunity_values(
        features, fixture_artefacts.baseline, fixture_artefacts.fold_models, form=fixture_artefacts.model_form
    )
    india = values.filter(pl.col("iso3") == "IND").sort("hs6")

    assert india["hs6"].to_list() == expected["hs6"].to_list()
    assert india["blocked_opportunity_value_usd"].to_list() == pytest.approx(expected["effect"].to_list(), rel=1e-6)


def test_blocked_values_remove_the_tariff_with_the_agreement(fixture_artefacts):
    result = scenarios.run(
        fixture_artefacts.baseline_features,
        fixture_artefacts.baseline,
        fixture_artefacts.fold_models,
        scenarios.Scenario(add_agreements=("IND",)),
        form=fixture_artefacts.model_form,
    )
    blocked = fixture_artefacts.baseline.join(result.pairs, on=["hs6", "iso3"])

    assert blocked["blocked_opportunity_value_usd"].to_list() == pytest.approx(blocked["effect_usd"].to_list())


def test_a_tariff_scenario_sets_the_tariff_in_the_chosen_markets_and_sectors(fixture_artefacts):
    before = fixture_artefacts.baseline_features
    change = scenarios.TariffChange(markets=("JPN", "IND"), rate_pct=10.0, sectors=("22", "87"))
    after = scenarios.apply(before, scenarios.Scenario(tariff=change))

    assert sorted(after.select("hs6", "iso3").rows()) == [
        ("220830", "IND"), ("220830", "JPN"), ("870323", "IND"), ("870323", "JPN")
    ]
    assert after["log_applied_tariff"].to_list() == pytest.approx([0.0953102] * 4, rel=1e-5)


def test_a_tariff_beyond_the_training_range_is_clipped_and_counted(fixture_artefacts):
    low, high = scenarios.tariff_range(fixture_artefacts.metadata)
    result = scenarios.run(
        fixture_artefacts.baseline_features,
        fixture_artefacts.baseline,
        fixture_artefacts.fold_models,
        scenarios.Scenario(tariff=scenarios.TariffChange(markets=("JPN",), rate_pct=400.0)),
        tariff_range=(low, high),
        form=fixture_artefacts.model_form,
    )
    after = scenarios.apply(
        fixture_artefacts.baseline_features,
        scenarios.Scenario(tariff=scenarios.TariffChange(markets=("JPN",), rate_pct=400.0)),
        tariff_range=(low, high),
    )

    assert (low, high) == pytest.approx((0.0, 0.9162907))  # India's 150% whisky
    assert result.clipped_pairs == 5
    assert after["log_applied_tariff"].to_list() == pytest.approx([high] * 5)


@pytest.mark.parametrize(
    "scenario",
    [
        scenarios.Scenario(tariff=scenarios.TariffChange(markets=("JPN",), rate_pct=-1.0)),
        scenarios.Scenario(tariff=scenarios.TariffChange(markets=("JPN",), rate_pct=5.0, sectors=("99",))),
        scenarios.Scenario(tariff=scenarios.TariffChange(markets=("XXX",), rate_pct=5.0)),
        scenarios.Scenario(tariff=scenarios.TariffChange(markets=(), rate_pct=5.0)),
        scenarios.Scenario(add_agreements=("IND",), tariff=scenarios.TariffChange(markets=("JPN",), rate_pct=5.0)),
    ],
)
def test_a_malformed_tariff_scenario_is_rejected(fixture_artefacts, scenario):
    with pytest.raises(scenarios.InvalidScenario):
        scenarios.apply(fixture_artefacts.baseline_features, scenario)


@pytest.mark.parametrize(
    "scenario",
    [
        *(scenarios.Scenario(preset=name) for name in scenarios.PRESETS),
        scenarios.Scenario(add_agreements=("IND", "HKG"), remove_agreements=("JPN",)),
        scenarios.Scenario(tariff=scenarios.TariffChange(markets=("JPN", "IND", "CAN"), rate_pct=500.0)),
    ],
)
def test_every_scenario_changes_only_policy_columns(fixture_artefacts, scenario):
    before = fixture_artefacts.baseline_features
    after = scenarios.apply(before, scenario, scenarios.tariff_range(fixture_artefacts.metadata))
    fixed = ["hs6", "iso3", "fold", *NON_POLICY, "mfn_tariff", "tariff_imputed"]
    fixed = [c for c in fixed if c in before.columns]
    original = before.join(after.select("hs6", "iso3"), on=["hs6", "iso3"]).sort("hs6", "iso3")

    assert after.columns == before.columns
    assert_frame_equal(after.sort("hs6", "iso3").select(fixed), original.select(fixed))


def test_a_scenario_re_predicts_in_the_served_form_and_leaves_the_naive_baseline_unchanged(fixture_artefacts):
    features = fixture_artefacts.baseline_features
    scenario = scenarios.Scenario(add_agreements=("IND",))
    changed = scenarios.apply(features, scenario)

    result = scenarios.run(
        features, fixture_artefacts.baseline, fixture_artefacts.fold_models, scenario, form=fixture_artefacts.model_form
    )

    before = features.join(changed.select("hs6", "iso3"), on=["hs6", "iso3"]).sort("hs6", "iso3")
    assert changed.sort("hs6", "iso3")["naive_usd"].to_list() == before["naive_usd"].to_list()
    expected = changed.select("hs6", "iso3").with_columns(
        expected=export_potential(changed, fixture_artefacts.fold_models, form=fixture_artefacts.model_form)
    )
    joined = result.pairs.join(expected, on=["hs6", "iso3"])
    assert joined.height == changed.height > 0
    assert joined["scenario_potential_usd"].to_list() == pytest.approx(joined["expected"].to_list())
    assert (joined["effect_usd"] >= -1e-3).all()


def test_the_summary_writes_counts_with_thousands_separators():
    pairs = pl.DataFrame(
        {
            "iso3": ["DEU"],
            "baseline_potential_usd": [1e9],
            "scenario_potential_usd": [1.5e9],
            "effect_usd": [5e8],
        }
    )
    result = scenarios.ScenarioResult(scenario=scenarios.Scenario(add_agreements=("DEU",)), pairs=pairs)

    assert "with 5,526 Blocked Opportunities." in result.summary(blocked=5526, year=2024)
