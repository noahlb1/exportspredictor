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
        baseline=fixture_artefacts.baseline.drop("blocked_opportunity_value_usd", "tariff_part_usd"),
    )
    store.save(older, tmp_path)
    # A version saved before the Tariff part refuses to load, and says how to add it.
    with pytest.raises(store.UnreadableArtefacts, match="exportspredictor.scenarios"):
        store.load_latest(tmp_path)

    folder = scenarios.rebuild_blocked_values(tmp_path, version="20260102T000000Z")
    rebuilt = store.load_latest(tmp_path)

    assert folder.name == rebuilt.version == "20260102T000000Z"
    assert rebuilt.metadata["blocked_values_from"] == "20260101T000000Z"
    assert_frame_equal(rebuilt.baseline, fixture_artefacts.baseline)
    assert_frame_equal(rebuilt.baseline_features, fixture_artefacts.baseline_features)


def test_rebuilt_blocked_values_rewrite_the_model_card_and_re_read_the_ppml_comparison(fixture_artefacts, tmp_path):
    saved = fixture_artefacts.ppml_benchmark
    stale = {**saved, "model": {**saved["model"], "effect_usd": 1.0, "effect_pct": 99.0}}
    older = replace(
        fixture_artefacts,
        metadata={**fixture_artefacts.metadata, "version": "20260101T000000Z"},
        baseline=fixture_artefacts.baseline.drop("blocked_opportunity_value_usd", "tariff_part_usd"),
        ppml_benchmark=stale,
    )
    store.save(older, tmp_path)

    folder = scenarios.rebuild_blocked_values(tmp_path, version="20260102T000000Z")
    rebuilt = store.load(folder)

    # The model's side of the PPML comparison is read again from the rebuilt values; the PPML fit is kept.
    assert rebuilt.ppml_benchmark["model"]["effect_usd"] == pytest.approx(saved["model"]["effect_usd"])
    assert rebuilt.ppml_benchmark["model"]["effect_pct"] == pytest.approx(saved["model"]["effect_pct"])
    assert rebuilt.ppml_benchmark["model"]["model_form"] == saved["model"]["model_form"]
    assert rebuilt.ppml_benchmark["ppml"] == saved["ppml"]
    card = (folder / "model_card.md").read_text(encoding="utf-8")
    assert "version 20260102T000000Z" in card
    assert "## Blocked Opportunities" in card


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

    values = _values(fixture_artefacts, features)
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


def test_a_scenarios_blocked_opportunities_need_room_to_grow_against_its_export_potential(fixture_artefacts):
    result = scenarios.run(
        fixture_artefacts.baseline_features,
        fixture_artefacts.baseline,
        fixture_artefacts.fold_models,
        scenarios.Scenario(add_agreements=("HKG",)),
        form=fixture_artefacts.model_form,
    )

    # Thresholds at 0 (the fixture's Scenario Effects are under $1M), so only room to grow decides. Cars in Hong Kong:
    # the UK already sells more than the Scenario's Export Potential (ADR 0011).
    flagged = scenarios.with_blocked_opportunity_flag(
        result.pairs,
        fixture_artefacts.history,
        fixture_artefacts.baseline_year,
        effect="effect_usd",
        baseline="baseline_potential_usd",
        min_share=0.0,
        min_usd=0.0,
    )
    [cars] = flagged.filter((pl.col("hs6") == "870323") & (pl.col("iso3") == "HKG")).iter_rows(named=True)
    blocked = flagged.filter("blocked_opportunity")

    assert cars["effect_usd"] >= 0 and cars["baseline_actual_exports_usd"] > cars["scenario_potential_usd"]
    assert cars["blocked_opportunity"] is False
    assert blocked.height > 0
    assert (blocked["baseline_actual_exports_usd"] < blocked["scenario_potential_usd"]).all()
    # At the fixed thresholds, none of these tiny effects clears $1M.
    assert result.blocked_opportunities(fixture_artefacts.history, fixture_artefacts.baseline_year).is_empty()


# --- The Tariff part (ADR 0011) -------------------------------------------------------------


def _values(artefacts, features: pl.DataFrame | None = None, baseline: pl.DataFrame | None = None) -> pl.DataFrame:
    return scenarios.blocked_opportunity_values(
        artefacts.baseline_features if features is None else features,
        artefacts.baseline if baseline is None else baseline,
        artefacts.fold_models,
        form=artefacts.model_form,
        tariff_range=scenarios.tariff_range(artefacts.metadata),
    )


def test_the_tariff_part_sets_a_known_tariff_to_0_and_never_switches_the_trade_agreement_on(fixture_artefacts):
    # India charges every Product a tariff above 0 and has no UK Trade Agreement in the fixture Baseline. Its tariffs
    # sit below the tiny fold models' tariff splits, so telephones are given one they price, ln(1 + t) = 0.9, and the
    # Baseline year's Export Potential is predicted again from these rows.
    models, form = fixture_artefacts.fold_models, fixture_artefacts.model_form
    phones = (pl.col("hs6") == "851712") & (pl.col("iso3") == "IND")
    features = fixture_artefacts.baseline_features.with_columns(
        pl.when(phones).then(0.9).otherwise(pl.col("log_applied_tariff")).cast(pl.Float32).alias("log_applied_tariff")
    )
    baseline = features.select("hs6", "iso3", baseline_export_potential_usd=export_potential(features, models, form=form))
    india = features.filter(pl.col("iso3") == "IND")
    tariff_only = india.with_columns(log_applied_tariff=pl.lit(0.0, features.schema["log_applied_tariff"]))
    expected = (
        india.select("hs6", "iso3")
        .with_columns(scenario=export_potential(tariff_only, models, form=form))
        .join(baseline, on=["hs6", "iso3"])
        .select("hs6", part=pl.col("scenario") - pl.col("baseline_export_potential_usd"))
        .sort("hs6")
    )

    values = _values(fixture_artefacts, features, baseline).filter(pl.col("iso3") == "IND").sort("hs6")

    assert (india["trade_agreement"] == 0).all() and (india["log_applied_tariff"] > 0).all()
    assert values["tariff_part_usd"].to_list() == pytest.approx(expected["part"].to_list(), rel=1e-6)
    assert values.filter(pl.col("hs6") == "851712")["tariff_part_usd"].item() > 0
    # The agreement adds to the tariff's part, never takes from it (monotone constraints, ADR 0008).
    assert (values["tariff_part_usd"] <= values["blocked_opportunity_value_usd"] + 1e-6).all()


BARRIERS = [
    (7.5, "No agreement, tariff 7.5%"),
    (150.0, "No agreement, tariff 150%"),
    (12.34, "No agreement, tariff 12.3%"),
    (0.0, "No agreement, tariff 0%"),
    (0.04, "No agreement, tariff under 0.1%"),
    (None, "No agreement, tariff unknown"),
    (float("nan"), "No agreement, tariff unknown"),
]


@pytest.mark.parametrize(("pct", "words"), BARRIERS)
def test_the_barrier_is_worded_from_the_applied_tariff(pct, words):
    assert scenarios.barrier_label(pct) == words


def test_the_barrier_column_words_every_row_as_its_label():
    rates = pl.DataFrame({"applied_tariff_pct": [pct for pct, _ in BARRIERS]}, schema={"applied_tariff_pct": pl.Float64})

    assert rates.select(scenarios.barrier(pl.col("applied_tariff_pct"))).to_series().to_list() == [w for _, w in BARRIERS]


def test_the_value_and_its_tariff_part_clip_to_the_same_training_range(fixture_artefacts):
    # A training range whose floor is above 0 (telephones in India priced at ln(1 + t) = 0.9, the floor 0.9): both runs
    # clip the removed tariff to it, so the value is the every-Market agreement Scenario run with that range (ADR 0006).
    models, form = fixture_artefacts.fold_models, fixture_artefacts.model_form
    phones = (pl.col("hs6") == "851712") & (pl.col("iso3") == "IND")
    features = fixture_artefacts.baseline_features.with_columns(
        pl.when(phones).then(0.9).otherwise(pl.col("log_applied_tariff")).cast(pl.Float32).alias("log_applied_tariff")
    )
    baseline = features.select("hs6", "iso3", baseline_export_potential_usd=export_potential(features, models, form=form))
    floor_range = (0.9, scenarios.tariff_range(fixture_artefacts.metadata)[1])
    every_market = scenarios.Scenario(add_agreements=tuple(features["iso3"].unique().sort()))
    expected = scenarios.run(features, baseline, models, every_market, floor_range, form=form).pairs

    values = scenarios.blocked_opportunity_values(features, baseline, models, tariff_range=floor_range, form=form)
    compared = values.join(expected, on=["hs6", "iso3"]).filter(pl.col("iso3") == "IND").sort("hs6")

    assert "851712" in compared["hs6"].to_list()
    assert compared["blocked_opportunity_value_usd"].to_list() == pytest.approx(compared["effect_usd"].to_list(), rel=1e-6)
    # At the floor, India's telephone tariff cannot fall, so its Tariff part is 0.
    assert values.filter(phones)["tariff_part_usd"].item() == 0


def test_the_tariff_part_is_0_where_the_tariff_is_0_unknown_or_an_agreement_is_in_force(fixture_artefacts):
    unknown = ("220830", "851712")
    features = _unknown_tariffs(fixture_artefacts.baseline_features, unknown)
    values = _values(fixture_artefacts, features).join(
        features.select("hs6", "iso3", "trade_agreement", "log_applied_tariff"), on=["hs6", "iso3"]
    )

    unknown_rows = values.filter(pl.col("log_applied_tariff").is_null())
    zero_tariff = values.filter((pl.col("iso3") == "HKG") & pl.col("log_applied_tariff").is_not_null())
    agreement = values.filter(pl.col("trade_agreement") == 1)
    assert unknown_rows.height == 4 and (unknown_rows["blocked_opportunity_value_usd"] > 0).all()
    assert (unknown_rows["tariff_part_usd"] == 0).all()
    assert zero_tariff.height > 0 and (zero_tariff["tariff_part_usd"] == 0).all()
    assert agreement.height > 0
    assert (agreement["tariff_part_usd"] == 0).all() and (agreement["blocked_opportunity_value_usd"] == 0).all()


def test_the_tariff_run_changes_only_the_known_tariffs_of_pairs_without_an_agreement(fixture_artefacts):
    unknown = ("220830", "851712")
    before = _unknown_tariffs(fixture_artefacts.baseline_features, unknown)
    after = scenarios.tariff_only(before, scenarios.tariff_range(fixture_artefacts.metadata)).sort("hs6", "iso3")
    original = before.join(after.select("hs6", "iso3"), on=["hs6", "iso3"]).sort("hs6", "iso3")

    # India's three pairs with a known tariff: Hong Kong's are 0% or unknown, Japan and Canada have agreements.
    assert after["iso3"].unique().to_list() == ["IND"] and after.height == 3
    assert (after["trade_agreement"] == 0).all() and (after["log_applied_tariff"] == 0).all()
    assert_frame_equal(after.drop("log_applied_tariff"), original.drop("log_applied_tariff"))


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


def _kinds(rates):
    frame = pl.DataFrame({"applied_tariff_pct": rates}, schema={"applied_tariff_pct": pl.Float64})
    return frame.select(scenarios.tariff_kind(pl.col("applied_tariff_pct"))).to_series().to_list()


def test_a_tariff_is_of_one_kind_above_0_0_percent_or_unknown():
    assert _kinds([7.5, 0.04, 0.0, None, float("nan")]) == [
        "Tariff above 0", "Tariff above 0", "Tariff 0%", "Tariff unknown", "Tariff unknown"
    ]
    assert scenarios.TARIFF_KINDS == ("Tariff above 0", "Tariff 0%", "Tariff unknown")


def _split(value, tariff_part, pct):
    rows = pl.DataFrame(
        {"blocked_opportunity_value_usd": [value], "tariff_part_usd": [tariff_part], "applied_tariff_pct": [pct]},
        schema={"blocked_opportunity_value_usd": pl.Float64, "tariff_part_usd": pl.Float64,
                "applied_tariff_pct": pl.Float64},
    )
    [sentence] = scenarios.value_split(rows).to_list()
    return sentence


def test_the_value_split_names_the_tariff_part_its_share_and_the_agreement_part():
    sentence = _split(18.5e6, 7.4e6, 150.0)

    assert sentence == (
        "Of the $18.5m rise, the model associates $7.4m (40%, the Tariff part) with the Applied Tariff of 150% going "
        "to 0, and $11.1m (the Agreement part) with having an average Trade Agreement (standards, customs, legal "
        "frameworks and so on)."
    )


def test_a_0_percent_or_unknown_tariff_rests_the_rise_entirely_on_an_average_agreement():
    zero, unknown = _split(5.1e6, 0.0, 0.0), _split(5.1e6, 0.0, None)

    assert "already 0%" in zero and "rests entirely" in zero and "not on any tariff" in zero
    assert "not known" in unknown and "no Tariff part" in unknown and "rests entirely" in unknown
    for sentence in (zero, unknown):
        assert "going to 0" not in sentence and "average Trade Agreement" in sentence


def test_a_tariff_with_no_tariff_part_says_the_model_associates_none_of_the_rise_with_it():
    sentence = _split(7.1e6, 0.0, 12.0)

    assert "none of the $7.1m rise" in sentence and "Applied Tariff of 12%" in sentence
    assert "rests entirely" in sentence


def test_a_tariff_under_0_1_percent_is_named_as_a_rate_not_as_a_kind_of_tariff():
    sentence = _split(4e6, 1e6, 0.04)

    assert "the Applied Tariff of under 0.1% going to 0" in sentence
    assert "the under 0.1% tariff" not in sentence


def test_a_tariff_part_above_the_value_reads_as_a_negative_agreement_part_never_a_share_above_100_percent():
    # The Agreement part is the remainder, so it is negative where the Tariff part exceeds the value (ADR 0011).
    sentence = _split(4e6, 6e6, 9.0)

    assert "$6.0m (the Tariff part)" in sentence and "more than the $4.0m rise" in sentence
    assert "$2.0m less" in sentence and "negative Agreement part" in sentence
    assert "150%" not in sentence and "-$" not in sentence
    assert "%," not in sentence  # no share is printed
