"""Training through the package: fixture world in, a Baseline table out-of-fold for every UK pair."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from exportspredictor.features import FEATURES, POLICY_FEATURES, FeatureTransform, build_features, uk_grid
from exportspredictor.fixture import UK
from exportspredictor.folds import assign_folds
from exportspredictor.pipeline import build_artefacts, fixture_training, refresh

CRUDE_OIL = "270900"  # the fixture world's Excluded Product


@pytest.fixture
def world(fixture_world):
    return fixture_world


@pytest.fixture
def artefacts(fixture_artefacts):
    return fixture_artefacts


def test_baseline_has_one_row_per_uk_product_market_pair(artefacts):
    baseline = artefacts.baseline

    assert baseline.height == 5 * 3
    assert baseline.select("hs6", "iso3").is_duplicated().sum() == 0


def test_baseline_actual_exports_are_the_gap_period_mean_with_missing_years_as_zero(world, artefacts):
    # GBR→JPN 851712 has no Flow in 2024, and GBR→CAN 851712 none in 2023: they count as $0.
    uk = world.flows.filter(pl.col("exporter") == UK, pl.col("year").is_between(2022, 2024))
    totals = uk.group_by("hs6", iso3="importer").agg(total=pl.col("value_usd").sum())
    joined = artefacts.baseline.join(totals, on=["hs6", "iso3"])

    assert joined.height == 15
    assert joined["actual_exports_usd"].to_list() == pytest.approx((joined["total"] / 3).to_list())
    phones_to_japan = joined.filter(pl.col("hs6") == "851712", pl.col("iso3") == "JPN").row(0, named=True)
    flows = uk.filter(pl.col("hs6") == "851712", pl.col("importer") == "JPN", pl.col("value_usd") > 0)
    assert flows.height == 2
    assert phones_to_japan["actual_exports_usd"] == pytest.approx(flows["value_usd"].sum() / 3)


def test_metadata_records_the_gap_period_and_the_baseline_year(artefacts):
    assert artefacts.metadata["gap_years"] == [2022, 2024]
    assert artefacts.metadata["baseline_year"] == 2024
    assert artefacts.gap_years == (2022, 2024)


def test_history_holds_each_gap_years_export_potential_and_gap_for_every_pair(artefacts):
    gap_rows = artefacts.history.filter(pl.col("year") >= 2022)

    assert gap_rows.height == 15 * 3
    assert gap_rows["export_potential_usd"].is_not_null().all()
    assert gap_rows["export_gap_usd"].to_list() == pytest.approx(
        (gap_rows["export_potential_usd"] - gap_rows["actual_exports_usd"]).to_list()
    )
    means = gap_rows.group_by("hs6", "iso3").agg(pl.col("export_potential_usd", "export_gap_usd").mean())
    joined = artefacts.baseline.join(means, on=["hs6", "iso3"], suffix="_mean")
    assert joined["export_gap_usd"].to_list() == pytest.approx(joined["export_gap_usd_mean"].to_list())
    assert joined["export_potential_usd"].to_list() == pytest.approx(joined["export_potential_usd_mean"].to_list())


def test_baseline_export_potential_is_the_baseline_years_score(artefacts):
    latest = artefacts.history.filter(pl.col("year") == 2024).select("hs6", "iso3", "export_potential_usd")
    joined = artefacts.baseline.join(latest, on=["hs6", "iso3"], suffix="_2024")

    assert joined.height == 15
    assert joined["baseline_export_potential_usd"].to_list() == pytest.approx(
        joined["export_potential_usd_2024"].to_list()
    )
    # The mean differs from the 2024 score: the Gap Period is not 2024 alone.
    assert not joined["baseline_export_potential_usd"].equals(joined["export_potential_usd"])


def test_each_gap_years_uk_score_comes_from_the_fold_model_that_held_the_pair_out(world, artefacts):
    transform = FeatureTransform.from_dict(artefacts.metadata["feature_transform"])
    for year in (2022, 2023, 2024):
        year_flows = world.flows.filter(pl.col("year") == year)
        grid = assign_folds(
            transform.apply(build_features(uk_grid(year_flows, UK), year_flows, world.sources)).rename(
                {"importer": "iso3"}
            )
        )
        scored = artefacts.history.filter(pl.col("year") == year).join(grid.drop("year"), on=["hs6", "iso3"])
        assert scored.height == 15
        for (fold,), rows in scored.group_by("fold"):
            model = artefacts.fold_models[fold]
            assert model.attr("held_out_fold") == str(fold)
            # The pair's fold model never trained on any of the pair's years (ADR 0001).
            expected = model.inplace_predict(rows.select(FEATURES).to_numpy().astype(np.float32))
            assert rows["export_potential_usd"].to_list() == pytest.approx(expected.tolist(), rel=1e-5)


def test_export_gap_is_potential_minus_actual(artefacts):
    row = artefacts.baseline.row(0, named=True)

    assert row["export_potential_usd"] > 0
    assert row["export_gap_usd"] == pytest.approx(
        row["export_potential_usd"] - row["actual_exports_usd"]
    )


def test_metadata_records_version_baseline_year_and_folds(artefacts):
    assert artefacts.metadata["version"] == "fixture-test"
    assert artefacts.metadata["baseline_year"] == 2024
    assert len(artefacts.fold_models) == artefacts.metadata["n_folds"] == 5


def test_fold_assignment_is_deterministic_and_keeps_a_pair_in_one_fold(world):
    uk = world.flows.filter(pl.col("exporter") == UK).select("hs6", iso3="importer")

    first, second = assign_folds(uk), assign_folds(uk)

    assert first.equals(second)
    assert first.group_by("hs6", "iso3").agg(pl.col("fold").n_unique())["fold"].max() == 1


def test_metadata_records_the_zero_sampling_rate_and_the_full_feature_set(artefacts):
    assert artefacts.metadata["zero_sampling_rate"] == pytest.approx(0.5)
    assert artefacts.metadata["features"] == FEATURES
    assert set(POLICY_FEATURES) <= set(artefacts.metadata["monotone_constraints"])


def test_baseline_year_data_never_reaches_the_fitted_transform(world, artefacts):
    baseline_year = pl.col("year") == 2024
    distorted = replace(
        world,
        flows=world.flows.with_columns(
            value_usd=pl.when(baseline_year).then(pl.col("value_usd") * 1000).otherwise("value_usd")
        ),
    )

    rebuilt = build_artefacts(distorted, version="distorted")

    assert rebuilt.metadata["feature_transform"] == artefacts.metadata["feature_transform"]
    assert rebuilt.metadata["training_years"] == [2021, 2022, 2023]


def test_hyperparameters_are_tuned_on_the_validation_year_and_recorded(artefacts):
    metadata = artefacts.metadata

    assert metadata["validation_year"] == 2022
    assert metadata["training_years"] == [2021, 2022, 2023]
    assert len(metadata["tuning"]) >= 2
    best = min(metadata["tuning"], key=lambda result: result["valid_metric"])
    assert metadata["params"] == best["config"]
    assert metadata["rounds"] == best["rounds"]


def test_metadata_records_the_device_and_the_search_and_training_time(artefacts):
    metadata = artefacts.metadata

    assert metadata["device"] == "cpu"  # the fixture; the real run records "cuda"
    assert metadata["tuning_seconds"] > 0
    assert metadata["training_seconds"] > 0


def test_baseline_features_hold_each_pairs_fold_and_model_inputs(artefacts):
    inputs = artefacts.baseline_features

    assert inputs.select("hs6", "iso3").sort("hs6", "iso3").equals(artefacts.baseline.select("hs6", "iso3"))
    assert set(FEATURES) <= set(inputs.columns)
    assert inputs.equals(assign_folds(inputs.drop("fold")).select(inputs.columns))


def test_history_holds_uk_actual_exports_up_to_the_baseline_year(world, artefacts):
    history = artefacts.history
    uk = world.flows.filter(pl.col("exporter") == UK, pl.col("value_usd") > 0, pl.col("hs6") != CRUDE_OIL)

    assert history.columns == ["hs6", "iso3", "year", "actual_exports_usd", "export_potential_usd", "export_gap_usd"]
    assert sorted(history["year"].unique().to_list()) == [2021, 2022, 2023, 2024]
    # Before the Gap Period only years with a Flow are kept, and they carry no score.
    before = history.filter(pl.col("year") < 2022)
    assert before.height == uk.filter(pl.col("year") < 2022).height
    assert before["export_potential_usd"].is_null().all()
    assert history["actual_exports_usd"].sum() == pytest.approx(uk["value_usd"].sum())


def test_metadata_holds_no_gbp_rate(artefacts):
    # Every output is nominal USD (#41); £ is no longer converted for display.
    assert not [key for key in artefacts.metadata if "gbp" in key.lower()]


def test_refresh_adds_derived_metadata_to_a_saved_bundle_without_retraining(artefacts):
    # A v1 bundle carries a £ rate; the refreshed copy drops it.
    v1_rate = {"usd_per_gbp": 1.2783, "year": 2024, "source": "Bank of England"}
    older = replace(artefacts, metadata={**artefacts.metadata, "gbp_rate": v1_rate})

    refreshed = refresh(older, version="refreshed")

    assert refreshed.version == "refreshed"
    assert refreshed.metadata["refreshed_from"] == "fixture-test"
    assert "gbp_rate" not in refreshed.metadata
    assert refreshed.fold_models is older.fold_models
    assert refreshed.baseline is older.baseline


# --- Excluded Products (ADR 0005) -----------------------------------------------------------

def test_excluded_products_appear_in_no_baseline_history_or_scenario_row(artefacts):
    for table in (artefacts.baseline, artefacts.baseline_features, artefacts.history):
        assert CRUDE_OIL not in set(table["hs6"])


def test_excluded_products_appear_in_no_training_row(world):
    for table in fixture_training(world).values():
        assert CRUDE_OIL not in set(table["hs6"])


def test_the_artefact_lists_excluded_products_with_reasons(artefacts):
    table = artefacts.excluded_products

    assert table["hs6"].to_list() == [CRUDE_OIL]
    assert "pipelines" in table["reason"][0]
    # The Product lookup keeps it, so a search can say it is excluded rather than unknown.
    assert CRUDE_OIL in set(artefacts.products["hs6"])
