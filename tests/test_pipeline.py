"""Training through the package: fixture world in, a Baseline table out-of-fold for every UK pair."""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from exportspredictor.evaluation import naive_potential
from exportspredictor.features import (
    FEATURES,
    POLICY_FEATURES,
    FeatureTransform,
    build_features,
    training_rows,
    uk_grid,
)
from exportspredictor.fixture import UK
from exportspredictor.folds import assign_folds
from exportspredictor.gap_period import with_baseline_actual_exports
from exportspredictor.loop import NAIVE_EXPERIMENT, SELECTION_MEASURE_TEXT
from exportspredictor.pipeline import (
    FIXTURE_ZERO_SAMPLING_RATE,
    build_artefacts,
    exporter_scope,
    fixture_training,
    refresh,
    rerange,
)
from exportspredictor.prediction import blend
from exportspredictor.search import FIXTURE_SEARCH

CRUDE_OIL = "270900"  # the fixture world's Excluded Product


@pytest.fixture
def world(fixture_world):
    return fixture_world


@pytest.fixture
def artefacts(fixture_artefacts):
    return fixture_artefacts


def test_baseline_has_one_row_per_uk_product_market_pair(artefacts):
    baseline = artefacts.baseline

    assert baseline.height == 5 * 4
    assert baseline.select("hs6", "iso3").is_duplicated().sum() == 0


def test_baseline_actual_exports_are_the_gap_period_mean_with_missing_years_as_zero(world, artefacts):
    # GBR→JPN 851712 has no Flow in 2024, and GBR→CAN 851712 none in 2023: they count as $0.
    uk = world.flows.filter(pl.col("exporter") == UK, pl.col("year").is_between(2022, 2024))
    totals = uk.group_by("hs6", iso3="importer").agg(total=pl.col("value_usd").sum())
    joined = artefacts.baseline.join(totals, on=["hs6", "iso3"])

    assert joined.height == 20
    assert joined["actual_exports_usd"].to_list() == pytest.approx((joined["total"] / 3).to_list())
    phones_to_japan = joined.filter(pl.col("hs6") == "851712", pl.col("iso3") == "JPN").row(0, named=True)
    flows = uk.filter(pl.col("hs6") == "851712", pl.col("importer") == "JPN", pl.col("value_usd") > 0)
    assert flows.height == 2
    assert phones_to_japan["actual_exports_usd"] == pytest.approx(flows["value_usd"].sum() / 3)


def test_baseline_year_actual_exports_are_zero_for_a_pair_missing_from_the_history():
    history = pl.DataFrame(
        {"hs6": ["220830", "220830"], "iso3": ["JPN", "JPN"], "year": [2023, 2024], "actual_exports_usd": [5.0, 7.0]}
    )
    pairs = pl.DataFrame({"hs6": ["851712", "220830"], "iso3": ["JPN", "JPN"]})

    shown = with_baseline_actual_exports(pairs, history, 2024)

    assert shown.rows() == [("851712", "JPN", 0.0), ("220830", "JPN", 7.0)]


def test_metadata_records_the_gap_period_and_the_baseline_year(artefacts):
    assert artefacts.metadata["gap_years"] == [2022, 2024]
    assert artefacts.metadata["baseline_year"] == 2024
    assert artefacts.gap_years == (2022, 2024)


def test_history_holds_each_gap_years_export_potential_and_gap_for_every_pair(artefacts):
    gap_rows = artefacts.history.filter(pl.col("year") >= 2022)

    assert gap_rows.height == 20 * 3
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

    assert joined.height == 20
    assert joined["baseline_export_potential_usd"].to_list() == pytest.approx(
        joined["export_potential_usd_2024"].to_list()
    )
    # The mean differs from the 2024 score: the Gap Period is not 2024 alone.
    assert not joined["baseline_export_potential_usd"].equals(joined["export_potential_usd"])


def test_each_gap_years_uk_score_comes_from_the_fold_model_that_held_the_pair_out(world, artefacts):
    transform = FeatureTransform.from_dict(artefacts.metadata["feature_transform"])
    weight = artefacts.model_form.naive_weight
    for year in (2022, 2023, 2024):
        year_flows = world.flows.filter(pl.col("year") == year)
        grid = assign_folds(
            transform.apply(
                naive_potential(build_features(uk_grid(year_flows, UK), world.flows, world.sources), year_flows)
            ).rename({"importer": "iso3"})
        )
        scored = artefacts.history.filter(pl.col("year") == year).join(grid.drop("year"), on=["hs6", "iso3"])
        assert scored.height == 20
        for (fold,), rows in scored.group_by("fold"):
            model = artefacts.fold_models[fold]
            assert model.attr("held_out_fold") == str(fold)
            # The pair's fold model never trained on any of the pair's years (ADR 0001); the
            # served blend takes its Export Potential with the year's naive baseline.
            model_alone = model.inplace_predict(rows.select(FEATURES).to_numpy().astype(np.float32))
            expected = blend(model_alone, rows["naive_usd"].to_numpy(), weight)
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

    assert metadata["validation_year"] == metadata["search"]["validation_year"] == 2022
    assert metadata["training_years"] == [2021, 2022, 2023]
    assert metadata["search"]["trials"][0]["v1_config"]
    assert len(metadata["search"]["trials"]) == FIXTURE_SEARCH.n_trials


def test_fold_models_take_the_configuration_the_ranking_experiments_chose_not_the_searchs(artefacts):
    metadata = artefacts.metadata
    best = metadata["ranking_experiments"]["best"]

    assert metadata["params"] == best["settings"]["params"]


def test_fold_models_take_the_rounds_the_selection_measure_chose_scaled_by_their_training_rows(artefacts):
    metadata = artefacts.metadata
    scaling = metadata["round_scaling"]
    kept = metadata["ranking_experiments"]["experiments"][NAIVE_EXPERIMENT]["arms"][0]

    # The fixture's loop has no five-fold confirmation: the kept arm's own rounds, never the search's.
    assert scaling["selected_rounds"] == kept["rounds"]
    assert "tuned_rounds" not in scaling
    assert scaling["tuning_years"] == [2021]
    assert scaling["training_years"] == [2021, 2022, 2023]
    assert scaling["ratio"] == pytest.approx(scaling["training_rows"] / scaling["tuning_rows"], rel=1e-3)
    assert 2.5 < scaling["ratio"] < 3.5  # three years of rows against one
    assert metadata["rounds"] == scaling["rounds"]
    assert metadata["rounds"] == pytest.approx(scaling["selected_rounds"] * scaling["ratio"], abs=1)


def test_metadata_records_the_device_and_the_search_and_training_time(artefacts):
    metadata = artefacts.metadata

    assert metadata["device"] == "cpu"  # the fixture; the real run records "cuda"
    assert metadata["tuning_seconds"] > 0
    assert metadata["training_seconds"] > 0


def test_the_artefact_serves_the_form_the_ranking_experiments_chose_and_records_them(artefacts):
    ranking = artefacts.metadata["ranking_experiments"]

    assert artefacts.metadata["model_form"] == {
        "form": "blend",
        "naive_weight": ranking["best"]["settings"]["naive_weight"],
    }
    assert ranking["selection_measure"] == SELECTION_MEASURE_TEXT
    assert {"selection_measures", "mean", "sd", "win_threshold"} <= set(ranking["noise_floor"])
    assert ranking["best"]["experiment"] == NAIVE_EXPERIMENT
    assert ranking["experiments"][NAIVE_EXPERIMENT]["arms"][0]["status"] == "kept"


def test_baseline_features_hold_each_pairs_fold_model_inputs_and_naive_baseline(artefacts):
    inputs = artefacts.baseline_features

    assert inputs.select("hs6", "iso3").sort("hs6", "iso3").equals(artefacts.baseline.select("hs6", "iso3"))
    assert set(FEATURES) <= set(inputs.columns)
    assert inputs["naive_usd"].null_count() == 0
    assert (inputs["naive_usd"] >= 0).all()
    assert inputs.equals(assign_folds(inputs.drop("fold")).select(inputs.columns))


def test_history_holds_uk_actual_exports_up_to_the_baseline_year(world, artefacts):
    history = artefacts.history
    uk = world.flows.filter(pl.col("exporter") == UK, pl.col("value_usd") > 0, pl.col("hs6") != CRUDE_OIL)

    assert history.columns == [
        "hs6", "iso3", "year", "actual_exports_usd", "export_potential_usd", "export_gap_usd", "tariff_imputed"
    ]
    assert sorted(history["year"].unique().to_list()) == [2021, 2022, 2023, 2024]
    # Before the Gap Period only years with a Flow are kept, and they carry no score.
    before = history.filter(pl.col("year") < 2022)
    assert before.height == uk.filter(pl.col("year") < 2022).height
    assert before["export_potential_usd"].is_null().all()
    assert history["actual_exports_usd"].sum() == pytest.approx(uk["value_usd"].sum())


def test_metadata_holds_no_gbp_rate(artefacts):
    # Every output is nominal USD (#41); £ is no longer converted for display.
    assert not [key for key in artefacts.metadata if "gbp" in key.lower()]


def test_metadata_records_the_thresholds_the_tables_were_built_with(artefacts):
    # Spec #39 *Artefacts*: the Blocked Opportunity rule, the Relative Export Gap floor and the Potential Range.
    assert artefacts.metadata["thresholds"] == {
        "blocked_min_share": 0.2,
        "blocked_min_usd": 1_000_000.0,
        "relative_gap_min_potential_usd": 1_000_000.0,
        "potential_range_coverage": pytest.approx(0.8),
        "potential_range_bands": 20,
        "potential_range_min_band_rows": 12,
    }


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


def test_refresh_records_the_share_of_actual_exports_that_summed_export_potential_reaches(artefacts):
    # Spec #113 "Totals understate": two pairs, $30m + $40m of Export Potential against $50m + $50m exported: 70%.
    two_pairs = artefacts.baseline.head(2).with_columns(
        export_potential_usd=pl.Series([30e6, 40e6]), actual_exports_usd=pl.Series([50e6, 50e6])
    )

    refreshed = refresh(replace(artefacts, baseline=two_pairs), version="refreshed")

    assert refreshed.metadata["potential_share_of_actual_exports"] == pytest.approx(0.7)
    assert refreshed.potential_share_of_actual_exports == pytest.approx(0.7)


def test_the_pipeline_records_the_share_of_actual_exports_that_summed_export_potential_reaches(artefacts):
    baseline = artefacts.baseline
    expected = baseline["export_potential_usd"].sum() / baseline["actual_exports_usd"].sum()
    assert artefacts.metadata["potential_share_of_actual_exports"] == pytest.approx(expected)


def test_rerange_rebuilds_a_dollar_offset_range_as_the_pipeline_now_builds_it(world, artefacts):
    # A version from before ADR 0007's log-scale amendment: dollar offsets and a range built from them.
    older = replace(
        artefacts,
        potential_range_bands=pl.DataFrame({
            "band": pl.Series([0], dtype=pl.Int16), "min_export_potential_usd": [0.0], "rows": [60],
            "low_offset_usd": [-5.0], "high_offset_usd": [5.0],
        }),
        baseline=artefacts.baseline.with_columns(
            potential_range_low_usd=pl.lit(0.0), potential_range_high_usd=pl.lit(1.0), uncertain_opportunity=False
        ),
        metadata={k: v for k, v in artefacts.metadata.items() if k != "potential_range"}
        | {"potential_range": {k: v for k, v in artefacts.metadata["potential_range"].items() if k != "scale"}},
    )
    year = artefacts.baseline_year
    year_flows = world.flows.filter(pl.col("year") == year)
    calibration = build_features(
        training_rows(year_flows, rate=FIXTURE_ZERO_SAMPLING_RATE, seed=0, uk=UK,
                      exporters=exporter_scope(world.country_year, uk=UK)),
        world.flows,
        world.sources,
    )

    ranged = rerange(older, calibration, year_flows, version="reranged")

    assert ranged.version == "reranged" and ranged.metadata["refreshed_from"] == "fixture-test"
    assert ranged.metadata["potential_range"]["scale"] == "log"
    assert ranged.fold_models is older.fold_models
    assert ranged.potential_range_bands.equals(artefacts.potential_range_bands)
    assert ranged.baseline.equals(artefacts.baseline)
    assert ranged.evaluation["potential_range"] == artefacts.evaluation["potential_range"]


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


def test_baseline_flags_flagged_products_and_re_export_hubs(artefacts):
    flags = {
        (row["hs6"], row["iso3"]): (row["flagged_product"], row["reexport_hub"])
        for row in artefacts.baseline.iter_rows(named=True)
    }

    assert flags[("711319", "HKG")] == (True, True)  # gold jewellery to Hong Kong
    assert flags[("711319", "JPN")] == (True, False)
    assert flags[("220830", "HKG")] == (False, True)
    assert flags[("220830", "JPN")] == (False, False)
    assert artefacts.baseline.schema["flagged_product"] == pl.Boolean
    assert sum(flagged for flagged, _ in flags.values()) == 4  # one Product in every Market
    assert sum(hub for _, hub in flags.values()) == 5  # every Product in one Market


def test_loop_results_validated_on_another_year_are_refused(world, artefacts):
    ranking = {**artefacts.metadata["ranking_experiments"], "validation_year": 2021}

    with pytest.raises(ValueError, match="validated on 2021"):
        build_artefacts(world, version="mismatch", search_results=artefacts.metadata["search"], ranking_results=ranking)


def test_loop_results_chosen_on_another_feature_set_are_refused(world, artefacts):
    """#127: the loop kept its configuration on the rebuilt `exporter_count` files; the build must read those."""
    ranking = artefacts.metadata["ranking_experiments"]
    rebuilt = {**ranking, "experiments": {**ranking["experiments"], "exporter count": {"feature_set": "exporter_count"}}}

    with pytest.raises(ValueError, match="EXPORTSPREDICTOR_FEATURE_SET=exporter_count"):
        build_artefacts(world, version="mismatch", search_results=artefacts.metadata["search"], ranking_results=rebuilt)


def test_the_artefact_records_the_feature_set_it_was_built_from(artefacts):
    assert artefacts.metadata["feature_set"] == ""  # the original files (the fixture's)
