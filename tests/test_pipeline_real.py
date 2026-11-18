"""The real-data training run, on a tiny data root laid out like the derived tables (CPU, seconds)."""

import json

import polars as pl
import pytest

from exportspredictor import constraints, features, pipeline, search
from exportspredictor.lookups import market_lookup



def _tiny_search(root):
    data = search.feature_data(root)
    return search.run(data, data, search.FIXTURE_SEARCH)


@pytest.fixture
def trained(derived_data_root, v1_frozen):
    features.build(derived_data_root, years=[2021, 2022, 2023, 2024], rate=1.0)
    return pipeline.build_real_artefacts(
        derived_data_root, version="real-test", search_results=_tiny_search(derived_data_root), device="cpu",
        non_uk_rate=0.5,
        backtest_plan=pipeline.FIXTURE_BACKTESTS, constraint_plan=constraints.FIXTURE_PLAN, v1_frozen=v1_frozen,
    )


def test_baseline_covers_every_uk_product_market_pair_of_the_baseline_year(trained, derived_data_root):
    grid = pl.read_parquet(features.features_dir(derived_data_root) / "uk_grid_2024.parquet")

    assert trained.baseline.height == grid.height == 5 * 4
    assert trained.baseline.select("hs6", "iso3").is_duplicated().sum() == 0
    assert trained.baseline.schema["hs6"] == pl.String
    assert (trained.baseline["export_potential_usd"] > 0).all()


def test_markets_cover_the_baseline_under_standard_iso3(trained):
    assert set(trained.baseline["iso3"]) <= set(trained.markets["iso3"])
    assert "S19" not in set(trained.markets["iso3"])
    assert "TWN" in set(trained.markets["iso3"])


def test_metadata_records_the_real_source_tuning_and_sampling(trained):
    metadata = trained.metadata

    assert metadata["source"] == "BACI V202601"
    assert metadata["baseline_year"] == 2024
    assert metadata["validation_year"] == 2022
    assert metadata["training_years"] == [2021, 2022, 2023]
    assert len(metadata["search"]["trials"]) == search.FIXTURE_SEARCH.n_trials
    assert metadata["zero_sampling_rate"] == 1.0
    assert metadata["non_uk_sampling_rate"] == 0.5


def test_market_lookup_uses_standard_iso3_and_names_taiwan():
    markets = pl.DataFrame(
        {"iso3": ["S19", "JPN"], "name": ["Other Asia, nes", "Japan"], "region": ["East Asia & Pacific"] * 2}
    )
    country_map = pl.DataFrame({"baci_iso3": ["S19", "JPN"], "iso3": ["TWN", "JPN"]})

    looked_up = market_lookup(markets, country_map)

    assert looked_up.sort("iso3").rows() == [
        ("JPN", "Japan", "East Asia & Pacific"),
        ("TWN", "Taiwan", "East Asia & Pacific"),
    ]


def test_the_real_run_evaluates_on_the_test_year(trained):
    report = trained.evaluation

    assert report["test_year"] == 2023
    assert report["test_training_years"] == [2021, 2022]
    assert report["non_uk_sampling_rate"] == 0.5
    assert report["accuracy"]["uk_out_of_fold"]["overall"]["rows"] == 5 * 4
    assert report["headline"]["text"]


def test_an_existing_artefact_is_evaluated_into_a_new_version_with_its_models_copied(trained, derived_data_root, tmp_path):
    from dataclasses import replace

    from exportspredictor import artefacts as store
    from exportspredictor import evaluation

    old = replace(trained, metadata={**trained.metadata, "version": "20000101T000000Z"}, evaluation=None)
    source = store.save(old, tmp_path)
    (source / "other_step.parquet").write_bytes(b"a table this code does not know")

    folder = evaluation.add_to_latest(derived_data_root, tmp_path, device="cpu")

    evaluated = store.load_latest(tmp_path)
    assert evaluated.version == folder.name != old.version
    assert evaluated.metadata["copied_from"] == old.version
    assert evaluated.baseline.equals(trained.baseline)
    assert evaluated.evaluation["test_year"] == 2023
    assert (folder / "other_step.parquet").read_bytes() == b"a table this code does not know"
    assert sorted(p.name for p in (folder / "models").iterdir()) == [f"fold_{k}.ubj" for k in range(5)]
    # The test-year models are cached under the data root, so a re-run does not retrain.
    assert (evaluation.test_models_dir(derived_data_root, trained.metadata) / evaluation.TEST_MODELS_FILE).exists()


def test_backtests_are_added_to_the_newest_evaluated_artefact_with_cached_models(
    trained, derived_data_root, tmp_path, v1_frozen
):
    from dataclasses import replace

    from exportspredictor import artefacts as store
    from exportspredictor import backtests

    evaluated = {k: v for k, v in trained.evaluation.items() if k not in ("lift", "event_backtest")}
    old = replace(trained, metadata={**trained.metadata, "version": "20000101T000000Z"}, evaluation=evaluated)
    store.save(old, tmp_path)

    folder = backtests.add_to_latest(
        derived_data_root, tmp_path, plan=pipeline.FIXTURE_BACKTESTS, device="cpu", v1_frozen=v1_frozen
    )

    result = store.load_latest(tmp_path)
    assert result.version == folder.name != old.version
    assert result.evaluation["test_year"] == 2023  # the accuracy report is kept
    assert result.evaluation["lift"]["ci"] and result.evaluation["event_backtest"]["events"]
    assert result.evaluation["lift"]["held_out_exporters"] == ["FRA"]  # v1's list within scope
    cached = sorted(p.name.split("_")[0] for p in backtests.cache_dir(derived_data_root).iterdir())
    assert cached == ["lift", "uk"]  # the Lift and the Event Backtest models; a re-run reads them
    again = store.load(
        backtests.add_to_latest(
            derived_data_root, tmp_path, plan=pipeline.FIXTURE_BACKTESTS, device="cpu", version="99990101T000000Z",
            v1_frozen=v1_frozen,
        )
    )
    assert again.evaluation["lift"]["lift"] == result.evaluation["lift"]["lift"]


def test_baseline_and_history_hold_no_excluded_product(trained):
    assert "270900" not in set(trained.baseline["hs6"]) | set(trained.history["hs6"])
    assert trained.excluded_products["hs6"].to_list() == ["270900"]


def test_feature_files_built_without_the_exclusions_are_refused(derived_data_root):
    folder = features.build(derived_data_root, years=[2021, 2022, 2023, 2024], rate=1.0)
    settings = json.loads((folder / "features.json").read_text())
    del settings["excluded_products"]  # as v1's feature build wrote it
    (folder / "features.json").write_text(json.dumps(settings))

    with pytest.raises(ValueError, match="rebuild"):
        pipeline.build_real_artefacts(derived_data_root, device="cpu")


def test_feature_files_built_without_the_gap_period_union_are_refused(derived_data_root):
    folder = features.build(derived_data_root, years=[2021, 2022, 2023, 2024], rate=1.0)
    settings = json.loads((folder / "features.json").read_text())
    del settings["gap_years"]  # grids built one year at a time, as before the Gap Period
    (folder / "features.json").write_text(json.dumps(settings))

    with pytest.raises(ValueError, match="Gap Period"):
        pipeline.build_real_artefacts(derived_data_root, device="cpu")


def test_the_real_run_scores_every_gap_period_year(trained):
    assert trained.metadata["gap_years"] == [2022, 2024]
    gap_rows = trained.history.filter(pl.col("year") >= 2022)
    assert gap_rows.height == trained.baseline.height * 3
    assert gap_rows["export_gap_usd"].is_not_null().all()


def test_the_real_run_without_a_saved_search_is_refused(derived_data_root):
    features.build(derived_data_root, years=[2021, 2022, 2023, 2024], rate=1.0)

    with pytest.raises(ValueError, match="exportspredictor.search"):
        pipeline.build_real_artefacts(derived_data_root, device="cpu")
