"""Artefacts are saved to and loaded from versioned folders that carry a metadata file."""

import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from polars.testing import assert_frame_equal

from exportspredictor import artefacts as store
from exportspredictor.paths import artefacts_dir, data_dir


def test_round_trip_keeps_tables_metadata_and_predictions(fixture_artefacts, tmp_path):
    folder = store.save(fixture_artefacts, tmp_path)
    loaded = store.load_latest(tmp_path)

    assert folder.name == fixture_artefacts.version
    assert json.loads((folder / "metadata.json").read_text())["baseline_year"] == 2024
    assert loaded.metadata == fixture_artefacts.metadata
    assert_frame_equal(loaded.baseline, fixture_artefacts.baseline)
    assert_frame_equal(loaded.products, fixture_artefacts.products)
    names = loaded.metadata["features"]
    probe = xgb.DMatrix(np.linspace(0.0, 20.0, len(names)).reshape(1, -1), feature_names=names)
    for before, after in zip(fixture_artefacts.fold_models, loaded.fold_models, strict=True):
        assert after.predict(probe) == pytest.approx(before.predict(probe))


def test_an_older_folder_is_read_under_the_glossary_column_names(fixture_artefacts, tmp_path):
    # The first real format-2 folders named these columns blocked_effect_usd and uncertain.
    older = fixture_artefacts.baseline.rename(
        {"blocked_opportunity_value_usd": "blocked_effect_usd", "uncertain_opportunity": "uncertain"}
    )
    store.save(replace(fixture_artefacts, baseline=older), tmp_path)

    assert_frame_equal(store.load_latest(tmp_path).baseline, fixture_artefacts.baseline)


def test_a_version_from_before_the_share_was_stored_reads_it_from_its_baseline(fixture_artefacts, tmp_path):
    # Spec #113 "Totals understate": versions saved before the figure was stored still serve it.
    metadata = {key: value for key, value in fixture_artefacts.metadata.items()
                if key != "potential_share_of_actual_exports"}
    two_pairs = fixture_artefacts.baseline.head(2).with_columns(
        export_potential_usd=pl.Series([30e6, 40e6]), actual_exports_usd=pl.Series([50e6, 50e6])
    )
    store.save(replace(fixture_artefacts, metadata=metadata, baseline=two_pairs), tmp_path)

    assert store.load_latest(tmp_path).potential_share_of_actual_exports == pytest.approx(0.7)


def test_load_latest_picks_the_newest_version(fixture_artefacts, tmp_path):
    for version in ["20260101T000000Z", "20260301T000000Z", "20260201T000000Z"]:
        store.save(replace(fixture_artefacts, metadata={**fixture_artefacts.metadata, "version": version}), tmp_path)

    assert store.load_latest(tmp_path).version == "20260301T000000Z"


def test_folder_without_metadata_is_ignored(fixture_artefacts, tmp_path):
    store.save(fixture_artefacts, tmp_path)
    (tmp_path / "zzz-incomplete").mkdir()

    assert store.load_latest(tmp_path).version == fixture_artefacts.version


def test_missing_artefacts_raise(tmp_path):
    with pytest.raises(store.ArtefactsNotFound):
        store.load_latest(tmp_path / "nothing-here")


def test_a_versions_copy_lineage_runs_back_through_each_version_it_was_copied_from(tmp_path):
    for version, copied_from in [("A", None), ("B", "A"), ("C", "B"), ("D", "gone")]:
        (tmp_path / version).mkdir()
        metadata = {"version": version, **({"copied_from": copied_from} if copied_from else {})}
        (tmp_path / version / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert store.copy_lineage(tmp_path, "C") == ["C", "B", "A"]
    assert store.copy_lineage(tmp_path, "A") == ["A"]
    assert store.copy_lineage(tmp_path, "D") == ["D", "gone"]  # a source no longer on disk ends the walk


def test_data_and_artefact_roots_come_from_env_vars(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPORTSPREDICTOR_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("EXPORTSPREDICTOR_ARTEFACTS_DIR", str(tmp_path / "a"))

    assert data_dir() == tmp_path / "d"
    assert artefacts_dir() == tmp_path / "a"


def test_round_trip_keeps_the_evaluation_report(fixture_artefacts, tmp_path):
    folder = store.save(fixture_artefacts, tmp_path)

    assert (folder / "evaluation.json").exists()
    assert store.load(folder).evaluation == fixture_artefacts.evaluation


def test_a_folder_without_an_evaluation_still_loads(fixture_artefacts, tmp_path):
    folder = store.save(replace(fixture_artefacts, evaluation=None), tmp_path)

    assert not (folder / "evaluation.json").exists()
    assert store.load(folder).evaluation is None


def test_fixture_artefacts_are_saved_as_format_4_with_the_model_form_and_market_reach_they_serve(
    fixture_artefacts, tmp_path
):
    folder = store.save(fixture_artefacts, tmp_path)
    loaded = store.load(folder)

    assert json.loads((folder / "metadata.json").read_text())["artefact_format"] == 4
    assert loaded.model_form == fixture_artefacts.model_form
    assert loaded.market_reach is not None and loaded.market_reach.equals(fixture_artefacts.market_reach)
    assert loaded.model_form.market_reach is not None
    assert loaded.model_form.market_reach.table is not None
    assert loaded.model_form.name == "blend"
    assert 0 < loaded.model_form.naive_weight < 1


@pytest.mark.parametrize(
    ("artefact_format", "named"),
    [(3, "artefact format 3"), (2, "artefact format 2"), (1, "format 1"), (None, "no artefact_format")],
)
def test_load_refuses_folders_of_another_format(fixture_artefacts, tmp_path, artefact_format, named):
    metadata = {k: v for k, v in fixture_artefacts.metadata.items() if k != "artefact_format"}
    if artefact_format is not None:
        metadata["artefact_format"] = artefact_format
    folder = store.save(replace(fixture_artefacts, metadata=metadata), tmp_path)

    with pytest.raises(store.UnreadableArtefacts, match=named):
        store.load(folder)


def test_load_refuses_a_folder_missing_a_table(fixture_artefacts, tmp_path):
    folder = store.save(fixture_artefacts, tmp_path)
    (folder / "history.parquet").unlink()

    with pytest.raises(store.UnreadableArtefacts, match="history"):
        store.load(folder)


def test_excluded_products_round_trip(fixture_artefacts, tmp_path):
    loaded = store.load(store.save(fixture_artefacts, tmp_path))

    assert loaded.excluded_products.equals(fixture_artefacts.excluded_products)


def test_round_trip_keeps_the_ppml_benchmark_of_a_full_run(fixture_artefacts, tmp_path):
    assert fixture_artefacts.ppml_benchmark is not None
    folder = store.save(fixture_artefacts, tmp_path)

    assert store.load(folder).ppml_benchmark == fixture_artefacts.ppml_benchmark


def test_a_format_4_folder_without_the_ppml_benchmark_still_loads(fixture_default_artefacts, tmp_path):
    folder = store.save(fixture_default_artefacts, tmp_path)

    assert not (folder / store.PPML_BENCHMARK_FILE).exists()
    bundle = store.load(folder)
    assert bundle.ppml_benchmark is None
    assert bundle.metadata["artefact_format"] == 4


def test_a_copied_version_keeps_the_sources_optional_results_unless_given_new_ones(fixture_artefacts, tmp_path):
    source = store.save(fixture_artefacts, tmp_path / "source")
    metadata = {**fixture_artefacts.metadata, "version": "copy"}

    kept = store.load(store.copy_version(source, tmp_path, metadata=metadata))
    benchmark = {**fixture_artefacts.ppml_benchmark, "seconds": -1.0}
    report = {**fixture_artefacts.evaluation, "marker": "new"}
    replaced = store.load(store.copy_version(
        source, tmp_path, metadata={**metadata, "version": "copy-2"}, evaluation=report, ppml_benchmark=benchmark,
    ))

    assert (kept.evaluation, kept.ppml_benchmark) == (fixture_artefacts.evaluation, fixture_artefacts.ppml_benchmark)
    assert (replaced.evaluation, replaced.ppml_benchmark) == (report, benchmark)


def test_load_refuses_a_baseline_without_a_declared_tier_s_columns_and_names_recaveat(fixture_artefacts, tmp_path):
    older = replace(fixture_artefacts, baseline=fixture_artefacts.baseline.drop("flagged_product_reason"))
    folder = store.save(older, tmp_path)

    with pytest.raises(store.UnreadableArtefacts, match="flagged_product_reason") as refused:
        store.load(folder)
    assert "python -m exportspredictor.pipeline recaveat" in str(refused.value)
    # `recaveat` itself reads it, marking the Baseline again; no tier is required then.
    assert store.load(folder, tiers=()).baseline.height == fixture_artefacts.baseline.height


def test_the_api_refuses_a_baseline_without_a_declared_tier_s_columns_with_the_same_message(
    fixture_artefacts, tmp_path
):
    from fastapi.testclient import TestClient

    from api.main import create_app

    store.save(replace(fixture_artefacts, baseline=fixture_artefacts.baseline.drop("flagged_product_reason")), tmp_path)

    with TestClient(create_app(tmp_path)) as client:
        health = client.get("/health")
        opportunities = client.get("/opportunities/top")

    assert health.status_code == 503 and "recaveat" in health.json()["reason"]
    assert opportunities.status_code == 503
