"""Artefacts are saved to and loaded from versioned folders that carry a metadata file."""

import json
from dataclasses import replace

import numpy as np
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


def test_fixture_artefacts_are_saved_as_format_2(fixture_artefacts, tmp_path):
    folder = store.save(fixture_artefacts, tmp_path)

    assert json.loads((folder / "metadata.json").read_text())["artefact_format"] == 2


@pytest.mark.parametrize(("artefact_format", "named"), [(1, "format 1"), (None, "no artefact_format")])
def test_load_refuses_folders_of_another_format(fixture_artefacts, tmp_path, artefact_format, named):
    metadata = {k: v for k, v in fixture_artefacts.metadata.items() if k != "artefact_format"}
    if artefact_format is not None:
        metadata["artefact_format"] = artefact_format
    folder = store.save(replace(fixture_artefacts, metadata=metadata), tmp_path)

    with pytest.raises(store.UnreadableArtefacts, match=named):
        store.load(folder)


def test_load_refuses_a_format_2_folder_missing_a_table(fixture_artefacts, tmp_path):
    folder = store.save(fixture_artefacts, tmp_path)
    (folder / "history.parquet").unlink()

    with pytest.raises(store.UnreadableArtefacts, match="history"):
        store.load(folder)


def test_excluded_products_round_trip(fixture_artefacts, tmp_path):
    loaded = store.load(store.save(fixture_artefacts, tmp_path))

    assert loaded.excluded_products.equals(fixture_artefacts.excluded_products)
