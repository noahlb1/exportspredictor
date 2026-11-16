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
