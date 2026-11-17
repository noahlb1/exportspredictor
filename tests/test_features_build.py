"""The real-data feature build, run on a tiny data root laid out like the derived BACI, gravity and policy tables."""

import json

import polars as pl

from exportspredictor import features
from exportspredictor.fixture import UK


def test_build_writes_training_rows_and_the_uk_grid_per_year(derived_data_root):
    out = features.build(derived_data_root, years=[2023, 2024], rate=0.5, seed=1)

    training = pl.read_parquet(out / "training_2023.parquet")
    grid = pl.read_parquet(out / "uk_grid_2024.parquet")
    assert set(training.columns) >= {*features.KEYS, *features.FEATURES, "value_usd", "weight"}
    assert set(grid["exporter"]) == {UK}
    assert grid.height == 5 * 3
    assert set(training["weight"]) == {1.0, 2.0}


def test_build_maps_baci_codes_to_standard_iso3(derived_data_root):
    out = features.build(derived_data_root, years=[2023], rate=1.0)

    taiwan = pl.read_parquet(out / "training_2023.parquet").filter(pl.col("exporter") == "TWN")
    assert taiwan.height > 0
    assert taiwan["log_gdp_o"].null_count() == 0
    assert taiwan["log_distance"].null_count() == 0


def test_build_records_its_settings(derived_data_root):
    out = features.build(derived_data_root, years=[2023], rate=0.5, seed=4)

    settings = json.loads((out / "features.json").read_text())
    assert settings["zero_sampling_rate"] == 0.5
    assert settings["zero_sampling_seed"] == 4
    assert settings["years"] == [2023]
    assert settings["features"] == features.FEATURES


def test_build_drops_excluded_products_and_records_the_list(derived_data_root):
    out = features.build(derived_data_root, years=[2024], rate=1.0)

    for name in ("training_2024", "uk_grid_2024"):
        assert "270900" not in set(pl.read_parquet(out / f"{name}.parquet")["hs6"].cast(pl.String))
    settings = json.loads((out / "features.json").read_text())
    assert settings["excluded_products"][0] == "Sector 27 except 2712"


def test_build_trains_only_on_the_recorded_exporter_scope(derived_data_root):
    out = features.build(derived_data_root, years=[2023], rate=0.5, scope_size=2)

    settings = json.loads((out / "features.json").read_text())
    scope = settings["exporter_scope"]
    assert UK in scope and len(scope) in (2, 3)
    training = pl.read_parquet(out / "training_2023.parquet")
    assert set(training["exporter"]) == set(scope) & {UK, "DEU", "FRA", "TWN"}
    assert len(set(training["exporter"])) < 4  # an Exporter fell out of scope
    assert settings["uk_zero_sampling_rate"] == 1.0
    uk_zeros = training.filter((pl.col("exporter") == UK) & (pl.col("value_usd") == 0))
    assert uk_zeros.height > 0 and (uk_zeros["weight"] == 1.0).all()
