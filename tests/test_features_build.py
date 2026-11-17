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
    assert grid.height == 5 * 4
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


def test_build_adds_relatedness_import_growth_and_the_reexport_ratio(derived_data_root):
    out = features.build(derived_data_root, years=[2024], rate=1.0)

    grid = pl.read_parquet(out / "uk_grid_2024.parquet")
    training = pl.read_parquet(out / "training_2024.parquet")
    for table in (grid, training):
        for name in ("relatedness", "market_import_growth", "reexport_ratio"):
            assert table[name].null_count() < table.height, name
    assert grid["market_import_growth"].null_count() == 0  # 2021 is on disk: three years earlier
    settings = json.loads((out / "features.json").read_text())
    assert settings["proximity_years"] == [2021]  # the fixture's only year before 2022
    assert pl.read_parquet(out / "proximity.parquet").height > 0
