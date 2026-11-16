"""The real-data feature build, run on a tiny data root laid out like the derived BACI, gravity and policy tables."""

import json

import polars as pl
import pytest

from exportspredictor import baci, features, policy
from exportspredictor.fixture import UK, build_world

TAIWAN_BACI = "S19"  # BACI's code for Taiwan ("Other Asia, nes")


@pytest.fixture
def data_root(tmp_path):
    """The fixture world written to disk as the BACI, gravity and policy build steps write it.

    The USA is renamed to Taiwan, which BACI keys as `S19` and every other table as `TWN`.
    """
    world = build_world()
    rename = {"USA": "TWN"}
    flows = world.flows.filter(pl.col("value_usd") > 0).with_columns(
        pl.col("exporter").replace({"USA": TAIWAN_BACI})
    )
    countries = sorted(set(flows["exporter"]) | set(flows["importer"]))
    codes = pl.Enum(countries)
    derived = tmp_path / "derived"
    baci.flows_dir(tmp_path).mkdir(parents=True)
    for year in world.flows["year"].unique():
        flows.filter(pl.col("year") == year).select(
            "year",
            pl.col("exporter").cast(codes),
            pl.col("importer").cast(codes),
            pl.col("hs6").cast(pl.Enum(sorted(world.products["hs6"]))),
            "value_usd",
        ).write_parquet(baci.flows_dir(tmp_path) / f"flows_{year}.parquet")
    baci.lookups_dir(tmp_path).mkdir(parents=True)
    pl.DataFrame(
        {
            "iso3": countries,
            "name": countries,
            "region": "x",
            "income_level": ["Lower middle income" if c == "IND" else "High income" for c in countries],
            "baci_code": list(range(len(countries))),
        }
    ).write_parquet(baci.lookups_dir(tmp_path) / "markets.parquet")
    pl.DataFrame(
        {"baci_code": list(range(len(countries))), "baci_iso3": countries,
         "iso3": [{TAIWAN_BACI: "TWN"}.get(c, c) for c in countries]}
    ).write_parquet(derived / "country_map.parquet")
    world.country_year.with_columns(pl.col("iso3").replace(rename)).write_parquet(derived / "country_year.parquet")
    world.pairs.with_columns(pl.col("iso3_o").replace(rename)).write_parquet(derived / "gravity_pairs.parquet")
    paths = policy.policy_paths(tmp_path)
    world.policy.with_columns(pl.col("exporter").replace(rename)).write_parquet(paths["agreements"])
    world.blocs.write_parquet(paths["trading_blocs"])
    return tmp_path


def test_build_writes_training_rows_and_the_uk_grid_per_year(data_root):
    out = features.build(data_root, years=[2023, 2024], rate=0.5, seed=1)

    training = pl.read_parquet(out / "training_2023.parquet")
    grid = pl.read_parquet(out / "uk_grid_2024.parquet")
    assert set(training.columns) >= {*features.KEYS, *features.FEATURES, "value_usd", "weight"}
    assert set(grid["exporter"]) == {UK}
    assert grid.height == 5 * 3
    assert set(training["weight"]) == {1.0, 2.0}


def test_build_maps_baci_codes_to_standard_iso3(data_root):
    out = features.build(data_root, years=[2023], rate=1.0)

    taiwan = pl.read_parquet(out / "training_2023.parquet").filter(pl.col("exporter") == "TWN")
    assert taiwan.height > 0
    assert taiwan["log_gdp_o"].null_count() == 0
    assert taiwan["log_distance"].null_count() == 0


def test_build_records_its_settings(data_root):
    out = features.build(data_root, years=[2023], rate=0.5, seed=4)

    settings = json.loads((out / "features.json").read_text())
    assert settings["zero_sampling_rate"] == 0.5
    assert settings["zero_sampling_seed"] == 4
    assert settings["years"] == [2023]
    assert settings["features"] == features.FEATURES
