"""The real feature build adds Relatedness, Market import growth and the re-export ratio;
skips without the real BACI data. Proximity is fitted on one year here, to stay fast."""

import math

import polars as pl
import pytest

from exportspredictor import baci, features
from exportspredictor.paths import data_dir

YEAR = 2024
PROXIMITY_YEAR = 2019
WHISKY = "220830"


@pytest.fixture(scope="module")
def uk_grid_features():
    needed = [baci.flows_dir() / f"flows_{y}.parquet" for y in (YEAR, YEAR - features.GROWTH_YEARS, PROXIMITY_YEAR)]
    if not all(path.exists() for path in needed) or not (data_dir() / "derived" / "country_map.parquet").exists():
        pytest.skip(f"Real BACI Flows are not under {data_dir()}")
    sources, to_iso3 = features.load_sources()
    proximity = features.fit_proximity(baci.scan_flows(), [PROXIMITY_YEAR])
    flows = features.load_year_flows(YEAR, to_iso3)
    earlier = features.load_year_flows(YEAR - features.GROWTH_YEARS, to_iso3)
    grid = features.uk_grid(flows, baci.UK)
    return features.build_features(grid, pl.concat([flows, earlier]), features.replace(sources, proximity=proximity))


def test_the_real_uk_grid_carries_all_three_features(uk_grid_features):
    table = uk_grid_features
    imported = table.filter(pl.col("log_demand_loo") > 0)  # the Market imports the Product from others

    for name in ("relatedness", "market_import_growth"):
        assert table[name].null_count() < 0.05 * table.height, name
    assert imported["reexport_ratio"].null_count() == 0
    assert table["relatedness"].min() >= 0 and table["relatedness"].max() <= 1
    assert table["reexport_ratio"].max() <= math.log1p(features.REEXPORT_RATIO_CAP) + 1e-6


def test_the_uk_is_more_related_to_whisky_than_the_average_product(uk_grid_features):
    relatedness = uk_grid_features.filter(pl.col("importer") == "JPN")
    whisky = relatedness.filter(pl.col("hs6").cast(pl.String) == WHISKY)["relatedness"].item()

    assert whisky > relatedness["relatedness"].median()
