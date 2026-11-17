"""The real-data training run, on a tiny data root laid out like the derived tables (CPU, seconds)."""

import polars as pl
import pytest

from exportspredictor import features, pipeline
from exportspredictor.lookups import market_lookup

CONFIGS = [{"max_depth": 2, "eta": 0.3}, {"max_depth": 3, "eta": 0.3}]


@pytest.fixture
def trained(derived_data_root):
    features.build(derived_data_root, years=[2021, 2022, 2023, 2024], rate=1.0)
    return pipeline.build_real_artefacts(
        derived_data_root, version="real-test", configs=CONFIGS, device="cpu", max_rounds=30, non_uk_rate=0.5
    )


def test_baseline_covers_every_uk_product_market_pair_of_the_baseline_year(trained, derived_data_root):
    grid = pl.read_parquet(features.features_dir(derived_data_root) / "uk_grid_2024.parquet")

    assert trained.baseline.height == grid.height == 5 * 3
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
    assert len(metadata["tuning"]) == len(CONFIGS)
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
