"""The real feature build drops Excluded Products (ADR 0005); skips without the real BACI data."""

import polars as pl
import pytest

from exportspredictor import baci, features
from exportspredictor.exclusions import excluded_products, is_excluded
from exportspredictor.paths import data_dir

YEAR = 2024
GOLD, CRUDE_OIL = "710812", "270900"


@pytest.fixture(scope="module")
def flows():
    if not (baci.flows_dir() / f"flows_{YEAR}.parquet").exists() or not (
        data_dir() / "derived" / "country_map.parquet"
    ).exists():
        pytest.skip(f"Real BACI Flows for {YEAR} are not under {data_dir()}")
    _, to_iso3 = features.load_sources()
    return features.load_year_flows(YEAR, to_iso3)


def test_real_flows_trade_excluded_products(flows):
    traded = set(flows.filter(is_excluded())["hs6"].cast(pl.String))

    assert {GOLD, CRUDE_OIL} <= traded


def test_the_real_uk_grid_and_training_rows_hold_no_excluded_product(flows):
    grid = features.uk_grid(flows, baci.UK)
    rows = features.training_rows(flows, rate=0.001)

    assert grid.height > 0 and rows.height > 0
    assert grid.filter(is_excluded()).is_empty()
    assert rows.filter(is_excluded()).is_empty()


def test_the_real_excluded_products_table_gives_a_reason_for_each(flows):
    table = excluded_products(baci.products())

    assert {GOLD, CRUDE_OIL} <= set(table["hs6"])
    assert "271210" not in set(table["hs6"])  # petroleum jelly, Sector 27's exception
    assert table["reason"].null_count() == 0
