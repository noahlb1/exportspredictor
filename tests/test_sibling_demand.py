"""Sibling Demand, the UK-free demand proxy (idea 9 of the open review).

Built through `build_features` on the fixture world, with Flows of a whisky sibling added (the fixture's
Products are each alone in their HS4 heading), then perturbed to show what the feature never sees.
"""

import math

import polars as pl
import pytest

from exportspredictor import features
from exportspredictor.features import FEATURES, build_features, uk_grid
from exportspredictor.fixture import UK, build_world

YEAR = 2023
WHISKY = "220830"
GIN = "220850"  # a sibling of whisky: same HS4 heading (2208, spirits)
FEATURE = "log_sibling_demand_loo"
# Gin Flows added to the world in YEAR: (exporter, importer) -> value
GIN_FLOWS = {("DEU", "JPN"): 100.0, ("FRA", "JPN"): 30.0, (UK, "JPN"): 1000.0, ("USA", "CAN"): 7.0}


@pytest.fixture(scope="module")
def world():
    return build_world()


@pytest.fixture(scope="module")
def flows(world) -> pl.DataFrame:
    """The world's Flows of YEAR with gin added."""
    year = world.flows.filter(pl.col("year") == YEAR)
    gin = pl.DataFrame(
        [(e, m, GIN, YEAR, v) for (e, m), v in GIN_FLOWS.items()], schema=year.schema, orient="row"
    )
    return pl.concat([year, gin])


def _value(flows: pl.DataFrame, exporter, importer, hs6) -> float:
    match = flows.filter((pl.col("exporter") == exporter) & (pl.col("importer") == importer) & (pl.col("hs6") == hs6))
    return match["value_usd"].sum()


def _rows(flows: pl.DataFrame, keys: list[tuple]) -> pl.DataFrame:
    rows = pl.DataFrame(
        [(*key, YEAR, _value(flows, *key)) for key in keys],
        schema={"exporter": pl.String, "importer": pl.String, "hs6": pl.String, "year": pl.Int16, "value_usd": pl.Float64},
        orient="row",
    )
    return rows


def _sibling_demand(world, flows, keys) -> list[float]:
    """Each row's Sibling Demand, back from its log."""
    table = build_features(_rows(flows, keys), flows, world.sources)
    return [round(math.expm1(v), 3) for v in table[FEATURE]]


def _set(flows: pl.DataFrame, exporter, importer, hs6, value: float) -> pl.DataFrame:
    hit = (pl.col("exporter") == exporter) & (pl.col("importer") == importer) & (pl.col("hs6") == hs6)
    return flows.with_columns(value_usd=pl.when(hit).then(value).otherwise(pl.col("value_usd")))


def test_sibling_demand_is_the_markets_imports_of_the_products_hs4_siblings_from_other_non_uk_exporters(world, flows):
    keys = [(UK, "JPN", WHISKY), ("DEU", "JPN", WHISKY), ("FRA", "JPN", WHISKY), (UK, "CAN", WHISKY), (UK, "JPN", GIN)]

    # Japan's gin: Germany 100 and France 30 count for the UK; the UK's own 1,000 never does. Germany's row
    # leaves out Germany's gin (its own trade with Japan in another Product), France's leaves out France's.
    # Canada's gin is the USA's 7. A gin row reads whisky, its sibling, never gin itself.
    expected_gin_row = sum(_value(flows, e, "JPN", WHISKY) for e in ("DEU", "FRA", "USA"))
    assert _sibling_demand(world, flows, keys) == [130.0, 30.0, 100.0, 7.0, pytest.approx(expected_gin_row, rel=1e-6)]  # stored as Float32


def test_a_uk_pairs_sibling_demand_never_moves_with_its_own_flows(world, flows):
    target = [(UK, "JPN", WHISKY)]
    before = _sibling_demand(world, flows, target)

    after_whisky = _sibling_demand(world, _set(flows, UK, "JPN", WHISKY, 9e9), target)
    after_gin = _sibling_demand(world, _set(flows, UK, "JPN", GIN, 9e9), target)

    assert before == after_whisky == after_gin == [130.0]


def test_sibling_demand_is_built_from_no_uk_flow(world, flows):
    """Every UK Flow removed: no row's Sibling Demand moves, the UK grid's or other Exporters'."""
    no_uk = flows.with_columns(value_usd=pl.when(pl.col("exporter") == UK).then(0.0).otherwise(pl.col("value_usd")))
    others = [("DEU", "JPN", WHISKY), ("FRA", "CAN", WHISKY), ("USA", "JPN", GIN)]

    grid_before = build_features(uk_grid(flows, UK), flows, world.sources)
    grid_after = build_features(uk_grid(flows, UK).with_columns(value_usd=pl.lit(0.0)), no_uk, world.sources)

    assert grid_before[FEATURE].null_count() == 0
    assert grid_before[FEATURE].to_list() == grid_after[FEATURE].to_list()
    assert _sibling_demand(world, flows, others) == _sibling_demand(world, no_uk, others)


def test_sibling_demand_is_a_candidate_feature_with_no_monotone_constraint():
    assert FEATURE in features.CANDIDATE_FEATURES
    assert FEATURE not in FEATURES
    assert FEATURE not in features.MONOTONE_CONSTRAINTS
