"""The fixture builder: a tiny, deterministic synthetic world shared by the test suite and the notebook."""

import polars as pl
from polars.testing import assert_frame_equal

from exportspredictor.fixture import UK, build_world


def test_world_has_the_spec_dimensions():
    world = build_world()

    assert world.flows["exporter"].n_unique() == 4
    assert UK in world.flows["exporter"].unique().to_list()
    assert world.flows["hs6"].n_unique() == 5
    assert world.flows["importer"].n_unique() == 3
    assert world.flows["year"].n_unique() == 4
    assert world.products.height == 5
    assert world.markets.height == 3


def test_world_is_deterministic():
    first, second = build_world(), build_world()

    assert_frame_equal(first.flows, second.flows)
    assert_frame_equal(first.policy, second.policy)


def test_world_contains_a_trade_agreement_change():
    switched = (
        build_world()
        .policy.group_by("exporter", "importer")
        .agg(pl.col("trade_agreement").n_unique().alias("states"))
        .filter(pl.col("states") > 1)
    )

    assert switched.height >= 1


def test_products_are_zero_padded_hs6_strings():
    world = build_world()

    assert all(len(code) == 6 and code.isdigit() for code in world.products["hs6"])
