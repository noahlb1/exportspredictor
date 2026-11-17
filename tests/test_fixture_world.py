"""The fixture builder: a tiny, deterministic synthetic world shared by the test suite and the notebook."""

import polars as pl
from polars.testing import assert_frame_equal

from exportspredictor.exclusions import exclusion_reason
from exportspredictor.fixture import UK, build_world


def test_world_has_the_spec_dimensions():
    world = build_world()

    assert world.flows["exporter"].n_unique() == 4
    assert UK in world.flows["exporter"].unique().to_list()
    assert world.flows["hs6"].n_unique() == 6
    assert world.flows["importer"].n_unique() == 3
    assert world.flows["year"].n_unique() == 4
    assert world.products.height == 6
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


def test_world_carries_every_feature_source_for_its_countries():
    world = build_world()
    countries = set(world.flows["exporter"]) | set(world.flows["importer"])

    assert set(world.country_year["iso3"]) == countries
    assert world.country_year.select("iso3", "year").is_duplicated().sum() == 0
    pairs = set(world.pairs.select("iso3_o", "iso3_d").iter_rows())
    assert {(e, i) for e, i in world.flows.select("exporter", "importer").unique().iter_rows()} <= pairs
    assert {"trade_agreement", "customs_union"} <= set(world.policy.columns)
    assert {"iso3", "year", "eu", "cptpp", "efta"} == set(world.blocs.columns)
    assert world.high_income <= countries


def test_world_has_exactly_one_excluded_product():
    world = build_world()

    assert [hs6 for hs6 in world.products["hs6"] if exclusion_reason(hs6)] == ["270900"]
