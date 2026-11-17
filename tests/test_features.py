"""The feature table: leakage guards, relevant-zero sampling and the UK scoring grid.

Rows and features are built through the package's public functions on the fixture world,
then the world's Flows are perturbed to show what each feature does and does not see.
"""

import polars as pl
import pytest

from exportspredictor.features import (
    DEMAND_FEATURES,
    FEATURES,
    KEYS,
    SUPPLY_FEATURES,
    FeatureTransform,
    build_features,
    training_rows,
    uk_grid,
)
from exportspredictor.fixture import UK, build_world

YEAR = 2023
WHISKY = "220830"
TARGET = (UK, "JPN", WHISKY)  # UK whisky to Japan
OTHER_MARKET = "CAN"
CRUDE_OIL = "270900"  # the fixture world's Excluded Product (Sector 27)


@pytest.fixture
def world():
    return build_world()


def _is(exporter, importer, hs6) -> pl.Expr:
    return (
        (pl.col("year") == YEAR)
        & (pl.col("exporter") == exporter)
        & (pl.col("importer") == importer)
        & (pl.col("hs6") == hs6)
    )


def _set_flows(flows: pl.DataFrame, changes: dict[tuple, float]) -> pl.DataFrame:
    value = pl.col("value_usd")
    for key, new in changes.items():
        value = pl.when(_is(*key)).then(pl.lit(new)).otherwise(value)
    return flows.with_columns(value_usd=value)


def _value(flows: pl.DataFrame, exporter, importer, hs6) -> float:
    return flows.filter(_is(exporter, importer, hs6))["value_usd"][0]


def _features_for(world, flows, keys: list[tuple]) -> pl.DataFrame:
    """Features for the given (exporter, importer, hs6) rows in YEAR, from `flows`."""
    year_flows = flows.filter(pl.col("year") == YEAR)
    rows = pl.DataFrame(
        [(*key, YEAR, _value(flows, *key)) for key in keys],
        schema={"exporter": pl.String, "importer": pl.String, "hs6": pl.String, "year": pl.Int16, "value_usd": pl.Float64},
        orient="row",
    )
    return build_features(rows, year_flows, world.sources)


def _same(before: pl.DataFrame, after: pl.DataFrame, row: int, names) -> bool:
    def close(a, b) -> bool:  # missing (None) only matches missing
        return a == b if a is None or b is None else a == pytest.approx(b, rel=1e-6, nan_ok=True)

    return all(close(before[n][row], after[n][row]) for n in names)


# --- Leave-one-out supply and demand ----------------------------------------------------


def test_supply_features_leave_out_the_target_market(world):
    keys = [TARGET, (UK, OTHER_MARKET, WHISKY)]
    before = _features_for(world, world.flows, keys)
    after = _features_for(world, _set_flows(world.flows, {TARGET: 9e9}), keys)

    # The target row's supply features never see its own Market...
    assert _same(before, after, 0, SUPPLY_FEATURES)
    # ...while the same Product's row for another Market does: left out, not ignored.
    assert not _same(before, after, 1, SUPPLY_FEATURES)


def test_demand_features_leave_out_the_target_exporter(world):
    keys = [TARGET, ("DEU", "JPN", WHISKY)]
    before = _features_for(world, world.flows, keys)
    after = _features_for(world, _set_flows(world.flows, {TARGET: 9e9}), keys)

    assert _same(before, after, 0, DEMAND_FEATURES)
    assert not _same(before, after, 1, ["log_demand_loo"])


def test_no_feature_is_built_from_the_pairs_trade_in_other_products(world):
    """Moving the UK's other-Product exports from Japan to Canada leaves UK–Japan whisky unchanged.

    A feature such as total bilateral trade would move with that shift. Every feature here
    depends only on the target Product, the two countries, the year and policy. Relatedness
    is the exception by design: it reads what the UK exports to every *other* Market, which
    this move changes, while it never reads Japan's Flows at all (see
    test_relatedness_and_market_trends.py), so it cannot carry the pair's trade either.
    """
    moves = {}
    for hs6 in sorted(set(world.products["hs6"]) - {WHISKY}):
        japan, canada = _value(world.flows, UK, "JPN", hs6), _value(world.flows, UK, OTHER_MARKET, hs6)
        moved = japan * 0.9
        moves[(UK, "JPN", hs6)] = japan - moved
        moves[(UK, OTHER_MARKET, hs6)] = canada + moved

    before = _features_for(world, world.flows, [TARGET])
    after = _features_for(world, _set_flows(world.flows, moves), [TARGET])

    assert _same(before, after, 0, [f for f in FEATURES if f != "relatedness"])


# --- Training rows: non-zero Flows plus sampled relevant zeros ----------------------------

JEWELLERY = "711319"


def _world_with_an_irrelevant_product(world) -> pl.DataFrame:
    """The USA stops exporting jewellery in 2023, so its jewellery cells that year are not relevant zeros."""
    usa_jewellery = (pl.col("exporter") == "USA") & (pl.col("hs6") == JEWELLERY) & (pl.col("year") == YEAR)
    return world.flows.with_columns(value_usd=pl.when(usa_jewellery).then(0.0).otherwise("value_usd"))


def _relevant_zero_cells(flows: pl.DataFrame) -> set[tuple]:
    """Worked out cell by cell: the Exporter exports the Product to someone and the Importer
    imports it from someone that year, but not from each other. Excluded Products have none."""
    active = {(e, i, k, y) for e, i, k, y, v in flows.iter_rows() if v > 0 and k != CRUDE_OIL}
    exports = {(e, k, y) for e, _, k, y in active}
    imports = {(i, k, y) for _, i, k, y in active}
    exporters = {e for e, *_ in exports}
    importers = {i for i, *_ in imports}
    years = {y for *_, y in active}
    products = {k for _, k, _ in exports}
    return {
        (e, i, k, y)
        for e in exporters for i in importers for k in products for y in years
        if e != i and (e, k, y) in exports and (i, k, y) in imports and (e, i, k, y) not in active
    }


def test_training_rows_keep_every_non_zero_flow_with_weight_one(world):
    rows = training_rows(world.flows, rate=0.5, seed=0)
    non_zero = world.flows.filter((pl.col("value_usd") > 0) & (pl.col("hs6") != CRUDE_OIL))

    kept = rows.filter(pl.col("value_usd") > 0)
    assert kept.height == non_zero.height
    assert (kept["weight"] == 1.0).all()
    assert kept.select(KEYS).sort(KEYS).equals(non_zero.select(KEYS).sort(KEYS))


def test_sampled_zeros_are_only_relevant_zeros_with_weight_one_over_rate(world):
    flows = _world_with_an_irrelevant_product(world)
    relevant = _relevant_zero_cells(flows)
    irrelevant_usa_jewellery = {("USA", i, JEWELLERY, YEAR) for i in ["JPN", "CAN", "IND", "HKG"]}
    assert not (irrelevant_usa_jewellery & relevant)  # the test world has irrelevant zeros

    zeros = training_rows(flows, rate=1.0, seed=0).filter(pl.col("value_usd") == 0)

    assert set(zeros.select(KEYS).iter_rows()) == relevant  # rate 1 keeps every relevant zero
    sampled = training_rows(flows, rate=0.25, seed=3).filter(pl.col("value_usd") == 0)
    assert set(sampled.select(KEYS).iter_rows()) <= relevant
    # Other Exporters' zeros carry 1/rate; UK Zero Cells are never sampled (test_training_scope).
    assert (sampled.filter(pl.col("exporter") != UK)["weight"] == 4.0).all()


def test_zero_weights_reproduce_the_number_of_relevant_zeros_on_average(world):
    flows = _world_with_an_irrelevant_product(world)
    true_count = len(_relevant_zero_cells(flows))
    rate, seeds = 0.3, range(400)

    estimates = [
        training_rows(flows, rate=rate, seed=seed).filter(pl.col("value_usd") == 0)["weight"].sum()
        for seed in seeds
    ]
    mean = sum(estimates) / len(estimates)
    # Each estimate is Binomial(true_count, rate) / rate; allow four standard errors.
    standard_error = (true_count * (1 - rate) / rate) ** 0.5 / len(estimates) ** 0.5
    assert abs(mean - true_count) < 4 * standard_error
    assert len(set(estimates)) > 1  # seeds really change the sample


def test_sampling_is_deterministic_for_a_seed(world):
    first = training_rows(world.flows, rate=0.5, seed=7)
    second = training_rows(world.flows, rate=0.5, seed=7)

    assert first.sort(KEYS).equals(second.sort(KEYS))


# --- The UK scoring grid ------------------------------------------------------------------


def test_uk_grid_has_every_product_and_market_with_actual_exports(world):
    flows = world.flows.filter(pl.col("year") == YEAR)

    grid = uk_grid(flows, UK)

    # 5 Products x 4 Markets: the UK is never its own Market, and crude oil is excluded.
    assert grid.height == 5 * 4
    assert set(grid["exporter"]) == {UK}
    assert grid.select("importer", "hs6").is_duplicated().sum() == 0
    assert grid.filter(_is(*TARGET))["value_usd"][0] == _value(flows, *TARGET)


# --- Transforms are fitted on training years only -----------------------------------------

TRAINING_YEARS = [2021, 2022]


def _all_year_features(world, flows) -> pl.DataFrame:
    return build_features(training_rows(flows, rate=1.0), flows, world.sources)


def test_transform_is_unchanged_by_test_year_data(world):
    test_year = pl.col("year") >= 2023
    distorted = world.flows.with_columns(value_usd=pl.when(test_year).then(pl.col("value_usd") * 1000).otherwise("value_usd"))

    fitted = FeatureTransform.fit(_all_year_features(world, world.flows), TRAINING_YEARS)
    refitted = FeatureTransform.fit(_all_year_features(world, distorted), TRAINING_YEARS)

    assert fitted == refitted
    # The distortion is large enough that fitting on every year would have moved it.
    assert FeatureTransform.fit(_all_year_features(world, distorted), [2021, 2022, 2023]) != fitted


def test_transform_clips_later_years_to_the_training_range(world):
    distorted = world.flows.with_columns(
        value_usd=pl.when(pl.col("year") == 2024).then(pl.col("value_usd") * 1000).otherwise("value_usd")
    )
    table = _all_year_features(world, distorted)
    fitted = FeatureTransform.fit(table, TRAINING_YEARS)
    training_max = table.filter(pl.col("year").is_in(TRAINING_YEARS))["log_supply_loo"].max()

    applied = fitted.apply(table.filter(pl.col("year") == 2024))

    assert applied["log_supply_loo"].max() == pytest.approx(training_max)
    assert applied.columns == table.columns


def test_transform_round_trips_through_metadata(world):
    fitted = FeatureTransform.fit(_all_year_features(world, world.flows), TRAINING_YEARS)

    assert FeatureTransform.from_dict(fitted.to_dict()) == fitted


# --- Excluded Products: dropped as rows, still counted in totals (ADR 0005) ---------------


def test_the_fixture_world_trades_an_excluded_product(world):
    assert world.flows.filter((pl.col("hs6") == CRUDE_OIL) & (pl.col("value_usd") > 0)).height > 0


def test_excluded_products_appear_in_no_training_row_and_no_uk_grid_row(world):
    rows = training_rows(world.flows, rate=1.0)
    grid = uk_grid(world.flows, UK)

    assert CRUDE_OIL not in set(rows["hs6"])
    assert CRUDE_OIL not in set(grid["hs6"])
    assert rows.filter(pl.col("value_usd") > 0).height == world.flows.filter(
        (pl.col("value_usd") > 0) & (pl.col("hs6") != CRUDE_OIL)
    ).height


def test_rca_and_aggregates_of_other_products_are_unchanged_by_the_exclusion(world):
    """Features of a non-Excluded row are what they would be with every Product as a row,
    because the totals (Exporter total, world totals in RCA, supply and demand) still count
    the Excluded Product's Flows."""
    year_flows = world.flows.filter(pl.col("year") == YEAR)
    built = build_features(training_rows(year_flows, rate=1.0), year_flows, world.sources).filter(_is(*TARGET))
    without_exclusion = _features_for(world, world.flows, [TARGET])

    assert _same(built, without_exclusion, 0, FEATURES)
    # Crude oil really is in the totals: dropping its Flows would move the UK's whisky RCA.
    no_crude = world.flows.filter(pl.col("hs6") != CRUDE_OIL)
    assert not _same(without_exclusion, _features_for(world, no_crude, [TARGET]), 0, ["rca_loo"])


# --- Flagged Products and Re-export Hubs: trained and scored as normal (ADR 0005) ----------

JEWELLERY_HUB = (UK, "HKG", JEWELLERY)


def test_training_rows_and_the_uk_grid_keep_flagged_products_and_re_export_hubs(world):
    rows = training_rows(world.flows, rate=1.0)
    grid = uk_grid(world.flows, UK)

    assert JEWELLERY in set(rows["hs6"]) and "HKG" in set(rows["importer"])
    assert rows.filter(_is(*JEWELLERY_HUB)).height == world.flows.filter(_is(*JEWELLERY_HUB)).height
    assert grid.filter(_is(*JEWELLERY_HUB)).height == 1
