"""Relatedness, Market import growth and the re-export ratio (ADR 0006): worked
examples on hand-built Flows, and the leakage guards on the fixture world.
"""

import math
from dataclasses import replace

import polars as pl
import pytest

from exportspredictor.features import (
    ALL_FEATURES,
    FEATURES,
    REEXPORT_RATIO_CAP,
    build_features,
    fit_proximity,
    uk_grid,
)
from exportspredictor.fixture import UK, build_world

WHISKY, MEDICINE, CARS = "220830", "300490", "870323"
FLOW_SCHEMA = {"exporter": pl.String, "importer": pl.String, "hs6": pl.String, "year": pl.Int16, "value_usd": pl.Float64}
NEW_FEATURES = ["relatedness", "market_import_growth", "reexport_ratio"]


def _flows(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=FLOW_SCHEMA, orient="row")


def _rows(keys: list[tuple], flows: pl.DataFrame) -> pl.DataFrame:
    """Rows keyed (exporter, importer, hs6, year), each carrying its own Flow (0 if none)."""
    keyed = _flows([(*key, 0.0) for key in keys]).drop("value_usd")
    return keyed.join(flows, on=["exporter", "importer", "hs6", "year"], how="left").with_columns(
        pl.col("value_usd").fill_null(0.0)
    )


def _proximity(pairs: dict[tuple[str, str], float]) -> pl.DataFrame:
    rows = [(a, b, p) for (a, b), p in pairs.items()] + [(b, a, p) for (a, b), p in pairs.items()]
    return pl.DataFrame(rows, schema={"hs6": pl.String, "hs6_neighbour": pl.String, "proximity": pl.Float32}, orient="row")


@pytest.fixture(scope="module")
def world():
    return build_world()


def test_the_three_features_are_built_and_those_the_ablation_kept_are_model_inputs():
    assert set(NEW_FEATURES) <= set(ALL_FEATURES)
    # The ablation dropped Market import growth: still built, no longer a model input.
    assert set(NEW_FEATURES) & set(FEATURES) == {"relatedness", "reexport_ratio"}


# --- Relatedness ---------------------------------------------------------------------------


def test_relatedness_is_the_proximity_weighted_share_of_neighbours_with_rca_above_one(world):
    # Leaving Japan out, the UK's exports are mostly Medicine, so its RCA is above 1 in
    # Medicine and below 1 in Cars. Whisky's neighbours: Medicine (0.5) and Cars (0.25).
    # So Relatedness = 0.5 / (0.5 + 0.25) = 2/3. Counting the UK's huge car exports to
    # Japan would make the UK specialised in Cars instead, and give a different value.
    flows = _flows([
        (UK, "CAN", WHISKY, 2023, 10.0),
        (UK, "CAN", MEDICINE, 2023, 100.0),
        (UK, "CAN", CARS, 2023, 1.0),
        ("DEU", "CAN", WHISKY, 2023, 50.0),
        ("DEU", "CAN", MEDICINE, 2023, 10.0),
        ("DEU", "CAN", CARS, 2023, 50.0),
        (UK, "JPN", CARS, 2023, 1000.0),
    ])
    sources = replace(world.sources, proximity=_proximity({(WHISKY, MEDICINE): 0.5, (WHISKY, CARS): 0.25}))

    table = build_features(_rows([(UK, "JPN", WHISKY, 2023)], flows), flows, sources)

    assert table["relatedness"].item() == pytest.approx(2 / 3)


def test_proximity_links_products_exported_competitively_together():
    # GBR is competitive in Whisky and Medicine (RCA 3 each); DEU and FRA only in Cars.
    flows = _flows([
        (UK, "JPN", WHISKY, 2020, 10.0),
        (UK, "JPN", MEDICINE, 2020, 10.0),
        ("DEU", "JPN", CARS, 2020, 20.0),
        ("FRA", "JPN", CARS, 2020, 20.0),
    ])

    proximity = fit_proximity(flows, years=[2020])

    pairs = {(a, b): p for a, b, p in proximity.iter_rows()}
    assert pairs == {(WHISKY, MEDICINE): 1.0, (MEDICINE, WHISKY): 1.0}


def test_proximity_never_reads_years_outside_its_training_years(world):
    later = world.flows.with_columns(
        value_usd=pl.when(pl.col("year") >= 2022).then(pl.col("value_usd") * pl.col("hs6").cast(pl.Int64) % 7)
        .otherwise(pl.col("value_usd"))
    )

    assert fit_proximity(later, years=[2021]).equals(fit_proximity(world.flows, years=[2021]))
    assert not fit_proximity(later, years=[2021, 2023]).equals(fit_proximity(world.flows, years=[2021, 2023]))


def _uk_features(world, flows, year=2023) -> pl.DataFrame:
    year_flows = flows.filter(pl.col("year") == year)
    return build_features(uk_grid(year_flows, UK), flows, world.sources)


def test_the_fixture_world_gives_uk_rows_a_relatedness(world):
    relatedness = _uk_features(world, world.flows)["relatedness"]

    assert relatedness.null_count() == 0
    assert relatedness.min() >= 0 and relatedness.max() <= 1
    assert relatedness.n_unique() > 1


def test_changing_the_target_markets_own_flows_leaves_uk_relatedness_unchanged(world):
    into_japan = (pl.col("importer") == "JPN") & (pl.col("year") == 2023)
    changed = world.flows.with_columns(
        value_usd=pl.when(into_japan).then(pl.col("value_usd") * 50 + 1e9).otherwise(pl.col("value_usd"))
    )

    before = _uk_features(world, world.flows).filter(pl.col("importer") == "JPN")
    after = _uk_features(world, changed).filter(pl.col("importer") == "JPN")

    assert after["relatedness"].to_list() == pytest.approx(before["relatedness"].to_list())


def test_uk_relatedness_follows_what_the_uk_exports_to_other_markets(world):
    # Make the UK export almost nothing but Whisky outside Japan: it is then competitive in
    # Whisky alone, and every other Product's Relatedness in Japan becomes Whisky's proximity share.
    elsewhere = (pl.col("exporter") == UK) & (pl.col("importer") != "JPN") & (pl.col("year") == 2023)
    changed = world.flows.with_columns(
        value_usd=pl.when(elsewhere & (pl.col("hs6") == WHISKY)).then(1e12)
        .when(elsewhere).then(0.0)
        .otherwise(pl.col("value_usd"))
    )

    after = _uk_features(world, changed).filter(pl.col("importer") == "JPN")

    proximity = world.sources.proximity
    for hs6, relatedness in after.select("hs6", "relatedness").iter_rows():
        if hs6 == WHISKY:
            continue
        neighbours = proximity.filter(pl.col("hs6") == hs6)
        to_whisky = neighbours.filter(pl.col("hs6_neighbour") == WHISKY)["proximity"].sum()
        assert relatedness == pytest.approx(to_whisky / neighbours["proximity"].sum(), abs=1e-6)


# --- Market import growth ------------------------------------------------------------------


def test_market_import_growth_compares_imports_from_other_exporters_with_three_years_earlier(world):
    # Japan's Whisky imports from everyone but the UK went from 50 to 150.
    flows = _flows([
        (UK, "JPN", WHISKY, 2020, 100.0),
        ("DEU", "JPN", WHISKY, 2020, 50.0),
        (UK, "JPN", WHISKY, 2023, 900.0),
        ("DEU", "JPN", WHISKY, 2023, 100.0),
        ("FRA", "JPN", WHISKY, 2023, 50.0),
    ])

    table = build_features(_rows([(UK, "JPN", WHISKY, 2023)], flows), flows, world.sources)

    assert table["market_import_growth"].item() == pytest.approx(math.log(151 / 51), rel=1e-6)


def test_market_import_growth_is_missing_without_the_earlier_year(world):
    year_flows = world.flows.filter(pl.col("year") == 2023)

    table = build_features(uk_grid(year_flows, UK), world.flows, world.sources)

    assert table.height > 0
    assert table["market_import_growth"].null_count() == table.height
    assert _uk_features(world, world.flows, year=2024)["market_import_growth"].null_count() == 0


def _training_features(world, flows, year=2024) -> pl.DataFrame:
    """Features for every Flow of the year (zero cells included), from `flows`."""
    rows = flows.filter(pl.col("year") == year)
    return build_features(rows, flows, world.sources).sort("exporter", "importer", "hs6")


def test_changing_the_rows_own_exporter_leaves_its_growth_and_reexport_ratio_unchanged(world):
    germany = pl.col("exporter") == "DEU"
    changed = world.flows.with_columns(
        value_usd=pl.when(germany).then(pl.col("value_usd") * 7 + 1e6).otherwise(pl.col("value_usd"))
    )

    before = _training_features(world, world.flows).filter(germany)
    after = _training_features(world, changed).filter(germany)

    for feature in ("market_import_growth", "reexport_ratio"):
        assert after[feature].to_list() == pytest.approx(before[feature].to_list(), rel=1e-5), feature
    others = _training_features(world, changed).filter(~germany)["market_import_growth"]
    assert others.to_list() != pytest.approx(
        _training_features(world, world.flows).filter(~germany)["market_import_growth"].to_list(), rel=1e-5
    )


# --- Re-export ratio -----------------------------------------------------------------------

_HUB_FLOWS = [
    ("JPN", "CAN", WHISKY, 2023, 30.0),  # Japan's exports of Whisky
    (UK, "JPN", WHISKY, 2023, 100.0),
    ("DEU", "JPN", WHISKY, 2023, 60.0),
    (UK, "CAN", WHISKY, 2023, 5.0),
]


def test_the_reexport_ratio_is_the_markets_exports_over_its_imports_from_other_exporters(world):
    flows = _flows(_HUB_FLOWS)

    table = build_features(_rows([(UK, "JPN", WHISKY, 2023)], flows), flows, world.sources)

    assert table["reexport_ratio"].item() == pytest.approx(math.log1p(30 / 60), rel=1e-6)


def test_the_reexport_ratio_ignores_the_rows_own_exporters_flows_but_not_the_markets_exports(world):
    flows = _flows(_HUB_FLOWS)
    row = [(UK, "JPN", WHISKY, 2023)]
    uk_changed = flows.with_columns(
        value_usd=pl.when(pl.col("exporter") == UK).then(pl.col("value_usd") * 40).otherwise(pl.col("value_usd"))
    )
    hub_changed = flows.with_columns(
        value_usd=pl.when(pl.col("exporter") == "JPN").then(pl.col("value_usd") * 2).otherwise(pl.col("value_usd"))
    )

    ratio = build_features(_rows(row, flows), flows, world.sources)["reexport_ratio"].item()
    assert build_features(_rows(row, uk_changed), uk_changed, world.sources)["reexport_ratio"].item() == ratio
    assert build_features(_rows(row, hub_changed), hub_changed, world.sources)["reexport_ratio"].item() == (
        pytest.approx(math.log1p(60 / 60), rel=1e-6)
    )


def test_the_reexport_ratio_is_capped(world):
    flows = _flows([*_HUB_FLOWS, ("JPN", "IND", WHISKY, 2023, 1e9)])

    table = build_features(_rows([(UK, "JPN", WHISKY, 2023)], flows), flows, world.sources)

    assert table["reexport_ratio"].item() == pytest.approx(math.log1p(REEXPORT_RATIO_CAP), rel=1e-6)
