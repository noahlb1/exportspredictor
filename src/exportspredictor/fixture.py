"""A tiny, deterministic synthetic world for tests, the fixture-mode notebook and local development.

The world has 4 Exporters (including the UK), 4 Markets (one a Re-export Hub), 5 Products
(one a Flagged Product, plus one Excluded Product) and 4 years.
Flows follow a gravity-like rule: Exporter and Importer size, the Exporter's strength in
the Product, and a positive association with a Trade Agreement, plus noise and some zeros.
The UK–Japan Trade Agreement switches on part-way through, so the world contains a
policy change for the model to learn from.

Applied Tariffs come from a TRAINS-shaped 2021 schedule per Market, so 2022–2024 are
carried forward, and the UK–Japan agreement is one that entered force after 2021.

`frozen_v1_tables` builds a synthetic stand-in for v1's frozen predictions, so the v1/v2
comparison (`comparison`) runs end to end on the fixture.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from exportspredictor import tariffs
from exportspredictor.features import Sources, fit_proximity
from exportspredictor.folds import assign_folds

UK = "GBR"

EXPORTERS = {"GBR": 1.0, "DEU": 1.4, "FRA": 1.1, "USA": 1.6}
MARKETS = {
    "JPN": ("Japan", "East Asia & Pacific", 1.2),
    "CAN": ("Canada", "North America", 0.9),
    "IND": ("India", "South Asia", 1.0),
    "HKG": ("Hong Kong", "East Asia & Pacific", 0.8),  # a Re-export Hub
}
PRODUCTS = {
    "711319": "Articles of jewellery and parts thereof, of precious metal other than silver",  # a Flagged Product
    "220830": "Whiskies",
    "300490": "Medicaments, packaged for retail sale, n.e.s.",
    "851712": "Telephones for cellular networks or other wireless networks",
    "870323": "Motor cars, spark-ignition engine of 1,500-3,000 cc",
}
YEARS = [2021, 2022, 2023, 2024]
# Proximity is fitted before the fixture's validation year (2022), as on real data.
PROXIMITY_YEARS = [2021]
# Large, so that leaving it out of the totals would visibly move other Products' RCA.
_HUB = "HKG"
EXCLUDED_PRODUCT = ("270900", "Petroleum oils and oils obtained from bituminous minerals; crude")
_EXCLUDED_SCALE = 10.0

# Trade Agreements in force, as (Exporter, Importer) -> first year in force.
_AGREEMENTS = {
    ("GBR", "JPN"): 2022,  # the policy change inside the fixture window
    ("DEU", "JPN"): 2019,
    ("FRA", "JPN"): 2019,
    ("GBR", "CAN"): 2021,
    ("DEU", "CAN"): 2017,
    ("FRA", "CAN"): 2017,
    ("USA", "CAN"): 1994,
}
# MFN rates (percent) each Market applied in 2021, the last TRAINS year, per Product.
_MFN_2021 = {
    "JPN": {"711319": 5.2, "220830": 2.0, "300490": 1.0, "851712": 4.0, "870323": 6.0, "270900": 0.0},
    "CAN": {"711319": 8.5, "220830": 3.0, "300490": 2.0, "851712": 0.0, "870323": 6.1, "270900": 0.0},
    "IND": {"711319": 20.0, "220830": 150.0, "300490": 10.0, "851712": 20.0, "870323": 60.0, "270900": 5.0},
    "HKG": {"711319": 0.0, "220830": 0.0, "300490": 0.0, "851712": 0.0, "870323": 0.0, "270900": 0.0},  # a free port
}
# 2021 preferences, as TRAINS lists them: Market -> WITS partner codes granted 0%.
# A36 and N04 are EU groups, which expand to the EU's members (here DEU and FRA).
_PREFERENCES_2021 = {"JPN": ["A36"], "CAN": ["826", "N04", "840"]}
_WITS_CODES = {"826": "GBR", "276": "DEU", "250": "FRA", "840": "USA", "392": "JPN", "124": "CAN", "356": "IND"}
_AGREEMENT_EFFECT = 0.6  # log-points; the association the model should recover
_ZERO_SHARE = 0.1

# Trading Bloc memberships in the fixture window (UK left the EU before 2021, joins CPTPP in 2025).
_BLOCS = {"DEU": "eu", "FRA": "eu", "JPN": "cptpp", "CAN": "cptpp"}
_HIGH_INCOME = frozenset({"GBR", "DEU", "FRA", "USA", "JPN", "CAN", "HKG"})
# Population (millions) and distance (km) from the Exporter to each Market, rounded.
_POPULATION = {"GBR": 67, "DEU": 83, "FRA": 68, "USA": 333, "JPN": 125, "CAN": 38, "IND": 1400, "HKG": 7}
_DISTANCE = {
    "GBR": {"JPN": 9560, "CAN": 5370, "IND": 6700, "HKG": 9620},
    "DEU": {"JPN": 9100, "CAN": 6100, "IND": 6200, "HKG": 9150},
    "FRA": {"JPN": 9700, "CAN": 5650, "IND": 6600, "HKG": 9600},
    "USA": {"JPN": 10900, "CAN": 730, "IND": 12000, "HKG": 12900},
}
_CONTIGUOUS = {("USA", "CAN")}
_COMMON_LANGUAGE = {
    ("GBR", "CAN"), ("GBR", "IND"), ("USA", "CAN"), ("USA", "IND"), ("FRA", "CAN"),
    ("GBR", "HKG"), ("USA", "HKG"),
}
_COLONIAL_TIE = {("GBR", "CAN"), ("GBR", "IND"), ("FRA", "CAN"), ("GBR", "HKG")}


@dataclass(frozen=True)
class World:
    """Flows, policy, gravity and lookups shaped like the real data the package builds."""

    flows: pl.DataFrame  # exporter, importer, hs6, year, value_usd (zeros included)
    policy: pl.DataFrame  # exporter, importer, year, trade_agreement, customs_union
    products: pl.DataFrame  # hs6, description, sector
    markets: pl.DataFrame  # iso3, name, region
    blocs: pl.DataFrame  # iso3, year, eu, cptpp, efta (as policy.bloc_membership)
    country_year: pl.DataFrame  # iso3, year, gdp_usd, population (as gravity's country_year)
    pairs: pl.DataFrame  # iso3_o, iso3_d, distance_km, contiguous, ... (as gravity_pairs)
    high_income: frozenset[str]  # ISO3 codes of high-income countries
    tariffs: tariffs.Tariffs  # the Applied Tariff panel, from a 2021 TRAINS-shaped schedule

    @property
    def sources(self) -> Sources:
        """The non-Flow tables the feature table is built from."""
        return Sources(
            agreements=self.policy,
            blocs=self.blocs,
            country_year=self.country_year,
            pairs=self.pairs,
            high_income=self.high_income,
            proximity=fit_proximity(self.flows, PROXIMITY_YEARS),
            tariffs=self.tariffs,
        )


def build_world(seed: int = 0) -> World:
    """Build the fixture world. The same seed always gives the same world."""
    rng = np.random.default_rng(seed)
    strength = {
        (exporter, hs6): rng.normal(0.0, 1.0) for exporter in EXPORTERS for hs6 in PRODUCTS
    }
    product_scale = {hs6: rng.normal(8.0, 0.5) for hs6 in PRODUCTS}

    # The Re-export Hub was added later; its Flows have their own stream, so the others are unchanged.
    hub_rng = np.random.default_rng([seed, 2])
    policy_rows, flow_rows = [], []
    for exporter, exporter_size in EXPORTERS.items():
        for importer, (_, _, importer_size) in MARKETS.items():
            start = _AGREEMENTS.get((exporter, importer))
            draw = hub_rng if importer == _HUB else rng
            for year in YEARS:
                agreement = int(start is not None and year >= start)
                policy_rows.append((exporter, importer, year, agreement))
                for hs6 in PRODUCTS:
                    log_value = (
                        product_scale[hs6]
                        + exporter_size
                        + importer_size
                        + strength[(exporter, hs6)]
                        + _AGREEMENT_EFFECT * agreement
                        + draw.normal(0.0, 0.3)
                    )
                    is_zero = draw.random() < _ZERO_SHARE
                    value = 0.0 if is_zero else round(float(np.exp(log_value)), 2)
                    flow_rows.append((exporter, importer, hs6, year, value))
    flow_rows += _excluded_product_flows(seed)

    flows = pl.DataFrame(
        flow_rows,
        schema={
            "exporter": pl.String,
            "importer": pl.String,
            "hs6": pl.String,
            "year": pl.Int16,
            "value_usd": pl.Float64,
        },
        orient="row",
    )
    policy = pl.DataFrame(
        policy_rows,
        schema={
            "exporter": pl.String,
            "importer": pl.String,
            "year": pl.Int16,
            "trade_agreement": pl.Int8,
        },
        orient="row",
    ).with_columns(customs_union=pl.lit(0, pl.Int8))  # no fixture pair shares a Customs Union
    products = pl.DataFrame(
        {
            "hs6": [*PRODUCTS, EXCLUDED_PRODUCT[0]],
            "description": [*PRODUCTS.values(), EXCLUDED_PRODUCT[1]],
            "sector": [hs6[:2] for hs6 in [*PRODUCTS, EXCLUDED_PRODUCT[0]]],
        }
    )
    markets = pl.DataFrame(
        {
            "iso3": list(MARKETS),
            "name": [name for name, _, _ in MARKETS.values()],
            "region": [region for _, region, _ in MARKETS.values()],
        }
    )
    return World(
        flows=flows,
        policy=policy,
        products=products,
        markets=markets,
        blocs=_blocs(),
        country_year=_country_year(),
        pairs=_pairs(),
        high_income=_HIGH_INCOME,
        tariffs=_tariffs(),
    )


def _tariffs() -> tariffs.Tariffs:
    """The Applied Tariff panel built, as the real one is, from raw TRAINS rows that end in 2021."""
    rows = [
        (market, partner, 2021, hs6, "H4", kind, rate)
        for market, rates in _MFN_2021.items()
        for hs6, mfn in rates.items()
        for partner, kind, rate in [("000", "MFN", mfn)]
        + [(code, "PREF", 0.0) for code in _PREFERENCES_2021.get(market, [])]
    ]
    raw = pl.DataFrame(rows, schema=tariffs.RAW_SCHEMA, orient="row")
    no_concordance = pl.DataFrame(schema={"nomenclature": pl.String, "product": pl.String, "hs6": pl.String})
    return tariffs.panel(raw, no_concordance, _WITS_CODES, YEARS)


def _excluded_product_flows(seed: int) -> list[tuple]:
    """Crude oil Flows from every Exporter to every Market, from their own random stream."""
    rng = np.random.default_rng([seed, 1])
    hs6 = EXCLUDED_PRODUCT[0]
    strength = {exporter: rng.normal(0.0, 1.0) for exporter in EXPORTERS}
    return [
        (exporter, importer, hs6, year, round(float(np.exp(
            _EXCLUDED_SCALE + exporter_size + importer_size + strength[exporter] + rng.normal(0.0, 0.3)
        )), 2))
        for exporter, exporter_size in EXPORTERS.items()
        for importer, (_, _, importer_size) in MARKETS.items()
        for year in YEARS
    ]


def _blocs() -> pl.DataFrame:
    """Trading Bloc membership, shaped like `policy.bloc_membership` (only countries in a bloc)."""
    rows = [
        (iso3, year, int(bloc == "eu"), int(bloc == "cptpp"), 0)
        for iso3, bloc in _BLOCS.items()
        for year in YEARS
    ]
    schema = {"iso3": pl.String, "year": pl.Int16, "eu": pl.Int8, "cptpp": pl.Int8, "efta": pl.Int8}
    return pl.DataFrame(rows, schema=schema, orient="row")


def _country_year() -> pl.DataFrame:
    """GDP grows 3% a year from a level set by each country's size in the Flow rule."""
    size = {**EXPORTERS, **{iso3: s for iso3, (_, _, s) in MARKETS.items()}}
    rows = [
        (iso3, year, float(np.exp(size[iso3]) * 1e12 * 1.03 ** (year - YEARS[0])), _POPULATION[iso3] * 1e6)
        for iso3 in size
        for year in YEARS
    ]
    schema = {"iso3": pl.String, "year": pl.Int16, "gdp_usd": pl.Float64, "population": pl.Float64}
    return pl.DataFrame(rows, schema=schema, orient="row")


def _pairs() -> pl.DataFrame:
    """Time-invariant pair features, shaped like `gravity.gravity_pairs`."""
    rows = [
        (e, i, km, int((e, i) in _CONTIGUOUS), int((e, i) in _COMMON_LANGUAGE), int((e, i) in _COLONIAL_TIE), 0)
        for e, markets in _DISTANCE.items()
        for i, km in markets.items()
    ]
    columns = ["iso3_o", "iso3_d", "distance_km", "contiguous", "common_language", "colonial_tie", "common_colonizer"]
    return pl.DataFrame(rows, schema=columns, orient="row")


# --- A synthetic frozen v1 (spec #39: the v1/v2 comparison's fixture) ----------------------------

# v1 had no Exporter scope: its frozen test rows hold Exporters v2 leaves out. The fixture's
# stand-in is one Exporter outside every fixture scope, with a few rows of its own.
OUT_OF_SCOPE_EXPORTER = "MEX"
_V1_NOISE = 0.8  # log-sd of v1's synthetic errors, so v1 ranks worse than a good model would
_V1_FLOOR_USD = 1_000.0  # v1's synthetic Export Potential where nothing was exported
_V1_ZERO_WEIGHT = 2.0  # v1's synthetic Zero Cell sample keeps every other Zero Cell


def frozen_v1_tables(
    world: World,
    *,
    test_year: int,
    lift_years: Sequence[int],
    lift_held_out: Sequence[str],
    event_directions: Sequence[tuple[str, Sequence[str], Sequence[str]]],
    event_years: Sequence[int],
    seed: int = 0,
) -> dict[str, pl.DataFrame]:
    """Stand-ins for v1's frozen predictions (`freeze.TABLES`), with the frozen schemas.

    v1's Export Potential is Actual Exports (plus a floor) times seeded log-normal noise.
    As v1 did, the tables keep the Excluded Product and Exporters outside v2's scope, and the
    test rows keep every positive Flow and a weighted sample of Zero Cells (every other one,
    weight 2). `event_directions` are (label, exporters, importers).
    """
    rng = np.random.default_rng(seed)

    def predicted(frame: pl.DataFrame, column: str = "export_potential_usd") -> pl.DataFrame:
        noise = rng.lognormal(0.0, _V1_NOISE, frame.height)
        return frame.with_columns((pl.col("value_usd") + _V1_FLOOR_USD).mul(pl.Series(noise)).alias(column))

    keys = ["exporter", "importer", "hs6"]
    flows = world.flows.sort("year", *keys)
    year_flows = flows.filter(pl.col("year") == test_year)
    zeros = year_flows.filter(pl.col("value_usd") == 0).with_row_index("_n").filter(pl.col("_n") % 2 == 0).drop("_n")
    outside = year_flows.filter((pl.col("exporter") == "FRA") & (pl.col("value_usd") > 0)).head(4)
    test_rows = pl.concat(
        [
            year_flows.filter(pl.col("value_usd") > 0).with_columns(weight=pl.lit(1.0)),
            zeros.with_columns(weight=pl.lit(_V1_ZERO_WEIGHT)),
            outside.with_columns(exporter=pl.lit(OUT_OF_SCOPE_EXPORTER), weight=pl.lit(1.0)),
        ]
    ).sort(*keys)
    active = year_flows.filter(pl.col("value_usd") > 0)
    grid = (
        active.select("hs6").unique()
        .join(active.filter(pl.col("importer") != UK).select("importer").unique(), how="cross")
        .join(active.filter(pl.col("exporter") == UK).select("importer", "hs6", "value_usd"), on=["importer", "hs6"], how="left")
        .with_columns(pl.col("value_usd").fill_null(0.0), year=pl.lit(test_year, pl.Int16))
        .sort("hs6", "importer")
    )
    uk_grid = (
        predicted(grid)
        .rename({"importer": "iso3", "value_usd": "actual_exports_usd"})
        .pipe(assign_folds)
        .select("hs6", "iso3", "year", "actual_exports_usd", "fold", "export_potential_usd")
    )
    held = flows.filter(pl.col("exporter").is_in(list(lift_held_out)) & pl.col("year").is_in(list(lift_years)))
    lift = predicted(held.with_columns(weight=pl.lit(1.0))).select(*keys, "year", "value_usd", "weight", "export_potential_usd")
    event = pl.concat(
        [
            flows.filter(
                pl.col("exporter").is_in(list(exporters))
                & pl.col("importer").is_in(list(importers))
                & pl.col("year").is_in(list(event_years))
            ).select(pl.lit(label).alias("direction"), *keys, "year", "value_usd", weight=pl.lit(1.0))
            for label, exporters, importers in event_directions
        ]
    )
    event = predicted(event, "export_potential_off_usd").with_columns(
        export_potential_on_usd=pl.col("export_potential_off_usd") * 1.2
    )
    return {
        "test_rows": predicted(test_rows).select(*keys, "year", "value_usd", "weight", "export_potential_usd"),
        "uk_grid": uk_grid,
        "lift": lift,
        "lift_held_out_exporters": pl.DataFrame({"exporter": sorted(lift_held_out)}, schema={"exporter": pl.String}),
        "event_backtest": event.select(
            "direction", *keys, "year", "value_usd", "weight", "export_potential_on_usd", "export_potential_off_usd"
        ),
    }
