"""A tiny, deterministic synthetic world for tests, the fixture-mode notebook and local development.

The world has 4 Exporters (including the UK), 4 Markets (one a Re-export Hub), 5 Products
(one a Flagged Product, plus one Excluded Product) and 4 years.
Flows follow a gravity-like rule: Exporter and Importer size, the Exporter's strength in
the Product, and a positive association with a Trade Agreement, plus noise and some zeros.
The UK–Japan Trade Agreement switches on part-way through, so the world contains a
policy change for the model to learn from.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from exportspredictor.features import Sources, fit_proximity

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
    )


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
