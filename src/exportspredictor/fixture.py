"""A tiny, deterministic synthetic world for tests, the fixture-mode notebook and local development.

The world has 4 Exporters (including the UK), 3 Markets, 5 Products and 4 years.
Flows follow a gravity-like rule: Exporter and Importer size, the Exporter's strength in
the Product, and a positive association with a Trade Agreement, plus noise and some zeros.
The UK–Japan Trade Agreement switches on part-way through, so the world contains a
policy change for the model to learn from.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

UK = "GBR"

EXPORTERS = {"GBR": 1.0, "DEU": 1.4, "FRA": 1.1, "USA": 1.6}
MARKETS = {
    "JPN": ("Japan", "East Asia & Pacific", 1.2),
    "CAN": ("Canada", "North America", 0.9),
    "IND": ("India", "South Asia", 1.0),
}
PRODUCTS = {
    "030420": "Fish fillets, frozen",
    "220830": "Whiskies",
    "300490": "Medicaments, packaged for retail sale, n.e.s.",
    "851712": "Telephones for cellular networks or other wireless networks",
    "870323": "Motor cars, spark-ignition engine of 1,500-3,000 cc",
}
YEARS = [2021, 2022, 2023, 2024]

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


@dataclass(frozen=True)
class World:
    """Flows, policy and lookups shaped like the real data the package builds."""

    flows: pl.DataFrame  # exporter, importer, hs6, year, value_usd (zeros included)
    policy: pl.DataFrame  # exporter, importer, year, trade_agreement
    products: pl.DataFrame  # hs6, description, sector
    markets: pl.DataFrame  # iso3, name, region


def build_world(seed: int = 0) -> World:
    """Build the fixture world. The same seed always gives the same world."""
    rng = np.random.default_rng(seed)
    strength = {
        (exporter, hs6): rng.normal(0.0, 1.0) for exporter in EXPORTERS for hs6 in PRODUCTS
    }
    product_scale = {hs6: rng.normal(8.0, 0.5) for hs6 in PRODUCTS}

    policy_rows, flow_rows = [], []
    for exporter, exporter_size in EXPORTERS.items():
        for importer, (_, _, importer_size) in MARKETS.items():
            start = _AGREEMENTS.get((exporter, importer))
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
                        + rng.normal(0.0, 0.3)
                    )
                    is_zero = rng.random() < _ZERO_SHARE
                    value = 0.0 if is_zero else round(float(np.exp(log_value)), 2)
                    flow_rows.append((exporter, importer, hs6, year, value))

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
    )
    products = pl.DataFrame(
        {
            "hs6": list(PRODUCTS),
            "description": list(PRODUCTS.values()),
            "sector": [hs6[:2] for hs6 in PRODUCTS],
        }
    )
    markets = pl.DataFrame(
        {
            "iso3": list(MARKETS),
            "name": [name for name, _, _ in MARKETS.values()],
            "region": [region for _, region, _ in MARKETS.values()],
        }
    )
    return World(flows=flows, policy=policy, products=products, markets=markets)
