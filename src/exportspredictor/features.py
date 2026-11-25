"""The feature table, computed identically for training and serving.

Rows are keyed by Exporter, Importer, Product (`hs6`) and year, and carry the Flow
`value_usd` (0 for a zero cell) and a sampling `weight`. Two kinds of rows exist:

- `training_rows`: every non-zero Flow (weight 1) plus a random sample of relevant zero
  cells (weight 1/rate);
- `uk_grid`: every Product × Market for the UK, for scoring.

`build_features` adds the model inputs (`FEATURES`) from the same year's Flows (and, for
Market import growth, those of GROWTH_YEARS earlier) and the gravity, policy and proximity
tables (`Sources`). Leakage rules are built in:

- **Leave-one-out supply** (ADR 0001): the Exporter's exports of the Product leave out the
  row's own Importer, so a UK row never sees its target Market.
- **Leave-one-out demand**: the Importer's imports of the Product leave out the row's own
  Exporter, and so do Market import growth and the re-export ratio built from them, and the
  Exporter count (the other Exporters supplying the Product to the Market): a UK row never
  sees a UK Flow in it.
- **Relatedness leaves out the Market**: the Exporter's RCA in neighbouring Products is
  computed without any Flow into the row's Importer, so a UK row never sees its Market.
- **No pair trade in other Products**: no feature uses the Exporter–Importer pair's trade in
  other Products (such as total bilateral trade). It would absorb the policy effect, so a
  Scenario that flips only policy columns would understate the Scenario Effect.
- **Training years only**: `FeatureTransform` (clip ranges and the Sector encoder) is
  fitted on training years, so test-year data cannot reach it. Proximity is fitted on
  PROXIMITY_YEARS, before the validation year.

`build()` (also `uv run python -m exportspredictor.features --help`) runs this on real data,
one year at a time so that peak memory stays far below 32 GB, and writes one feature set
(`FEATURE_SETS`) under the data root. Every reader opens the set `EXPORTSPREDICTOR_FEATURE_SET`
names, the original by default; a set already built is never overwritten unless asked:

    derived/features_<BACI version>[_<set>]/...

    derived/features_<BACI version>/training_<year>.parquet   training rows with features
    derived/features_<BACI version>/uk_grid_<year>.parquet    the full UK grid with features (Gap Period
                                                              years: the union of their grids)
    derived/features_<BACI version>/proximity.parquet         Product proximity (Relatedness)
    derived/features_<BACI version>/features.json             scope, rates, seed, years, feature lists,
                                                              monotone constraints, gravity fill

Every table is keyed by standard ISO3 codes (BACI's `S19` becomes `TWN`), as the gravity
and policy tables are.
"""

import argparse
import json
import os
from collections.abc import Collection, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl

from exportspredictor import baci, gravity, policy, scope
from exportspredictor.downloads import write_parquet
from exportspredictor.exclusions import EXCLUSIONS, drop_excluded
from exportspredictor.gap_period import gap_period, gap_years
from exportspredictor.paths import data_dir
from exportspredictor.sectors import sector_of
from exportspredictor.tariffs import Tariffs, applied_tariffs
from exportspredictor.tariffs import load as load_tariffs

KEYS = ["exporter", "importer", "hs6", "year"]

GRAVITY_FEATURES = [
    "log_gdp_o",
    "log_gdp_d",
    "log_population_o",
    "log_population_d",
    "log_distance",
    "contiguous",
    "common_language",
    "colonial_tie",
    "common_colonizer",
]
# The Exporter's exports of the Product to all *other* Importers, and its revealed
# comparative advantage in the Product (with the row's own Flow left out).
SUPPLY_FEATURES = ["log_supply_loo", "rca_loo"]
# The Importer's imports of the Product from all *other* Exporters, and the share of
# those imports that come from high-income Exporters.
DEMAND_FEATURES = ["log_demand_loo", "high_income_share_loo"]
PRODUCT_FEATURES = ["sector"]  # HS2 chapter, as a number
# Product features finer than Sector (#74), fitted on training years by `fit_product_detail`
# and joined on the Product; built for the experiment loop's rows only, not by `build_features`.
PRODUCT_DETAIL_FEATURES = ["trading_pair_share", "importing_markets", "hs4"]
# Relatedness: how close the Product is to the Products the Exporter already exports
# competitively (RCA > 1, leaving out the row's Importer), weighted by proximity.
RELATEDNESS_FEATURES = ["relatedness"]
# Market import growth over GROWTH_YEARS, and the Market's exports of the Product over its
# imports of it (log-capped), both leaving out the row's own Exporter.
MARKET_TREND_FEATURES = ["market_import_growth", "reexport_ratio"]
# The Exporter count (spec #113's "supplier count"): ln(1 + the number of Exporters other than
# the row's own with a non-zero Flow of the Product into the Market that year), from every
# Exporter's Flows (ADR 0004). A UK row never counts the UK. How open a Market is to a Product;
# not a policy feature, so it has no monotone constraint.
EXPORTER_COUNT_FEATURES = ["log_exporters_loo"]
# Sibling Demand (spec #155, #164; idea 9 of docs/research/open-review-2022.md), the UK-free demand
# proxy: ln(1 + the Market's imports that year of the Product's HS4 siblings, the other Products of
# its heading, from every Exporter except the UK and the row's own). It reads no UK Flow at all, so a
# UK-dominated Market does not look small, and a UK row never sees its own pair or the UK's other
# trade with the Market; leaving out the row's own Exporter keeps every row free of the pair's trade
# in other Products. Not a policy feature, so it has no monotone constraint. Its "loo" (leave-one-out)
# leaves out more than the pair: the UK and the row's own Exporter, from every sibling Product; the
# name stays as the built `sibling_demand` files store it.
SIBLING_DEMAND_FEATURES = ["log_sibling_demand_loo"]
# Features `build_features` computes that are not yet model inputs: the experiment loop tries each
# as an Arm (spec #113 Phase 2) and a win moves it into ALL_FEATURES. A feature set stores those
# its `FeatureSet.candidates` names.
CANDIDATE_FEATURES = [*EXPORTER_COUNT_FEATURES, *SIBLING_DEMAND_FEATURES]
# The Applied Tariff as ln(1 + t), with t the rate as a fraction (ADR 0006).
TARIFF_FEATURES = ["log_applied_tariff"]
POLICY_FEATURES = [*policy.POLICY_FEATURES, *TARIFF_FEATURES]
# Every feature `build_features` computes, in model column order.
ALL_FEATURES = [
    *GRAVITY_FEATURES,
    *SUPPLY_FEATURES,
    *DEMAND_FEATURES,
    *PRODUCT_FEATURES,
    *RELATEDNESS_FEATURES,
    *MARKET_TREND_FEATURES,
    *POLICY_FEATURES,
]
# Features the ablation (`experiments`, ticket #53) found to hurt: on 2022, the model without
# Market import growth had lower deviance and a higher UK within-Product Spearman. It is still
# computed and stored beside the features, but it is not a model input.
ABLATED_FEATURES = ["market_import_growth"]
# The model inputs, in column order.
FEATURES = [f for f in ALL_FEATURES if f not in ABLATED_FEATURES]
# +1 on every agreement and bloc flag: a Trade Agreement never lowers Export Potential. -1 on
# the Applied Tariff: raising it never raises Export Potential (ADR 0002, ADR 0006).
MONOTONE_CONSTRAINTS = {**{name: 1 for name in policy.POLICY_FEATURES}, **{name: -1 for name in TARIFF_FEATURES}}
# Continuous features clipped to their training-year range at serving time.
CLIPPED_FEATURES = [f for f in [
    "log_gdp_o",
    "log_gdp_d",
    "log_population_o",
    "log_population_d",
    "log_distance",
    *SUPPLY_FEATURES,
    *DEMAND_FEATURES,
    *EXPORTER_COUNT_FEATURES,
    *SIBLING_DEMAND_FEATURES,
    *RELATEDNESS_FEATURES,
    *MARKET_TREND_FEATURES,
    *TARIFF_FEATURES,
] if f in FEATURES]
# Row metadata carried beside the features, never a model input: `tariff_imputed` marks a
# tariff carried forward from 2021 (ADR 0006), and `mfn_tariff` (percent) is the Importer's
# MFN rate, which a Scenario restores when it removes a Trade Agreement.
TARIFF_METADATA = ["tariff_imputed", "mfn_tariff"]
# Share of other Exporters' Zero Cells kept for training on real data (ADR 0004). UK Zero Cells
# are always kept. Settled by the learning curve (`experiments`, ticket #53): on 2022 at a
# non-UK rate of 0.65, 5% gave the lowest deviance (412.1 against 413.6 at 2% and 413.8 at
# 10%) with UK Spearman within 0.003 of 2%, and leaves a higher non-UK rate than 10% does.
ZERO_SAMPLING_RATE = 0.05
_FLOAT_FEATURES = [
    *GRAVITY_FEATURES, *SUPPLY_FEATURES, *DEMAND_FEATURES, *EXPORTER_COUNT_FEATURES, *SIBLING_DEMAND_FEATURES,
]
# Market import growth compares the year with this many years earlier.
GROWTH_YEARS = 3
# The re-export ratio is ln(1 + min(exports / imports, cap)): beyond 10x its imports a
# Market is a producer of the Product, and how much more says nothing extra.
REEXPORT_RATIO_CAP = 10.0
# Proximity is fitted on the years before the validation year (2022), so it never sees
# validation, test or Baseline data.
PROXIMITY_YEARS = list(range(2012, 2022))


@dataclass(frozen=True)
class Sources:
    """The non-Flow tables features are built from, all keyed by standard ISO3 codes."""

    agreements: pl.DataFrame  # exporter, importer, year, trade_agreement, customs_union
    blocs: pl.DataFrame  # iso3, year, eu, cptpp, efta
    country_year: pl.DataFrame  # iso3, year, gdp_usd, population
    pairs: pl.DataFrame  # iso3_o, iso3_d, distance_km, contiguous, common_language, ...
    high_income: frozenset[str]  # ISO3 codes of high-income countries
    # hs6, hs6_neighbour, proximity (from `fit_proximity`); without it Relatedness is missing
    proximity: pl.DataFrame | None = None
    # The Applied Tariff panel (`tariffs.load`); without it the Applied Tariff is missing
    tariffs: Tariffs | None = None


def training_rows(
    flows: pl.DataFrame | pl.LazyFrame,
    *,
    rate: float,
    seed: int = 0,
    uk: str = baci.UK,
    exporters: Collection[str] | None = None,
    keep_excluded: bool = False,
) -> pl.DataFrame:
    """Every non-zero Flow (weight 1), every UK Zero Cell (weight 1) and a random sample of
    other Exporters' Zero Cells (weight 1/rate), leaving out Excluded Products unless `keep_excluded`
    (#171: their rows kept by the same rules, for an Arm that trains on them).

    A Zero Cell is an Exporter–Importer–Product–year where the Exporter exports the
    Product to someone and the Importer imports it from someone that year, but not from
    each other. The full grid of cells is never built: most cells are irrelevant, and
    the relevant ones are about 95 M a year on BACI. Each non-UK Zero Cell is kept with
    probability `rate`, decided by a hash of its keys and `seed`. So the sample is
    deterministic, a cell's fate never depends on other data, and the weights make
    weighted totals unbiased (Horvitz–Thompson). UK Zero Cells are never sampled (ADR 0004).

    `exporters` is the training scope (`scope.exporter_scope`): rows of other Exporters
    are dropped, the UK's never. Every Importer is kept, and `flows` is not filtered, so
    features built from it still count out-of-scope Exporters' Flows.
    """
    if not 0 < rate <= 1:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    flows = flows.lazy()
    products = flows.filter(pl.col("value_usd") > 0)
    if not keep_excluded:
        products = products.pipe(drop_excluded)
    active = products.select(*KEYS, "value_usd")
    is_uk = pl.col("exporter").cast(pl.String) == uk
    if exporters is not None:
        active = active.filter(is_uk | pl.col("exporter").cast(pl.String).is_in(sorted(set(exporters))))
    cells = (
        active.select("exporter", "hs6", "year")
        .unique()
        .join(
            products.select("importer", "hs6", "year").unique(),
            on=["hs6", "year"],
        )
        .filter(pl.col("exporter").cast(pl.String) != pl.col("importer").cast(pl.String))
    )
    if rate < 1:
        cells = cells.filter(is_uk | _zero_cell_kept(rate, seed))
    zeros = cells.join(active, on=KEYS, how="anti").with_columns(
        value_usd=pl.lit(0.0), weight=pl.when(is_uk).then(1.0).otherwise(1.0 / rate)
    )
    rows = pl.concat([active.with_columns(weight=pl.lit(1.0)), zeros.select(*KEYS, "value_usd", "weight")])
    return rows.collect(engine="streaming")


def _zero_cell_kept(rate: float, seed: int) -> pl.Expr:
    """Whether a Zero Cell is in the sample at `rate`: a hash of its keys and `seed` below rate × 2^64.

    A cell kept at one rate is kept at every higher rate, so samples at different rates nest.
    """
    if rate >= 1:
        return pl.lit(True)  # every cell; rate × 2^64 does not fit in a UInt64
    return pl.struct(KEYS).hash(seed) < pl.lit(int(rate * 2**64), pl.UInt64)


def subsample_zeros(
    rows: pl.DataFrame,
    *,
    built_rate: float,
    rate: float,
    seed: int = 0,
    uk: str = baci.UK,
    uk_zeros_sampled: bool = False,
) -> pl.DataFrame:
    """Training rows built at zero rate `built_rate`, cut to the rows `training_rows` would keep at `rate`.

    Flows and UK Zero Cells are kept. Other Exporters' Zero Cells are kept by the same hash
    rule as `training_rows` (same `seed`), and reweighted to 1/`rate`. Samples nest, so this
    gives exactly the rows a build at `rate` would have written, without rebuilding the
    features: the learning curve uses it to vary the zero rate below the built one.

    With `uk_zeros_sampled`, UK Zero Cells are sampled and weighted as other Exporters' are
    (v1's rule, before ADR 0004 kept them all at weight 1), at `rate` even when it is the
    built rate, since the built rows keep every UK Zero Cell.
    """
    if not 0 < rate <= built_rate:
        raise ValueError(f"The zero rate must be in (0, {built_rate}], the rate the rows were built at; got {rate}")
    zero = pl.col("value_usd") == 0
    is_uk = pl.col("exporter").cast(pl.String) == uk
    if uk_zeros_sampled:
        sampled = zero
    elif rate == built_rate:
        return rows
    else:
        sampled = zero & ~is_uk
    return rows.filter(~sampled | _zero_cell_kept(rate, seed)).with_columns(
        weight=pl.when(sampled).then(pl.lit(1.0 / rate)).otherwise(pl.col("weight"))
    )


def uk_grid(flows: pl.DataFrame, uk: str, *, pairs: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every Product × Market for the UK in each year of `flows`, with Actual Exports (0 if none).

    Products are those traded in that year, except Excluded Products, and Markets every
    Importer in that year except the UK itself. Weight is 1: the grid is scored in full,
    never sampled.

    `pairs` (hs6, importer), when given, replaces each year's Products × Markets: the Gap
    Period scores the union of its years' grids (`grid_pairs`) in every one of its years.
    """
    active = flows.filter(pl.col("value_usd") > 0)
    if pairs is None:
        products = active.select("hs6", "year").unique().pipe(drop_excluded)
        markets = active.filter(pl.col("importer") != uk).select("importer", "year").unique()
        keys = products.join(markets, on="year")
    else:
        wanted = pairs.select(
            pl.col("hs6").cast(flows.schema["hs6"]), pl.col("importer").cast(flows.schema["importer"])
        )
        keys = (
            wanted.unique()
            .pipe(drop_excluded)
            .filter(pl.col("importer") != uk)
            .join(flows.select("year").unique(), how="cross")
        )
    uk_flows = active.filter(pl.col("exporter") == uk).select("importer", "hs6", "year", "value_usd")
    return (
        keys
        .with_columns(exporter=pl.lit(uk).cast(flows.schema["exporter"]))
        .join(uk_flows, on=["importer", "hs6", "year"], how="left")
        .with_columns(pl.col("value_usd").fill_null(0.0), weight=pl.lit(1.0))
        .select(*KEYS, "value_usd", "weight")
        .sort(KEYS)
    )


def grid_pairs(grids: Iterable[pl.DataFrame | pl.LazyFrame]) -> pl.DataFrame:
    """The union of the UK Product–Market pairs (hs6, importer) of several years' grids."""
    return (
        pl.concat([grid.lazy().select("hs6", "importer") for grid in grids])
        .unique()
        .sort("hs6", "importer")
        .collect()
    )


def build_features(rows: pl.DataFrame, flows: pl.DataFrame, sources: Sources, *, uk: str = baci.UK) -> pl.DataFrame:
    """Return KEYS, ALL_FEATURES (FEATURES and the ablated ones), CANDIDATE_FEATURES, `value_usd` and `weight` for each row.

    `rows` are keyed by KEYS with the row's own Flow `value_usd` (0 for a zero cell);
    `flows` holds every Flow of the same years, from which the supply and demand
    aggregates are computed. Each row's own Flow is subtracted from them (leave-one-out).
    `flows` keeps Excluded Products, so every total counts them (ADR 0005). Market import
    growth also reads the Flows of GROWTH_YEARS earlier: it is missing when `flows` holds
    none for that year. Relatedness needs `sources.proximity`. Sibling Demand reads no Flow
    of `uk`.
    """
    if "weight" not in rows.columns:
        rows = rows.with_columns(weight=pl.lit(1.0))
    high_income = sorted(sources.high_income)
    active = flows.filter(pl.col("value_usd") > 0).with_columns(
        high_income=pl.col("exporter").cast(pl.String).is_in(high_income)
    )
    value = pl.col("value_usd")
    supply = active.group_by("exporter", "hs6", "year").agg(supply=value.sum())
    exporter_total = active.group_by("exporter", "year").agg(exporter_total=value.sum())
    world_product = active.group_by("hs6", "year").agg(world_product=value.sum())
    world = active.group_by("year").agg(world=value.sum())
    demand = active.group_by("importer", "hs6", "year").agg(
        demand=value.sum(), demand_high_income=value.filter(pl.col("high_income")).sum(), exporter_count=pl.len()
    )

    own = pl.col("value_usd")
    year_type = rows.schema["year"]
    earlier = (pl.col("year") + GROWTH_YEARS).cast(year_type)
    demand_then = demand.select("importer", "hs6", year=earlier, demand_then=pl.col("demand"))
    own_then = active.select("exporter", "importer", "hs6", year=earlier, own_then=value)
    market_exports = supply.select(
        _d=pl.col("exporter").cast(pl.String), hs6="hs6", year="year", market_exports=pl.col("supply")
    )
    flow_years = active["year"].unique().to_list()

    own_high_income = pl.when(pl.col("exporter").cast(pl.String).is_in(high_income)).then(own).otherwise(0.0)
    supply_loo = (pl.col("supply") - own).clip(lower_bound=0.0)
    demand_loo = (pl.col("demand") - own).clip(lower_bound=0.0)
    # Exporters with a non-zero Flow of the Product into the Market, the row's own left out.
    exporters_loo = (pl.col("exporter_count") - (own > 0).cast(pl.UInt32)).clip(lower_bound=0)
    # Revealed comparative advantage with the row's own Flow left out of every total. The
    # Exporter's total keeps its other-Product exports to this Importer: they are a sliver
    # of a country-year total, the same for every Market, so they say nothing about the pair.
    product_share = supply_loo / (pl.col("exporter_total") - own)
    world_share = (pl.col("world_product") - own) / (pl.col("world") - own)
    # The Importer's imports of the Product from everyone but the row's Exporter, now and
    # GROWTH_YEARS earlier; and its own exports of the Product against those imports.
    demand_then_loo = (pl.col("demand_then") - pl.col("own_then")).clip(lower_bound=0.0)
    has_earlier_year = (pl.col("year") - GROWTH_YEARS).is_in(flow_years)
    reexport = (pl.col("market_exports") / demand_loo).clip(upper_bound=REEXPORT_RATIO_CAP)

    table = (
        rows.join(supply, on=["exporter", "hs6", "year"], how="left")
        .join(exporter_total, on=["exporter", "year"], how="left")
        .join(world_product, on=["hs6", "year"], how="left")
        .join(world, on="year", how="left")
        .join(demand, on=["importer", "hs6", "year"], how="left")
        .join(demand_then, on=["importer", "hs6", "year"], how="left")
        .join(own_then, on=["exporter", "importer", "hs6", "year"], how="left")
        .with_columns(
            pl.col(
                "supply", "exporter_total", "world_product", "world", "demand", "demand_high_income", "demand_then",
                "own_then",
            ).fill_null(0.0),
            pl.col("exporter_count").fill_null(0).cast(pl.UInt32),
            _d=pl.col("importer").cast(pl.String),
        )
        .join(market_exports, on=["_d", "hs6", "year"], how="left")
        .with_columns(pl.col("market_exports").fill_null(0.0))
        .with_columns(
            log_supply_loo=supply_loo.log1p(),
            rca_loo=pl.when((world_share > 0) & (pl.col("exporter_total") > own)).then(product_share / world_share),
            log_demand_loo=demand_loo.log1p(),
            log_exporters_loo=exporters_loo.cast(pl.Float64).log1p(),
            high_income_share_loo=pl.when(demand_loo > 0).then(
                ((pl.col("demand_high_income") - own_high_income) / demand_loo).clip(0.0, 1.0)
            ),
            sector=sector_of("hs6").cast(pl.Int8),
            market_import_growth=pl.when(has_earlier_year).then(demand_loo.log1p() - demand_then_loo.log1p()),
            reexport_ratio=pl.when(demand_loo > 0)
            .then(reexport.log1p())
            .when(pl.col("market_exports") > 0)
            .then(pl.lit(REEXPORT_RATIO_CAP).log1p()),
            _o=pl.col("exporter").cast(pl.String),
        )
    )
    return (
        table.pipe(_attach_sibling_demand, active, uk)
        .pipe(_attach_gravity, sources)
        .pipe(_attach_policy, sources)
        .pipe(_attach_relatedness, active, sources.proximity)
        .pipe(_attach_tariffs, sources)
        .select(
            *KEYS,
            *(pl.col(f).cast(pl.Float32) for f in _FLOAT_FEATURES),
            *PRODUCT_FEATURES,
            *(pl.col(f).cast(pl.Float32) for f in [*RELATEDNESS_FEATURES, *MARKET_TREND_FEATURES]),
            *policy.POLICY_FEATURES,
            *(pl.col(f).cast(pl.Float32) for f in TARIFF_FEATURES),
            *TARIFF_METADATA,
            "value_usd",
            "weight",
        )
    )


def fit_product_detail(flows: pl.DataFrame | pl.LazyFrame, training_years: list[int], *, uk: str) -> pl.DataFrame:
    """Product features finer than Sector (#74), one row per Product (`hs6` as text), fitted on `training_years`.

    - `trading_pair_share`: the share of Exporter–Importer pairs trading anything that year
      which trade the Product, averaged over the years;
    - `importing_markets`: the number of Markets importing the Product, averaged over the years;
    - `hs4`: the Product's HS4 heading, as a number.

    Only Flows of `training_years` are read, and only other Exporters' Flows: no UK Flow
    reaches a feature, so no UK pair can see its own Market or its own past trade. A Product
    is the same for every row, so the features say how widely a Product is traded, never
    anything about one pair. A Product with no Flow in those years gets none (missing).
    """
    active = (
        flows.lazy()
        .filter(pl.col("year").is_in(list(training_years)) & (pl.col("value_usd") > 0))
        .filter(pl.col("exporter").cast(pl.String) != uk)
        .select(
            exporter=pl.col("exporter").cast(pl.String),
            importer=pl.col("importer").cast(pl.String),
            hs6=pl.col("hs6").cast(pl.String),
            year=pl.col("year").cast(pl.Int32),
        )
    )
    pairs = active.group_by("year").agg(_pairs=pl.struct("exporter", "importer").n_unique())
    by_year = (
        active.group_by("hs6", "year")
        .agg(_traded=pl.struct("exporter", "importer").n_unique(), _markets=pl.col("importer").n_unique())
        .join(pairs, on="year")
    )
    # A year in which the Product has no Flow counts as 0 pairs and 0 Markets.
    n_years = pl.lit(len(set(training_years)), pl.Float64)
    return (
        by_year.group_by("hs6")
        .agg(
            trading_pair_share=(pl.col("_traded") / pl.col("_pairs")).sum() / n_years,
            importing_markets=pl.col("_markets").sum().cast(pl.Float64) / n_years,
        )
        .with_columns(
            pl.col("trading_pair_share", "importing_markets").cast(pl.Float32),
            hs4=pl.col("hs6").str.slice(0, 4).cast(pl.Int16),
        )
        .select("hs6", *PRODUCT_DETAIL_FEATURES)
        .collect()
    )


@dataclass(frozen=True)
class FeatureTransform:
    """Clip ranges and the Sector encoder, fitted on training years only.

    At serving time continuous features are clipped to the range seen in training
    (as tariffs are in ADR 0002), and a Sector never seen in training becomes missing.
    Trees cannot extrapolate past their training range anyway, so clipping changes no
    prediction; it makes the guard explicit and keeps Scenario inputs inside known ground.
    """

    bounds: dict[str, tuple[float, float]]
    sectors: tuple[int, ...]

    @classmethod
    def fit(cls, features: pl.DataFrame | pl.LazyFrame, training_years: list[int]) -> "FeatureTransform":
        """Fit on the rows of `training_years` only; rows of other years are never read."""
        training = features.lazy().filter(pl.col("year").is_in(list(training_years)))
        stats = training.select(
            *(pl.col(f).min().cast(pl.Float64).alias(f"{f}_min") for f in CLIPPED_FEATURES),
            *(pl.col(f).max().cast(pl.Float64).alias(f"{f}_max") for f in CLIPPED_FEATURES),
        ).collect().row(0, named=True)
        sectors = training.select(pl.col("sector").drop_nulls().unique().sort()).collect()["sector"]
        return cls(
            bounds={f: (stats[f"{f}_min"], stats[f"{f}_max"]) for f in CLIPPED_FEATURES},
            sectors=tuple(int(s) for s in sectors),
        )

    def apply(self, features: pl.DataFrame) -> pl.DataFrame:
        """The same table with features clipped and unseen Sectors set to missing."""
        return features.with_columns(
            *(pl.col(f).clip(lo, hi).cast(features.schema[f]) for f, (lo, hi) in self.bounds.items()),
            sector=pl.when(pl.col("sector").is_in(list(self.sectors))).then(pl.col("sector")),
        )

    def to_dict(self) -> dict:
        """A JSON-ready form for the artefact metadata."""
        return {"bounds": {f: list(b) for f, b in self.bounds.items()}, "sectors": list(self.sectors)}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureTransform":
        return cls(bounds={f: tuple(b) for f, b in data["bounds"].items()}, sectors=tuple(data["sectors"]))


def _attach_sibling_demand(table: pl.DataFrame, active: pl.DataFrame, uk: str) -> pl.DataFrame:
    """Add `log_sibling_demand_loo` (SIBLING_DEMAND_FEATURES) from the non-zero Flows `active`.

    The Market's imports of the row's HS4 heading from Exporters other than the UK, less those of the
    Product itself, less the row's own Exporter's imports of the heading's other Products. No UK Flow
    is read, so the UK's own row and its trade with the Market in other Products never reach it.
    """
    hs4 = pl.col("hs6").cast(pl.String).str.slice(0, 4).alias("_hs4")
    value = pl.col("value_usd")
    non_uk = active.filter(pl.col("exporter").cast(pl.String) != uk).with_columns(hs4)
    heading = non_uk.group_by("importer", "_hs4", "year").agg(_heading=value.sum())
    product = non_uk.group_by("importer", "hs6", "year").agg(_product=value.sum())
    own = non_uk.group_by("exporter", "importer", "_hs4", "year").agg(_own_heading=value.sum())
    own_product = non_uk.group_by("exporter", "importer", "hs6", "year").agg(_own_product=value.sum())
    siblings = (
        pl.col("_heading") - pl.col("_product") - (pl.col("_own_heading") - pl.col("_own_product"))
    ).clip(lower_bound=0.0)
    temporary = ["_hs4", "_heading", "_product", "_own_heading", "_own_product"]
    return (
        table.with_columns(hs4)
        .join(heading, on=["importer", "_hs4", "year"], how="left")
        .join(product, on=["importer", "hs6", "year"], how="left")
        .join(own, on=["exporter", "importer", "_hs4", "year"], how="left")
        .join(own_product, on=["exporter", "importer", "hs6", "year"], how="left")
        .with_columns(pl.col(*temporary[1:]).fill_null(0.0))
        .with_columns(log_sibling_demand_loo=siblings.log1p())
        .drop(temporary)
    )


def _attach_gravity(table: pl.DataFrame, sources: Sources) -> pl.DataFrame:
    year_type = table.schema["year"]
    country_year = sources.country_year.select("iso3", pl.col("year").cast(year_type), "gdp_usd", "population")
    pairs = sources.pairs.select(
        "iso3_o", "iso3_d", "distance_km", "contiguous", "common_language", "colonial_tie", "common_colonizer"
    )
    side = {"o": "_o", "d": "_d"}
    for suffix, key in side.items():
        table = table.join(
            country_year.rename({"iso3": key, "gdp_usd": f"gdp_{suffix}", "population": f"population_{suffix}"}),
            on=[key, "year"],
            how="left",
        )
    return table.join(pairs.rename({"iso3_o": "_o", "iso3_d": "_d"}), on=["_o", "_d"], how="left").with_columns(
        log_gdp_o=pl.col("gdp_o").log(),
        log_gdp_d=pl.col("gdp_d").log(),
        log_population_o=pl.col("population_o").log(),
        log_population_d=pl.col("population_d").log(),
        log_distance=pl.col("distance_km").cast(pl.Float64).log(),
    )


def _attach_policy(table: pl.DataFrame, sources: Sources) -> pl.DataFrame:
    keys = table.select(exporter="_o", importer="_d", year="year").unique()
    agreements = sources.agreements.select("exporter", "importer", "year", *policy.PAIR_FEATURES)
    flags = policy.attach_policy(keys, agreements, sources.blocs)
    return table.join(flags.rename({"exporter": "_o", "importer": "_d"}), on=["_o", "_d", "year"], how="left")


def _attach_tariffs(table: pl.DataFrame, sources: Sources) -> pl.DataFrame:
    """Add `log_applied_tariff` = ln(1 + t/100) and TARIFF_METADATA (`tariffs.applied_tariffs`).

    The same function serves training rows and the UK grid. The panel is first cut to the
    schedules these rows' years read, so a year's rows never join the whole panel.
    """
    if sources.tariffs is None:
        return table.with_columns(
            log_applied_tariff=pl.lit(None, pl.Float64),
            tariff_imputed=pl.lit(None, pl.Boolean),
            mfn_tariff=pl.lit(None, pl.Float64),
        )
    panel = sources.tariffs
    years = table["year"].unique().cast(pl.Int64).to_list()
    schedules = panel.schedules.filter(pl.col("year").cast(pl.Int64).is_in(years))
    used = schedules.select("schedule", "source_year").unique()
    cut = Tariffs(
        schedules=schedules,
        mfn=panel.mfn.join(used, on=["schedule", "source_year"], how="semi"),
        preferences=panel.preferences.join(used, on=["schedule", "source_year"], how="semi"),
    )
    keys = table.with_row_index("_row").select(
        "_row", "year", exporter="_o", importer="_d", hs6=pl.col("hs6").cast(pl.String)
    )
    applied = applied_tariffs(keys, cut, sources.agreements).select(
        "_row",
        "tariff_imputed",
        "mfn_tariff",
        log_applied_tariff=(pl.col("applied_tariff") / 100).log1p(),
    )
    return table.with_row_index("_row").join(applied, on="_row", how="left").drop("_row")


# --- Relatedness ---------------------------------------------------------------------------


def fit_proximity(flows: pl.DataFrame | pl.LazyFrame, years: list[int]) -> pl.DataFrame:
    """Proximity between Products, fitted on the Flows of `years` only.

    An Exporter is competitive in a Product in a year when its RCA is above 1. The
    proximity of Products p and q is the number of Exporter-years competitive in both,
    divided by the larger of the counts competitive in each: the lower of the two
    conditional probabilities (Hidalgo et al., 2007). Pooling the years keeps it stable.
    Returns hs6, hs6_neighbour, proximity for every pair with proximity above 0 (p != q,
    both directions), with `hs6` as a string.
    """
    value = pl.col("value_usd")
    active = (
        flows.lazy()
        .filter(pl.col("year").is_in(list(years)) & (value > 0))
        .select(pl.col("exporter").cast(pl.String), pl.col("hs6").cast(pl.String), "year", "value_usd")
    )
    supply = active.group_by("exporter", "hs6", "year").agg(supply=value.sum())
    competitive = (
        supply.join(
            supply.group_by("exporter", "year").agg(exporter_total=pl.col("supply").sum()), on=["exporter", "year"]
        )
        .join(supply.group_by("hs6", "year").agg(world_product=pl.col("supply").sum()), on=["hs6", "year"])
        .join(supply.group_by("year").agg(world=pl.col("supply").sum()), on="year")
        .filter(pl.col("supply") / pl.col("exporter_total") > pl.col("world_product") / pl.col("world"))
        .select("exporter", "year", "hs6")
        .collect(engine="streaming")
    )
    schema = {"hs6": pl.String, "hs6_neighbour": pl.String, "proximity": pl.Float32}
    if competitive.is_empty():
        return pl.DataFrame(schema=schema)
    products = competitive["hs6"].unique().sort()
    columns = competitive["hs6"].replace_strict(products, pl.Series(range(len(products)), dtype=pl.UInt32))
    rows = competitive.select(pl.struct("exporter", "year").rank("dense") - 1).to_series()
    present = np.zeros((rows.max() + 1, len(products)), dtype=np.float32)
    present[rows.to_numpy(), columns.to_numpy()] = 1.0
    together = present.T @ present
    counts = np.diag(together).copy()
    proximity = together / np.maximum.outer(counts, counts)
    np.fill_diagonal(proximity, 0.0)
    left, right = np.nonzero(proximity)
    names = products.to_numpy()
    return pl.DataFrame(
        {"hs6": names[left], "hs6_neighbour": names[right], "proximity": proximity[left, right]}, schema=schema
    )


def _attach_relatedness(table: pl.DataFrame, active: pl.DataFrame, proximity: pl.DataFrame | None) -> pl.DataFrame:
    """Add `relatedness`: the proximity-weighted share of the Product's neighbours in which
    the Exporter has RCA > 1.

    RCA leaves out every Flow into the row's Importer, from the Exporter's totals and the
    world's alike (ADR 0001), so the target Market's own Flows cannot reach it. A Product
    is not its own neighbour. Missing for a Product without neighbours.
    """
    if proximity is None or proximity.is_empty():
        return table.with_columns(relatedness=pl.lit(None, pl.Float64))
    products = pl.concat([proximity["hs6"], proximity["hs6_neighbour"]]).unique().sort()
    position = pl.Series(range(len(products)), dtype=pl.UInt32)
    weights = np.zeros((len(products), len(products)), dtype=np.float32)
    weights[
        proximity["hs6"].replace_strict(products, position).to_numpy(),
        proximity["hs6_neighbour"].replace_strict(products, position).to_numpy(),
    ] = proximity["proximity"].to_numpy()
    product_index = pl.col("hs6").cast(pl.String).replace_strict(products, position, default=None)
    keyed = table.with_row_index("_row").select("_row", "_o", "_d", "year", _p=product_index)
    flows = active.select(
        "year",
        "value_usd",
        _o=pl.col("exporter").cast(pl.String),
        _d=pl.col("importer").cast(pl.String),
        _p=product_index,
    )
    values = pl.concat(
        _relatedness_in_year(year_rows, flows.filter(pl.col("year") == year), weights)
        for (year,), year_rows in keyed.partition_by("year", as_dict=True).items()
    )
    return table.with_row_index("_row").join(values, on="_row", how="left").drop("_row")


def _relatedness_in_year(rows: pl.DataFrame, flows: pl.DataFrame, weights: np.ndarray) -> pl.DataFrame:
    """`_row`, `relatedness` for rows (`_row`, `_o`, `_d`, `_p`) of one year, from that year's Flows."""
    n_products = weights.shape[0]
    neighbours = weights.sum(axis=1)
    exporters = rows["_o"].unique().sort()
    exporter_position = pl.Series(range(len(exporters)), dtype=pl.UInt32)
    flows = flows.with_columns(_e=pl.col("_o").replace_strict(exporters, exporter_position, default=None))
    known = flows.filter(pl.col("_p").is_not_null())
    own = flows.filter(pl.col("_e").is_not_null())
    own_known = own.filter(pl.col("_p").is_not_null())
    total = pl.col("value_usd").sum()

    def scatter(table: pl.DataFrame | None, *index: str) -> np.ndarray:
        shape = tuple({"_e": len(exporters), "_p": n_products}[i] for i in index)
        out = np.zeros(shape)
        if table is not None:
            out[tuple(table[i].to_numpy() for i in index)] = table["value_usd"].to_numpy()
        return out

    world = float(flows["value_usd"].sum())
    world_product = scatter(known.group_by("_p").agg(total), "_p")
    exporter_total = scatter(own.group_by("_e").agg(total), "_e")
    supply = scatter(own_known.group_by("_e", "_p").agg(total), "_e", "_p")
    market_imports = dict(flows.group_by("_d").agg(total).iter_rows())
    market_product_imports = known.group_by("_d", "_p").agg(total).partition_by("_d", as_dict=True)
    exports_to_market = own.group_by("_d", "_e").agg(total).partition_by("_d", as_dict=True)
    product_exports_to_market = own_known.group_by("_d", "_e", "_p").agg(total).partition_by("_d", as_dict=True)

    parts = []
    for (market,), market_rows in rows.partition_by("_d", as_dict=True).items():
        key = (market,)
        # Every share leaves out all Flows into this Market.
        with np.errstate(invalid="ignore", divide="ignore"):
            exporter_share = (supply - scatter(product_exports_to_market.get(key), "_e", "_p")) / (
                exporter_total - scatter(exports_to_market.get(key), "_e")
            )[:, None]
            world_share = (world_product - scatter(market_product_imports.get(key), "_p")) / (
                world - market_imports.get(market, 0.0)
            )
            competitive = np.nan_to_num(exporter_share, nan=0.0) > world_share
            density = (competitive.astype(np.float32) @ weights) / neighbours
        known_rows = market_rows.filter(pl.col("_p").is_not_null())
        exporter_rows = known_rows["_o"].replace_strict(exporters, exporter_position).to_numpy()
        parts.append(
            pl.DataFrame(
                {"_row": known_rows["_row"], "relatedness": density[exporter_rows, known_rows["_p"].to_numpy()]},
                schema={"_row": pl.UInt32, "relatedness": pl.Float64},
            )
        )
    return pl.concat(parts).fill_nan(None)


# --- Real-data build ----------------------------------------------------------------------

YEARS = list(range(2012, 2025))
SETTINGS_FILE = "features.json"
# Feature builds are versioned as feature sets, each in its own folder, so a rebuild never
# overwrites the files that cached models and served artefacts were built on.


@dataclass(frozen=True)
class FeatureSet:
    """How one feature set is built: the country-year gravity table it reads (under `derived/`), the candidate
    features it stores, the training scope (`scope.exporter_scope`'s `n`), and whether its training rows keep the
    Excluded Products (never scored or ranked: the UK grid leaves them out either way)."""

    country_year: str
    candidates: tuple[str, ...] = tuple(EXPORTER_COUNT_FEATURES)
    scope_size: int = scope.TOP_EXPORTERS
    training_excluded: bool = False


# The sets' names, each defined once here and imported wherever a set is named:
# - "" (`features_<BACI version>`): the original build, on the primary sources only;
ORIGINAL_SET = ""
# - "exporter_count" (#126): with the Exporter count stored and GDP and population filled from secondary
#   sources for the Markets the primary ones miss;
EXPORTER_COUNT_SET = "exporter_count"
# - "all_exporters" (#168): `exporter_count` with every Exporter's rows, not only ADR 0004's top 100. The top
#   100's rows are the same as in `exporter_count`, so the two sets differ only by the other Exporters' rows. Its
#   rows are not sampled at build time: the loop samples other Exporters' rows at a non-UK rate that fits the
#   VRAM budget, and GPU external memory can train on all of them;
ALL_EXPORTERS_SET = "all_exporters"
# - "sibling_demand" (#164): `exporter_count` with Sibling Demand stored as a second candidate feature; every
#   other column is the same;
SIBLING_DEMAND_SET = "sibling_demand"
# - "excluded_products" (#171): `exporter_count` with the Excluded Products' training rows kept; the UK grid still
#   leaves them out. Features are built from all Flows either way (ADR 0005), so every row the two sets share is
#   the same.
EXCLUDED_PRODUCTS_SET = "excluded_products"
FEATURE_SETS: dict[str, FeatureSet] = {
    ORIGINAL_SET: FeatureSet("country_year.parquet"),
    EXPORTER_COUNT_SET: FeatureSet(gravity.COUNTRY_YEAR_FILLED),
    ALL_EXPORTERS_SET: FeatureSet(gravity.COUNTRY_YEAR_FILLED, scope_size=scope.ALL_EXPORTERS),
    SIBLING_DEMAND_SET: FeatureSet(
        gravity.COUNTRY_YEAR_FILLED, candidates=(*EXPORTER_COUNT_FEATURES, *SIBLING_DEMAND_FEATURES)
    ),
    EXCLUDED_PRODUCTS_SET: FeatureSet(gravity.COUNTRY_YEAR_FILLED, training_excluded=True),
}
# The feature set every reader (the loop, the pipeline, evaluation) opens; unset means "".
FEATURE_SET_ENV = "EXPORTSPREDICTOR_FEATURE_SET"


def feature_set(name: str | None = None) -> str:
    """`name`, or the feature set `FEATURE_SET_ENV` names ("" when unset); an unknown set raises."""
    name = os.environ.get(FEATURE_SET_ENV, ORIGINAL_SET) if name is None else name
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {name!r}; known: {sorted(FEATURE_SETS)}")
    return name


def features_dir(root: Path | None = None, *, feature_set_name: str | None = None) -> Path:
    """The folder of a feature set (`feature_set`): `derived/features_<BACI version>[_<set>]`."""
    name = feature_set(feature_set_name)
    suffix = f"_{name}" if name else ""
    return Path(root or data_dir()) / "derived" / f"features_{baci.BACI_VERSION}{suffix}"


def load_sources(root: Path | None = None, *, feature_set_name: str | None = None) -> tuple[Sources, dict[str, str]]:
    """The derived gravity and policy tables, and the BACI ISO3 -> standard ISO3 mapping.

    The country-year table is the one the feature set reads (`FEATURE_SETS`).
    """
    root = Path(root or data_dir())
    derived = root / "derived"
    country_map = pl.read_parquet(derived / "country_map.parquet").select("baci_iso3", "iso3").unique()
    to_iso3 = dict(country_map.iter_rows())
    markets = baci.markets(root)
    high_income = markets.filter(pl.col("income_level") == "High income")["iso3"]
    paths = policy.policy_paths(root)
    sources = Sources(
        agreements=pl.read_parquet(paths["agreements"]),
        blocs=pl.read_parquet(paths["trading_blocs"]),
        country_year=pl.read_parquet(derived / FEATURE_SETS[feature_set(feature_set_name)].country_year),
        pairs=pl.read_parquet(derived / "gravity_pairs.parquet"),
        high_income=frozenset(to_iso3.get(code, code) for code in high_income),
        tariffs=load_tariffs(root),
    )
    return sources, to_iso3


def _scan_years(root: Path | None, years: Iterable[int]) -> pl.LazyFrame:
    """The BACI Flows of `years`, read from those years' own files only, so no other year's file is opened.

    With none of them on disk, an empty frame of the Flows' schema (from the earliest year's file).
    """
    files = [baci.flows_dir(root) / f"flows_{y}.parquet" for y in sorted(set(years))]
    files = [str(f) for f in files if f.exists()]
    if files:
        return pl.scan_parquet(files)
    earliest = min(baci.flows_dir(root).glob("flows_*.parquet"))
    return pl.scan_parquet(earliest).head(0)


def load_year_flows(year: int, to_iso3: dict[str, str], root: Path | None = None) -> pl.DataFrame:
    """One year of BACI Flows with Exporter and Importer as standard ISO3 strings (empty if not on disk)."""
    return (
        _scan_years(root, [year])
        .filter(pl.col("year") == year)
        .with_columns(
            pl.col(side).cast(pl.String).replace_strict(to_iso3, return_dtype=pl.String)
            for side in ("exporter", "importer")
        )
        .collect()
    )


def build(
    root: Path | None = None,
    *,
    years: list[int] = YEARS,
    rate: float = ZERO_SAMPLING_RATE,
    seed: int = 0,
    uk: str = baci.UK,
    scope_size: int | None = None,
    proximity_years: list[int] = PROXIMITY_YEARS,
    baseline_year: int = YEARS[-1],
    feature_set: str | None = None,
    overwrite: bool = False,
    resume: bool = False,
    parts: int = 1,
) -> Path:
    """Write the training rows and the UK grid, with features, for each year; return the folder.

    Each year is built from that year's Flows and, for Market import growth, those of
    GROWTH_YEARS earlier, so no year's file reads a later year's Flows. Proximity is fitted
    once on `proximity_years` (those on disk), all before the validation year, and saved as
    `proximity.parquet`. Transforms are not applied here: they are fitted on the training
    years when the model is trained (`FeatureTransform`). The grids of the Gap Period ending
    with `baseline_year` all hold the union of those years' pairs, so every pair is scored
    in every Gap Period year (`gap_period`).

    The files go to the folder of `feature_set` (`features_dir`; by default the one
    `FEATURE_SET_ENV` names). A folder already built is never overwritten unless `overwrite`.
    `scope_size` is the number of top Exporters whose rows train the model, `scope.ALL_EXPORTERS`
    for every Exporter; by default the set's own (`FeatureSet.scope_size`).
    `resume` keeps the years and the proximity of an unfinished build already on disk. `parts` builds each year's
    training rows' features in that many groups of Markets, to keep peak memory down; every
    feature reads the whole year's Flows, so the rows are the same.
    """
    out = features_dir(root, feature_set_name=feature_set)
    name = out.name.removeprefix(f"features_{baci.BACI_VERSION}").removeprefix("_")
    if (out / SETTINGS_FILE).exists() and not (overwrite or resume):
        raise FileExistsError(f"{out} is already built; pass overwrite=True or build another feature set")
    sources, to_iso3 = load_sources(root, feature_set_name=name)
    built = FEATURE_SETS[name]
    if scope_size is None:
        scope_size = built.scope_size
    exporters = None if scope_size == scope.ALL_EXPORTERS else scope.exporter_scope(
        sources.country_year, uk=uk, n=scope_size
    )
    out.mkdir(parents=True, exist_ok=True)
    fitted_on = [y for y in proximity_years if (baci.flows_dir(root) / f"flows_{y}.parquet").exists()]
    if not fitted_on:
        raise ValueError(f"No BACI Flows on disk for the proximity years {proximity_years}")
    if resume and (out / "proximity.parquet").exists():
        proximity = pl.read_parquet(out / "proximity.parquet")  # fitted on the same years, so the same
    else:
        proximity = fit_proximity(_scan_years(root, fitted_on), fitted_on)
        write_parquet(proximity, out / "proximity.parquet", compression="zstd")
    sources = replace(sources, proximity=proximity)
    unstored = [f for f in CANDIDATE_FEATURES if f not in built.candidates]
    gap =[y for y in gap_years(baseline_year) if y in years]
    pairs = grid_pairs(uk_grid(load_year_flows(y, to_iso3, root), uk) for y in gap) if gap else None
    for year in years:
        written = [out / f"{kind}_{year}.parquet" for kind in ("training", "uk_grid")]
        if resume and all(path.exists() for path in written):
            continue  # each file is renamed into place only when complete (`write_parquet`)
        flows = load_year_flows(year, to_iso3, root)
        with_earlier = pl.concat([flows, load_year_flows(year - GROWTH_YEARS, to_iso3, root)])
        rows = training_rows(
            flows, rate=rate, seed=seed, uk=uk, exporters=exporters, keep_excluded=built.training_excluded
        )
        write_parquet(
            _features_in_parts(rows, with_earlier, sources, parts, uk=uk).drop(unstored), written[0], compression="zstd"
        )
        del rows
        grid = uk_grid(flows, uk, pairs=pairs if year in gap else None)
        write_parquet(
            build_features(grid, with_earlier, sources, uk=uk).drop(unstored), out / f"uk_grid_{year}.parquet",
            compression="zstd",
        )
        del flows, with_earlier, grid
    if exporters is None:  # every Exporter: record those the training rows hold
        exporters = sorted(
            pl.concat([pl.scan_parquet(out / f"training_{year}.parquet").select(
                pl.col("exporter").cast(pl.String).unique()
            ) for year in years]).unique().collect()["exporter"]
        )
    settings = {
        "baci_version": baci.BACI_VERSION,
        "years": list(years),
        "exporter_scope": exporters,
        "exporter_scope_size": "all" if scope_size == scope.ALL_EXPORTERS else scope_size,
        "zero_sampling_rate": rate,
        "uk_zero_sampling_rate": 1.0,
        "zero_sampling_seed": seed,
        "feature_set": name,
        "features": FEATURES,
        "candidate_features": [f for f in CANDIDATE_FEATURES if f not in unstored],
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "gravity": {"country_year": built.country_year, **gravity.fill_record(sources.country_year)},
        "excluded_products": [exclusion.codes for exclusion in EXCLUSIONS],
        "excluded_products_trained": built.training_excluded,
        "proximity_years": fitted_on,
        "gap_years": gap_period(baseline_year),
    }
    (out / SETTINGS_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return out


def _features_in_parts(
    rows: pl.DataFrame, flows: pl.DataFrame, sources: Sources, parts: int, *, uk: str = baci.UK
) -> pl.DataFrame:
    """`build_features` on `rows` split into `parts` groups of Markets, concatenated.

    By Market, because Relatedness is computed one Market at a time: each Market's work is done once.
    """
    if parts <= 1:
        return build_features(rows, flows, sources, uk=uk)
    group = pl.col("importer").cast(pl.String).hash() % parts
    groups = [rows.filter(group == part) for part in range(parts)]
    return pl.concat([build_features(g, flows, sources, uk=uk) for g in groups if not g.is_empty()])


def main(argv: list[str] | None = None) -> None:
    """`python -m exportspredictor.features [--set NAME] [--first-year Y] [--last-year Y] [--overwrite|--resume] [--parts N]`."""
    parser = argparse.ArgumentParser(description="Build the feature files of one feature set.")
    parser.add_argument("--set", default=None, choices=sorted(FEATURE_SETS), help="the feature set (default: $"
                        f"{FEATURE_SET_ENV}, else the original)")
    parser.add_argument("--first-year", type=int, default=YEARS[0])
    parser.add_argument("--last-year", type=int, default=YEARS[-1])
    parser.add_argument("--overwrite", action="store_true", help="overwrite a feature set already built")
    parser.add_argument("--resume", action="store_true", help="keep the years an unfinished build already wrote")
    parser.add_argument("--parts", type=int, default=1, help="build each year's training rows in this many parts")
    args = parser.parse_args(argv)
    years = list(range(args.first_year, args.last_year + 1))
    print(build(years=years, feature_set=args.set, overwrite=args.overwrite, resume=args.resume, parts=args.parts))


if __name__ == "__main__":
    main()
