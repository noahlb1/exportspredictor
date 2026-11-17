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
  Exporter, and so do Market import growth and the re-export ratio built from them.
- **Relatedness leaves out the Market**: the Exporter's RCA in neighbouring Products is
  computed without any Flow into the row's Importer, so a UK row never sees its Market.
- **No pair trade in other Products**: no feature uses the Exporter–Importer pair's trade in
  other Products (such as total bilateral trade). It would absorb the policy effect, so a
  Scenario that flips only policy columns would understate the Scenario Effect.
- **Training years only**: `FeatureTransform` (clip ranges and the Sector encoder) is
  fitted on training years, so test-year data cannot reach it. Proximity is fitted on
  PROXIMITY_YEARS, before the validation year.

`build()` (also `uv run python -m exportspredictor.features`) runs this on real data, one
year at a time so that peak memory stays far below 32 GB, and writes under the data root:

    derived/features_<BACI version>/training_<year>.parquet   training rows with features
    derived/features_<BACI version>/uk_grid_<year>.parquet    the full UK grid with features
    derived/features_<BACI version>/proximity.parquet         Product proximity (Relatedness)
    derived/features_<BACI version>/features.json             rate, seed, years, feature list

Every table is keyed by standard ISO3 codes (BACI's `S19` becomes `TWN`), as the gravity
and policy tables are.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl

from exportspredictor import baci, policy
from exportspredictor.downloads import write_parquet
from exportspredictor.exclusions import EXCLUSIONS, drop_excluded
from exportspredictor.paths import data_dir

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
# Relatedness: how close the Product is to the Products the Exporter already exports
# competitively (RCA > 1, leaving out the row's Importer), weighted by proximity.
RELATEDNESS_FEATURES = ["relatedness"]
# Market import growth over GROWTH_YEARS, and the Market's exports of the Product over its
# imports of it (log-capped), both leaving out the row's own Exporter.
MARKET_TREND_FEATURES = ["market_import_growth", "reexport_ratio"]
POLICY_FEATURES = list(policy.POLICY_FEATURES)
FEATURES = [
    *GRAVITY_FEATURES,
    *SUPPLY_FEATURES,
    *DEMAND_FEATURES,
    *PRODUCT_FEATURES,
    *RELATEDNESS_FEATURES,
    *MARKET_TREND_FEATURES,
    *POLICY_FEATURES,
]
# +1 on every policy feature: a Trade Agreement never lowers Export Potential (ADR 0002).
MONOTONE_CONSTRAINTS = {name: 1 for name in POLICY_FEATURES}
# Continuous features clipped to their training-year range at serving time.
CLIPPED_FEATURES = [
    "log_gdp_o",
    "log_gdp_d",
    "log_population_o",
    "log_population_d",
    "log_distance",
    *SUPPLY_FEATURES,
    *DEMAND_FEATURES,
    *RELATEDNESS_FEATURES,
    *MARKET_TREND_FEATURES,
]
# Share of relevant zero cells kept for training on real data. About 95 M relevant zeros a
# year on BACI, so 2% keeps about 1.9 M a year, roughly one for every six non-zero Flows.
ZERO_SAMPLING_RATE = 0.02
_FLOAT_FEATURES = [*GRAVITY_FEATURES, *SUPPLY_FEATURES, *DEMAND_FEATURES]
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


def training_rows(flows: pl.DataFrame | pl.LazyFrame, *, rate: float, seed: int = 0) -> pl.DataFrame:
    """Every non-zero Flow (weight 1) plus a random sample of relevant zero cells (weight 1/rate),
    leaving out Excluded Products.

    A relevant zero cell is an Exporter–Importer–Product–year where the Exporter exports the
    Product to someone and the Importer imports it from someone that year, but not from
    each other. The full grid of cells is never built: most cells are irrelevant, and
    the relevant ones are about 95 M a year on BACI. Each relevant cell is kept with
    probability `rate`, decided by a hash of its keys and `seed`. So the sample is
    deterministic, a cell's fate never depends on other data, and the weights make
    weighted totals unbiased (Horvitz–Thompson).
    """
    if not 0 < rate <= 1:
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    active = flows.lazy().filter(pl.col("value_usd") > 0).select(*KEYS, "value_usd").pipe(drop_excluded)
    cells = (
        active.select("exporter", "hs6", "year")
        .unique()
        .join(active.select("importer", "hs6", "year").unique(), on=["hs6", "year"])
        .filter(pl.col("exporter").cast(pl.String) != pl.col("importer").cast(pl.String))
    )
    if rate < 1:
        threshold = int(rate * 2**64)
        cells = cells.filter(pl.struct(KEYS).hash(seed) < pl.lit(threshold, pl.UInt64))
    zeros = cells.join(active, on=KEYS, how="anti").with_columns(value_usd=pl.lit(0.0), weight=pl.lit(1.0 / rate))
    rows = pl.concat([active.with_columns(weight=pl.lit(1.0)), zeros.select(*KEYS, "value_usd", "weight")])
    return rows.collect(engine="streaming")


def uk_grid(flows: pl.DataFrame, uk: str) -> pl.DataFrame:
    """Every Product × Market for the UK in each year of `flows`, with Actual Exports (0 if none).

    Products are those traded in that year, except Excluded Products, and Markets every
    Importer in that year except the UK itself. Weight is 1: the grid is scored in full,
    never sampled.
    """
    active = flows.filter(pl.col("value_usd") > 0)
    products = active.select("hs6", "year").unique().pipe(drop_excluded)
    markets = active.filter(pl.col("importer") != uk).select("importer", "year").unique()
    uk_flows = active.filter(pl.col("exporter") == uk).select("importer", "hs6", "year", "value_usd")
    return (
        products.join(markets, on="year")
        .with_columns(exporter=pl.lit(uk).cast(flows.schema["exporter"]))
        .join(uk_flows, on=["importer", "hs6", "year"], how="left")
        .with_columns(pl.col("value_usd").fill_null(0.0), weight=pl.lit(1.0))
        .select(*KEYS, "value_usd", "weight")
        .sort(KEYS)
    )


def build_features(rows: pl.DataFrame, flows: pl.DataFrame, sources: Sources) -> pl.DataFrame:
    """Return KEYS, FEATURES, `value_usd` and `weight` for each row.

    `rows` are keyed by KEYS with the row's own Flow `value_usd` (0 for a zero cell);
    `flows` holds every Flow of the same years, from which the supply and demand
    aggregates are computed. Each row's own Flow is subtracted from them (leave-one-out).
    `flows` keeps Excluded Products, so every total counts them (ADR 0005). Market import
    growth also reads the Flows of GROWTH_YEARS earlier: it is missing when `flows` holds
    none for that year. Relatedness needs `sources.proximity`.
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
        demand=value.sum(), demand_high_income=value.filter(pl.col("high_income")).sum()
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
            _d=pl.col("importer").cast(pl.String),
        )
        .join(market_exports, on=["_d", "hs6", "year"], how="left")
        .with_columns(pl.col("market_exports").fill_null(0.0))
        .with_columns(
            log_supply_loo=supply_loo.log1p(),
            rca_loo=pl.when((world_share > 0) & (pl.col("exporter_total") > own)).then(product_share / world_share),
            log_demand_loo=demand_loo.log1p(),
            high_income_share_loo=pl.when(demand_loo > 0).then(
                ((pl.col("demand_high_income") - own_high_income) / demand_loo).clip(0.0, 1.0)
            ),
            sector=pl.col("hs6").cast(pl.String).str.slice(0, 2).cast(pl.Int8),
            market_import_growth=pl.when(has_earlier_year).then(demand_loo.log1p() - demand_then_loo.log1p()),
            reexport_ratio=pl.when(demand_loo > 0)
            .then(reexport.log1p())
            .when(pl.col("market_exports") > 0)
            .then(pl.lit(REEXPORT_RATIO_CAP).log1p()),
            _o=pl.col("exporter").cast(pl.String),
        )
    )
    return (
        table.pipe(_attach_gravity, sources)
        .pipe(_attach_policy, sources)
        .pipe(_attach_relatedness, active, sources.proximity)
        .select(
            *KEYS,
            *(pl.col(f).cast(pl.Float32) for f in _FLOAT_FEATURES),
            *PRODUCT_FEATURES,
            *(pl.col(f).cast(pl.Float32) for f in [*RELATEDNESS_FEATURES, *MARKET_TREND_FEATURES]),
            *POLICY_FEATURES,
            "value_usd",
            "weight",
        )
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


def features_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / f"features_{baci.BACI_VERSION}"


def load_sources(root: Path | None = None) -> tuple[Sources, dict[str, str]]:
    """The derived gravity and policy tables, and the BACI ISO3 -> standard ISO3 mapping."""
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
        country_year=pl.read_parquet(derived / "country_year.parquet"),
        pairs=pl.read_parquet(derived / "gravity_pairs.parquet"),
        high_income=frozenset(to_iso3.get(code, code) for code in high_income),
    )
    return sources, to_iso3


def load_year_flows(year: int, to_iso3: dict[str, str], root: Path | None = None) -> pl.DataFrame:
    """One year of BACI Flows with Exporter and Importer as standard ISO3 strings."""
    return (
        baci.scan_flows(root)
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
    proximity_years: list[int] = PROXIMITY_YEARS,
) -> Path:
    """Write the training rows and the UK grid, with features, for each year; return the folder.

    Each year is built from that year's Flows and, for Market import growth, those of
    GROWTH_YEARS earlier, so no year's file reads a later year. Proximity is fitted once on
    `proximity_years` (those on disk), all before the validation year, and saved as
    `proximity.parquet`. Transforms are not applied here: they are fitted on the training
    years when the model is trained (`FeatureTransform`).
    """
    sources, to_iso3 = load_sources(root)
    out = features_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    fitted_on = [y for y in proximity_years if (baci.flows_dir(root) / f"flows_{y}.parquet").exists()]
    if not fitted_on:
        raise ValueError(f"No BACI Flows on disk for the proximity years {proximity_years}")
    proximity = fit_proximity(baci.scan_flows(root), fitted_on)
    write_parquet(proximity, out / "proximity.parquet", compression="zstd")
    sources = replace(sources, proximity=proximity)
    for year in years:
        flows = load_year_flows(year, to_iso3, root)
        with_earlier = pl.concat([flows, load_year_flows(year - GROWTH_YEARS, to_iso3, root)])
        write_parquet(
            build_features(training_rows(flows, rate=rate, seed=seed), with_earlier, sources),
            out / f"training_{year}.parquet",
            compression="zstd",
        )
        write_parquet(
            build_features(uk_grid(flows, uk), with_earlier, sources),
            out / f"uk_grid_{year}.parquet",
            compression="zstd",
        )
        del flows, with_earlier
    settings = {
        "baci_version": baci.BACI_VERSION,
        "years": list(years),
        "zero_sampling_rate": rate,
        "zero_sampling_seed": seed,
        "features": FEATURES,
        "excluded_products": [exclusion.codes for exclusion in EXCLUSIONS],
        "proximity_years": fitted_on,
    }
    (out / SETTINGS_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
