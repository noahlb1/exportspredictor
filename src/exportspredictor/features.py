"""The feature table, computed identically for training and serving.

Rows are keyed by Exporter, Importer, Product (`hs6`) and year, and carry the Flow
`value_usd` (0 for a zero cell) and a sampling `weight`. Two kinds of rows exist:

- `training_rows`: every non-zero Flow (weight 1) plus a random sample of relevant zero
  cells (weight 1/rate);
- `uk_grid`: every Product × Market for the UK, for scoring.

`build_features` adds the model inputs (`FEATURES`) from the same year's Flows and the
gravity and policy tables (`Sources`). Leakage rules are built in:

- **Leave-one-out supply** (ADR 0001): the Exporter's exports of the Product leave out the
  row's own Importer, so a UK row never sees its target Market.
- **Leave-one-out demand**: the Importer's imports of the Product leave out the row's own
  Exporter.
- **No pair trade in other Products**: no feature uses the Exporter–Importer pair's trade in
  other Products (such as total bilateral trade). It would absorb the policy effect, so a
  Scenario that flips only policy columns would understate the Scenario Effect.
- **Training years only**: `FeatureTransform` (clip ranges and the Sector encoder) is
  fitted on training years, so test-year data cannot reach it.

`build()` (also `uv run python -m exportspredictor.features`) runs this on real data, one
year at a time so that peak memory stays far below 32 GB, and writes under the data root:

    derived/features_<BACI version>/training_<year>.parquet   training rows with features
    derived/features_<BACI version>/uk_grid_<year>.parquet    the full UK grid with features
    derived/features_<BACI version>/features.json             rate, seed, years, feature list

Every table is keyed by standard ISO3 codes (BACI's `S19` becomes `TWN`), as the gravity
and policy tables are.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from exportspredictor import baci, policy
from exportspredictor.exclusions import EXCLUSIONS, drop_excluded
from exportspredictor.downloads import write_parquet
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
POLICY_FEATURES = list(policy.POLICY_FEATURES)
FEATURES = [*GRAVITY_FEATURES, *SUPPLY_FEATURES, *DEMAND_FEATURES, *PRODUCT_FEATURES, *POLICY_FEATURES]
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
]
# Share of relevant zero cells kept for training on real data. About 95 M relevant zeros a
# year on BACI, so 2% keeps about 1.9 M a year, roughly one for every six non-zero Flows.
ZERO_SAMPLING_RATE = 0.02
_FLOAT_FEATURES = [*GRAVITY_FEATURES, *SUPPLY_FEATURES, *DEMAND_FEATURES]


@dataclass(frozen=True)
class Sources:
    """The non-Flow tables features are built from, all keyed by standard ISO3 codes."""

    agreements: pl.DataFrame  # exporter, importer, year, trade_agreement, customs_union
    blocs: pl.DataFrame  # iso3, year, eu, cptpp, efta
    country_year: pl.DataFrame  # iso3, year, gdp_usd, population
    pairs: pl.DataFrame  # iso3_o, iso3_d, distance_km, contiguous, common_language, ...
    high_income: frozenset[str]  # ISO3 codes of high-income countries


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
    `flows` keeps Excluded Products, so every total counts them (ADR 0005).
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
    own_high_income = pl.when(pl.col("exporter").cast(pl.String).is_in(high_income)).then(own).otherwise(0.0)
    supply_loo = (pl.col("supply") - own).clip(lower_bound=0.0)
    demand_loo = (pl.col("demand") - own).clip(lower_bound=0.0)
    # Revealed comparative advantage with the row's own Flow left out of every total. The
    # Exporter's total keeps its other-Product exports to this Importer: they are a sliver
    # of a country-year total, the same for every Market, so they say nothing about the pair.
    product_share = supply_loo / (pl.col("exporter_total") - own)
    world_share = (pl.col("world_product") - own) / (pl.col("world") - own)

    table = (
        rows.join(supply, on=["exporter", "hs6", "year"], how="left")
        .join(exporter_total, on=["exporter", "year"], how="left")
        .join(world_product, on=["hs6", "year"], how="left")
        .join(world, on="year", how="left")
        .join(demand, on=["importer", "hs6", "year"], how="left")
        .with_columns(
            pl.col("supply", "exporter_total", "world_product", "world", "demand", "demand_high_income").fill_null(0.0)
        )
        .with_columns(
            log_supply_loo=supply_loo.log1p(),
            rca_loo=pl.when((world_share > 0) & (pl.col("exporter_total") > own)).then(product_share / world_share),
            log_demand_loo=demand_loo.log1p(),
            high_income_share_loo=pl.when(demand_loo > 0).then(
                ((pl.col("demand_high_income") - own_high_income) / demand_loo).clip(0.0, 1.0)
            ),
            sector=pl.col("hs6").cast(pl.String).str.slice(0, 2).cast(pl.Int8),
            _o=pl.col("exporter").cast(pl.String),
            _d=pl.col("importer").cast(pl.String),
        )
    )
    return (
        table.pipe(_attach_gravity, sources)
        .pipe(_attach_policy, sources)
        .select(
            *KEYS,
            *(pl.col(f).cast(pl.Float32) for f in _FLOAT_FEATURES),
            *PRODUCT_FEATURES,
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
) -> Path:
    """Write the training rows and the UK grid, with features, for each year; return the folder.

    Each year is built from that year's Flows alone, so a year's file never depends on
    another year's data. Transforms are not applied here: they are fitted on the training
    years when the model is trained (`FeatureTransform`).
    """
    sources, to_iso3 = load_sources(root)
    out = features_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    for year in years:
        flows = load_year_flows(year, to_iso3, root)
        write_parquet(
            build_features(training_rows(flows, rate=rate, seed=seed), flows, sources),
            out / f"training_{year}.parquet",
            compression="zstd",
        )
        write_parquet(
            build_features(uk_grid(flows, uk), flows, sources), out / f"uk_grid_{year}.parquet", compression="zstd"
        )
        del flows
    settings = {
        "baci_version": baci.BACI_VERSION,
        "years": list(years),
        "zero_sampling_rate": rate,
        "zero_sampling_seed": seed,
        "features": FEATURES,
        "excluded_products": [exclusion.codes for exclusion in EXCLUSIONS],
    }
    (out / SETTINGS_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
