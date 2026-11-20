"""The loop's data (fixture and real, never the test year) and its results file."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from exportspredictor import baci, experiments, market_reach
from exportspredictor.evaluation import naive_from_totals
from exportspredictor.experiments import Data, uk_grid
from exportspredictor.features import (
    build_features,
    feature_set,
    features_dir,
    fit_product_detail,
)
from exportspredictor.paths import data_dir

RESULTS_FILE = "results.json"
MODELS_DIR = "models"


@dataclass(frozen=True)
class LoopData:
    """The experiment rows of every Product, and the validation year's Flows for the naive baseline."""

    data: Data
    flows: Callable[[], pl.DataFrame]  # every Flow of the validation year (exporter, importer, hs6, value_usd)
    zero_seed: int = 0  # the seed the rows' Zero Cells were sampled with
    # World exports of each Product in each training year (hs6, year, world_product_usd), for the
    # naive baseline of the training rows; read only by an arm that starts from it.
    world_products: Callable[[], pl.DataFrame] | None = None
    # Product features finer than Sector, fitted on the training years (`features.fit_product_detail`);
    # read only by an arm whose features include one of them.
    product_detail: Callable[[], pl.DataFrame] | None = None
    # The feature set the rows come from (`features.feature_set`); "" is the original build. A
    # model cached on one set is never reused on another (`Loop`'s cache key).
    feature_set: str = ""
    # The UK grid of the year Market Reach learns from (`market_reach.LAG_YEARS` before the validation
    # year): hs6, importer, value_usd and the Market's demand; read only by an arm scored with Market Reach.
    reach_source: Callable[[], pl.DataFrame] | None = None

    @property
    def validation_year(self) -> int:
        return self.data.validation_year


def naive_from_features(rows: pl.DataFrame, world_products: pl.DataFrame) -> pl.Series:
    """The naive baseline (`evaluation.naive_from_totals`) of untransformed feature rows, from their own features.

    The rows' leave-one-out supply and demand are already features (`log_supply_loo`,
    `log_demand_loo`); `world_products` (hs6, year, world_product_usd) gives the Product's world
    exports, from which the row's own Flow is left out. So the training rows need no Flows.
    """
    world = world_products.select(
        _p=pl.col("hs6").cast(pl.String),
        _y=pl.col("year").cast(pl.Int32),
        _world_usd=pl.col("world_product_usd").cast(pl.Float64),
    )
    keys = rows.select(
        _p=pl.col("hs6").cast(pl.String),
        _y=pl.col("year").cast(pl.Int32),
        _supply_usd=pl.col("log_supply_loo").cast(pl.Float64).exp() - 1,
        _demand_usd=pl.col("log_demand_loo").cast(pl.Float64).exp() - 1,
        _own_usd=pl.col("value_usd").cast(pl.Float64),
    ).join(world, on=["_p", "_y"], how="left", maintain_order="left")
    other_world_usd = pl.col("_world_usd").fill_null(0.0) - pl.col("_own_usd")
    return keys.select(
        naive_usd=naive_from_totals(pl.col("_supply_usd"), pl.col("_demand_usd"), other_world_usd)
    )["naive_usd"]


def world_products(flows: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
    """World exports of each Product in each year (hs6, year, world_product_usd), from non-zero Flows."""
    value = pl.col("value_usd")
    return (
        flows.lazy()
        .filter(value > 0)
        .group_by("hs6", "year")
        .agg(world_product_usd=value.cast(pl.Float64).sum())
        .collect()
    )


def read_results(folder: Path | None) -> dict | None:
    if folder is None or not (folder / RESULTS_FILE).exists():
        return None
    return json.loads((folder / RESULTS_FILE).read_text(encoding="utf-8"))


def save_results(folder: Path, results: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / RESULTS_FILE
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(results, indent=2, default=experiments.json_default), encoding="utf-8")
    partial.replace(path)


def fixture_data(world, *, validation_year: int = experiments.VALIDATION_YEAR) -> LoopData:
    """The fixture world's rows for every Product, built only from Flows up to the validation year."""
    rows = experiments.fixture_data(world, validation_year=validation_year)
    flows = world.flows.filter(pl.col("year") == validation_year)
    world_exports = world_products(world.flows.filter(pl.col("year") < validation_year))
    detail = fit_product_detail(world.flows, sorted(rows.training), uk=baci.UK)
    lag = validation_year - market_reach.LAG_YEARS
    before = world.flows.filter(pl.col("year") <= lag)
    reach = build_features(uk_grid(before.filter(pl.col("year") == lag), baci.UK), before, world.sources)
    return LoopData(
        data=rows, flows=lambda: flows, world_products=lambda: world_exports, product_detail=lambda: detail,
        reach_source=lambda: reach,
    )


def real_data(root: Path | None = None, *, validation_year: int = experiments.VALIDATION_YEAR) -> LoopData:
    """Every Product's rows from `features.build()`'s files, and the validation year's Flows; never a later year.

    The files are those of the feature set `features.FEATURE_SET_ENV` names (the original by default).
    """
    experiments._check_years([validation_year])
    root = Path(root or data_dir())
    name = feature_set()
    folder = features_dir(root, feature_set_name=name)
    settings = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    rows = Data(
        training={y: folder / f"training_{y}.parquet" for y in settings["years"] if y < validation_year},
        validation=folder / f"training_{validation_year}.parquet",
        grid=folder / f"uk_grid_{validation_year}.parquet",
        validation_year=validation_year,
        built_zero_rate=settings["zero_sampling_rate"],
    )

    def flows() -> pl.DataFrame:
        country_map = pl.read_parquet(root / "derived" / "country_map.parquet").select("baci_iso3", "iso3").unique()
        return experiments._year_flows(root, validation_year, dict(country_map.iter_rows()))

    def world_exports() -> pl.DataFrame:
        # Each training year's own file, so no later year is ever opened.
        files = [str(experiments._flows_file(root, y)) for y in sorted(rows.training)]
        return world_products(pl.scan_parquet(files).select("hs6", "year", "value_usd"))

    def product_detail() -> pl.DataFrame:
        # Each training year's own file, so no later year is ever opened.
        years = sorted(rows.training)
        files = [str(experiments._flows_file(root, y)) for y in years]
        flows = pl.scan_parquet(files).select("exporter", "importer", "hs6", "year", "value_usd")
        return fit_product_detail(flows, years, uk=baci.UK)

    def reach_source() -> pl.DataFrame:
        # The same feature set's UK grid of the year before (2021 for 2022), as #124 measured it.
        lag = validation_year - market_reach.LAG_YEARS
        columns = ["hs6", "importer", "value_usd", market_reach.DEMAND]
        return pl.read_parquet(folder / f"uk_grid_{lag}.parquet", columns=columns)

    return LoopData(
        data=rows, flows=flows, zero_seed=settings.get("zero_sampling_seed", 0), world_products=world_exports,
        product_detail=product_detail, feature_set=name, reach_source=reach_source,
    )


def loop_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / f"loop_{baci.BACI_VERSION}"


def load(root: Path | None = None) -> dict:
    """The saved results of the real loop."""
    return json.loads((loop_dir(root) / RESULTS_FILE).read_text(encoding="utf-8"))
