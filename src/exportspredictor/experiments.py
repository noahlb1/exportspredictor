"""The feature ablation and the learning curve (ticket #53): features and sampling rates decided on evidence.

Both experiments train v1's configuration on one fold and the Sector subset
(`training.SECTOR_SUBSET`), on the years before the validation year (2022), and measure
each model on the validation year only:

- **validation deviance**: the weighted Tweedie deviance (p = 1.5, as model selection uses)
  of the validation year's training rows, for every other Exporter and the UK pairs of the
  held-out fold;
- **UK Spearman**: the within-Product Spearman rank correlation of the validation year's
  UK grid, for the held-out fold's pairs only (ADR 0001: out-of-fold).

The **ablation** trains the model with every feature (`features.ALL_FEATURES`), then once
without each of v2's new features (`CANDIDATE_FEATURES`). A feature hurts when the model
without it is better on both measures; it is then dropped, except the Applied Tariff, which
Scenarios need (`kept_features`), and `features.FEATURES` leaves it out
(`features.ABLATED_FEATURES`). The **learning curve** trains on the kept features at every
pair of non-UK rate and zero rate in the plan. The zero rate is varied below the rate the
rows were built at with `features.subsample_zeros`, so one feature build serves every point.

**The test year (2023) is never read.** The experiment rows are built (`build`) from the
Flows of the validation year and the years before it only, read file by file, and every
entry point refuses a validation year that is not before `TEST_YEAR` (`TestYearRead`).

The real runs take about an hour on the GPU, so they run from the command line and save
their tables (`save`); the notebook and the pipeline read them back (`load`):

    uv run python -m exportspredictor.experiments build      # CPU, about 15 min
    uv run python -m exportspredictor.experiments ablation   # GPU
    uv run python -m exportspredictor.experiments curve      # GPU, on the kept features

The results are associations measured on one year and one fold: they say which inputs and
rates the model does better with, not why.
"""

import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from exportspredictor import baci, scope
from exportspredictor.downloads import write_parquet
from exportspredictor.evaluation import MIN_RANKED_ROWS, TWEEDIE_POWER, tweedie_deviance
from exportspredictor.features import (
    ALL_FEATURES,
    FEATURES,
    GROWTH_YEARS,
    KEYS,
    MARKET_TREND_FEATURES,
    PROXIMITY_YEARS,
    RELATEDNESS_FEATURES,
    TARIFF_FEATURES,
    FeatureTransform,
    build_features,
    features_dir,
    fit_proximity,
    load_sources,
    subsample_zeros,
    training_rows,
    uk_grid,
)
from exportspredictor.fixture import World
from exportspredictor.folds import N_FOLDS, assign_folds, outside_fold
from exportspredictor.paths import data_dir
from exportspredictor.prediction import export_potential
from exportspredictor.training import (
    SECTOR_SUBSET,
    SELECTION_METRIC,
    Chunk,
    RowCounts,
    fit_non_uk_rate,
    model_params,
    preparer,
    read_chunk,
    row_budget,
    scan_chunks,
    training_matrix,
)

# The test year. Feature and rate choices are made on the validation year, before it.
TEST_YEAR = 2023
VALIDATION_YEAR = 2022
# v2's new features, each dropped in turn by the ablation (ADR 0006).
CANDIDATE_FEATURES = [*RELATEDNESS_FEATURES, *MARKET_TREND_FEATURES, *TARIFF_FEATURES]
# Kept whatever the ablation says: Scenarios change it.
ALWAYS_KEPT = [*TARIFF_FEATURES]
# v1's winning configuration (artefact 20261117T022550Z: depth 8, variance power 1.3), with
# v1's round cap and early stopping on the validation year.
V1_CONFIG = {
    "colsample_bytree": 0.8,
    "eta": 0.1,
    "max_depth": 8,
    "min_child_weight": 1,
    "subsample": 0.8,
    "tweedie_variance_power": 1.3,
}
V1_MAX_ROUNDS = 1500
EARLY_STOPPING_ROUNDS = 30
# The experiment rows are built with this share of other Exporters' Zero Cells, the
# learning curve's highest zero rate; lower rates are subsampled from them.
BUILT_ZERO_RATE = 0.10
# A curve has plateaued when its last step (the two highest non-UK rates) improves the
# validation deviance by less than this share and the UK Spearman by less than this much.
PLATEAU_DEVIANCE_SHARE = 0.01
PLATEAU_SPEARMAN = 0.005
SETTINGS_FILE = "experiments.json"
RESULTS_FILE = "results.json"


class TestYearRead(ValueError):
    """An experiment was asked to read the test year, or a later one."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class Plan:
    """What the experiments train and measure."""

    params: dict
    max_rounds: int
    sectors: tuple[str, ...] | None  # HS2 chapters; None: every Sector
    built_zero_rate: float  # the zero rate the rows were built at
    ablation_rates: tuple[float, float]  # (non-UK rate, zero rate) of the ablation
    non_uk_rates: tuple[float, ...]  # the learning curve's grid
    zero_rates: tuple[float, ...]
    validation_year: int = VALIDATION_YEAR
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS
    fold: int = 0  # the held-out fold of UK pairs
    n_folds: int = N_FOLDS
    device: str = "cuda"
    seed: int = 0
    uk: str = baci.UK

    def with_validation_year(self, year: int) -> "Plan":
        return replace(self, validation_year=year)


# ADR 0004's starting rates for the ablation (the non-UK rate about what the budget allows,
# 5% of Zero Cells); the learning curve's grid is decision #32's.
REAL_PLAN = Plan(
    params=V1_CONFIG,
    max_rounds=V1_MAX_ROUNDS,
    sectors=SECTOR_SUBSET,
    built_zero_rate=BUILT_ZERO_RATE,
    ablation_rates=(0.65, 0.05),
    non_uk_rates=(0.1, 0.3, 0.65),
    zero_rates=(0.02, 0.05, 0.10),
)
# The fixture world has a few hundred rows: every Sector, small trees, high rates, and two
# folds so that the held-out fold holds a few pairs.
FIXTURE_PLAN = Plan(
    params={"max_depth": 2, "eta": 0.3},
    max_rounds=50,
    sectors=None,
    built_zero_rate=1.0,
    ablation_rates=(1.0, 1.0),
    non_uk_rates=(0.3, 0.6, 1.0),
    zero_rates=(0.25, 0.5, 1.0),
    early_stopping_rounds=10,
    n_folds=2,
    device="cpu",
)


@dataclass(frozen=True)
class Data:
    """The experiment rows: training years, the validation year's rows and its UK grid, all with features."""

    training: Mapping[int, Chunk]  # years before the validation year
    validation: Chunk  # the validation year's training rows
    grid: Chunk  # the validation year's UK grid
    validation_year: int
    built_zero_rate: float
    # Rows of the full feature build (every Sector) by kind, to say which rates fit the
    # VRAM budget: uk, non_uk_flows, non_uk_zeros (at `zero_rate`), and the years counted.
    full_rows: dict | None = field(default=None)

    def __post_init__(self):
        _check_years([*self.training, self.validation_year])
        if any(year >= self.validation_year for year in self.training):
            raise ValueError(f"Training years {sorted(self.training)} must come before {self.validation_year}")


def _check_years(years: Sequence[int]) -> None:
    late = sorted(year for year in years if year >= TEST_YEAR)
    if late:
        raise TestYearRead(f"The experiments never read the test year {TEST_YEAR} or later; asked for {late}")


# --- The experiments ------------------------------------------------------------------------


def run(data: Data, plan: Plan) -> dict:
    """The ablation, the kept features and the learning curve on them: every table the notebook shows."""
    results = run_ablation(data, plan)
    return {**results, **run_learning_curve(data, plan, results["kept_features"])}


def run_ablation(data: Data, plan: Plan) -> dict:
    """The model with every feature, then without each candidate feature; and the features kept."""
    _check_plan(data, plan)
    fitter = _Fitter(data, plan)
    non_uk_rate, zero_rate = plan.ablation_rates
    rows = []
    for dropped in [None, *CANDIDATE_FEATURES]:
        kept = [f for f in ALL_FEATURES if f != dropped]
        row = fitter.fit(kept, non_uk_rate=non_uk_rate, zero_rate=zero_rate)
        rows.append({"dropped": dropped, **row})
        _log(f"ablation without {dropped}: {row}")
    full = rows[0]
    for row in rows:
        row["deviance_change"] = row["validation_deviance"] - full["validation_deviance"]
        row["spearman_change"] = _difference(row["uk_spearman"], full["uk_spearman"])
        row["hurts"] = row["dropped"] is not None and _hurts(row, full)
    return {"plan": _plan_dict(plan), "ablation": rows, "kept_features": kept_features(rows)}


def run_learning_curve(data: Data, plan: Plan, features: Sequence[str] = FEATURES) -> dict:
    """The model at every (non-UK rate, zero rate) of the plan, on `features`."""
    _check_plan(data, plan)
    fitter = _Fitter(data, plan)
    budget = _budget_rates(data.full_rows, plan.zero_rates, len(features))
    rows = []
    for zero_rate in plan.zero_rates:
        for non_uk_rate in plan.non_uk_rates:
            row = fitter.fit(features, non_uk_rate=non_uk_rate, zero_rate=zero_rate)
            full_rate = budget.get(zero_rate)
            rows.append(
                {
                    "non_uk_rate": non_uk_rate,
                    "zero_rate": zero_rate,
                    **row,
                    "full_data_non_uk_rate": full_rate,
                    "fits_budget": None if full_rate is None else non_uk_rate <= full_rate,
                }
            )
            _log(f"learning curve at non-UK {non_uk_rate}, zeros {zero_rate}: {row}")
    return {
        "curve_plan": _plan_dict(plan),
        "curve_features": list(features),
        "learning_curve": rows,
        "plateau": plateau(rows),
        "full_rows": data.full_rows,
    }


def kept_features(ablation: Sequence[Mapping]) -> list[str]:
    """ALL_FEATURES without every feature that hurts: the model without it is better on both measures.

    `features.FEATURES` is set from this (`features.ABLATED_FEATURES`).

    `ablation` holds the full model (`dropped` None) and one row per dropped feature, each
    with `validation_deviance` and `uk_spearman`. The Applied Tariff is always kept.
    """
    full = next(row for row in ablation if row["dropped"] is None)
    hurt = {row["dropped"] for row in ablation if row["dropped"] is not None and _hurts(row, full)}
    return [f for f in ALL_FEATURES if f not in hurt or f in ALWAYS_KEPT]


def _hurts(row: Mapping, full: Mapping) -> bool:
    better_deviance = row["validation_deviance"] < full["validation_deviance"]
    spearman = _difference(row["uk_spearman"], full["uk_spearman"])
    return better_deviance and (spearman is None or spearman >= 0)


def plateau(curve: Sequence[Mapping]) -> list[dict]:
    """For each zero rate, the last step of the curve (two highest non-UK rates) and whether it is flat."""
    table = pl.DataFrame(curve)
    out = []
    for zero_rate in table["zero_rate"].unique().sort():
        points = table.filter(pl.col("zero_rate") == zero_rate).sort("non_uk_rate").tail(2).to_dicts()
        if len(points) < 2:
            continue
        before, after = points
        deviance_share = (before["validation_deviance"] - after["validation_deviance"]) / before["validation_deviance"]
        spearman = _difference(after["uk_spearman"], before["uk_spearman"])
        out.append(
            {
                "zero_rate": zero_rate,
                "from_non_uk_rate": before["non_uk_rate"],
                "to_non_uk_rate": after["non_uk_rate"],
                "deviance_improvement_share": deviance_share,
                "spearman_change": spearman,
                "plateaued": deviance_share < PLATEAU_DEVIANCE_SHARE
                and (spearman is None or spearman < PLATEAU_SPEARMAN),
            }
        )
    return out


def _difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _check_plan(data: Data, plan: Plan) -> None:
    _check_years([plan.validation_year])
    if plan.validation_year != data.validation_year:
        raise ValueError(f"The plan validates on {plan.validation_year}, the data on {data.validation_year}")
    if plan.built_zero_rate != data.built_zero_rate:
        raise ValueError(f"The plan expects rows built at zero rate {plan.built_zero_rate}, not {data.built_zero_rate}")


def _budget_rates(full_rows: dict | None, zero_rates: Sequence[float], n_features: int) -> dict:
    """For each zero rate, the largest non-UK rate whose full-data rows fit the VRAM budget."""
    if not full_rows:
        return {}
    rates = {}
    for zero_rate in zero_rates:
        zeros = full_rows["non_uk_zeros"] * zero_rate / full_rows["zero_rate"]
        counts = RowCounts(uk=full_rows["uk"], non_uk=round(full_rows["non_uk_flows"] + zeros))
        rates[zero_rate] = fit_non_uk_rate(counts, row_budget(n_features))
    return rates


class _Fitter:
    """Trains one model per call on the plan's fold and Sectors, and measures it on the validation year."""

    def __init__(self, data: Data, plan: Plan):
        self.data, self.plan = data, plan
        self.in_sectors = _in_sectors(plan.sectors)
        years = sorted(data.training)
        self.chunks = [data.training[y] for y in years]
        # Fitted on the training years only, as the fold models' transform is.
        self.transform = FeatureTransform.fit(scan_chunks(self.chunks).filter(self.in_sectors), years)
        validation = read_chunk(data.validation).filter(self.in_sectors)
        is_uk = pl.col("exporter").cast(pl.String) == plan.uk
        self.validation = self.transform.apply(
            pl.concat([validation.filter(~is_uk), self._held_out(validation.filter(is_uk))])
        )
        grid = read_chunk(data.grid).filter(self.in_sectors & (pl.col("exporter").cast(pl.String) == plan.uk))
        self.grid = self.transform.apply(self._held_out(grid))

    def _held_out(self, uk_rows: pl.DataFrame) -> pl.DataFrame:
        folded = assign_folds(uk_rows.rename({"importer": "iso3"}), self.plan.n_folds).rename({"iso3": "importer"})
        return folded.filter(pl.col("fold") == self.plan.fold).drop("fold")

    def fit(self, features: Sequence[str], *, non_uk_rate: float, zero_rate: float) -> dict:
        """Train the plan's configuration on fresh matrices and measure it."""
        started = time.perf_counter()
        train, valid = self.matrices(features, non_uk_rate=non_uk_rate, zero_rate=zero_rate)
        row = self.measure(
            train, valid, features, self.plan.params,
            max_rounds=self.plan.max_rounds, early_stopping_rounds=self.plan.early_stopping_rounds,
        )
        del train, valid
        return {**row, "seconds": round(time.perf_counter() - started, 1)}

    def matrices(
        self, features: Sequence[str], *, non_uk_rate: float, zero_rate: float
    ) -> tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix]:
        """The training matrix (the plan's Sectors, fold and rates) and the validation matrix.

        Built once, they can train any number of configurations (`measure`), as the
        hyperparameter search does.
        """
        plan = self.plan
        held_out = outside_fold(plan.uk, plan.n_folds, plan.fold)
        in_sectors, built = self.in_sectors, self.data.built_zero_rate

        def keep(table: pl.DataFrame) -> pl.DataFrame:
            rows = subsample_zeros(table.filter(in_sectors), built_rate=built, rate=zero_rate, seed=plan.seed, uk=plan.uk)
            # Row order decides which rows `subsample` draws, so fix it: the same rows, the same model.
            return held_out(rows).sort(KEYS)

        prepare = preparer(uk=plan.uk, transform=self.transform, non_uk_rate=non_uk_rate, seed=plan.seed, keep=keep)
        train = training_matrix(self.chunks, prepare, device=plan.device, features=features)
        valid = training_matrix([self.validation], ref=train, device=plan.device, features=features)
        return train, valid

    def measure(
        self,
        train: xgb.QuantileDMatrix,
        valid: xgb.QuantileDMatrix,
        features: Sequence[str],
        params: Mapping,
        *,
        max_rounds: int,
        early_stopping_rounds: int,
        callbacks: Sequence[xgb.callback.TrainingCallback] = (),
    ) -> dict:
        """Train `params` on the matrices with early stopping on the validation year, and measure the model."""
        plan, started = self.plan, time.perf_counter()
        model = xgb.train(
            model_params(dict(params), device=plan.device, seed=plan.seed, features=features)
            | {"eval_metric": SELECTION_METRIC},
            train,
            num_boost_round=max_rounds,
            evals=[(valid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            callbacks=list(callbacks),
        )
        rounds = model.best_iteration + 1
        scored = self.validation.with_columns(prediction=_predict(model, self.validation, features, rounds, plan.device))
        grid = self.grid.with_columns(prediction=_predict(model, self.grid, features, rounds, plan.device))
        return {
            "features": len(features),
            "training_rows": train.num_row(),
            "rounds": rounds,
            "hit_round_cap": rounds >= max_rounds,
            "validation_year": self.data.validation_year,
            "validation_rows": scored.height,
            "validation_deviance": float(scored.select(tweedie_deviance("prediction", power=TWEEDIE_POWER)).item()),
            "uk_pairs": grid.height,
            "uk_spearman": spearman_within_product(grid, "prediction"),
            "seconds": round(time.perf_counter() - started, 1),
        }


def _predict(model: xgb.Booster, table: pl.DataFrame, features: Sequence[str], rounds: int, device: str) -> pl.Series:
    """The first `rounds` trees' predictions, on the device the model trained on."""
    return export_potential(table, model, features=features, rounds=rounds, device=device)


def spearman_within_product(rows: pl.DataFrame, prediction: str) -> float | None:
    """Mean over Products (with at least MIN_RANKED_ROWS rows) of the Spearman correlation of `prediction` and `value_usd`."""
    within = (
        rows.group_by("hs6")
        .agg(_n=pl.len(), _rho=pl.corr(prediction, "value_usd", method="spearman"))
        .filter((pl.col("_n") >= MIN_RANKED_ROWS) & pl.col("_rho").is_not_nan())
    )
    value = within["_rho"].mean() if within.height else None
    return None if value is None else float(value)


def _in_sectors(sectors: Sequence[str] | None) -> pl.Expr:
    if not sectors:
        return pl.lit(True)
    return pl.col("hs6").cast(pl.String).str.slice(0, 2).is_in(list(sectors))


def _plan_dict(plan: Plan) -> dict:
    return {**asdict(plan), "sectors": list(plan.sectors) if plan.sectors else None}


def _log(message: str) -> None:
    print(message, flush=True)


# --- Data: the fixture world, and the real experiment rows ----------------------------------


def fixture_data(world: World, *, validation_year: int = VALIDATION_YEAR, built_zero_rate: float = 1.0) -> Data:
    """The fixture world's experiment rows, built only from Flows up to the validation year."""
    _check_years([validation_year])
    flows = world.flows.filter(pl.col("year") <= validation_year)
    sources = world.sources
    exporters = scope.exporter_scope(world.country_year, uk=baci.UK)

    def rows(year: int) -> pl.DataFrame:
        year_flows = flows.filter(pl.col("year") == year)
        sampled = training_rows(year_flows, rate=built_zero_rate, uk=baci.UK, exporters=exporters)
        return build_features(sampled, flows, sources)

    years = sorted(y for y in flows["year"].unique().to_list() if y < validation_year)
    return Data(
        training={year: rows(year) for year in years},
        validation=rows(validation_year),
        grid=build_features(uk_grid(flows.filter(pl.col("year") == validation_year), baci.UK), flows, sources),
        validation_year=validation_year,
        built_zero_rate=built_zero_rate,
    )


def experiments_dir(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "derived" / f"experiments_{baci.BACI_VERSION}"


def build(
    root: Path | None = None,
    *,
    validation_year: int = VALIDATION_YEAR,
    years: Sequence[int] | None = None,
    rate: float = BUILT_ZERO_RATE,
    sectors: Sequence[str] | None = SECTOR_SUBSET,
    seed: int = 0,
    uk: str = baci.UK,
    scope_size: int = scope.TOP_EXPORTERS,
) -> Path:
    """Write the experiment rows: each year's training rows of the Sectors, and the validation year's UK grid.

    The rows are those `features.build` writes, cut to `sectors` and built at zero rate
    `rate`; features still come from every Flow. Each year's Flows file (and the one
    GROWTH_YEARS before it) is read on its own, and only for years up to the validation
    year, so the test year is never read. Proximity is fitted on PROXIMITY_YEARS before it.
    """
    root = Path(root or data_dir())
    years = sorted(years or range(min(PROXIMITY_YEARS), validation_year + 1))
    _check_years([*years, validation_year])
    sources, to_iso3 = load_sources(root)
    exporters = scope.exporter_scope(sources.country_year, uk=uk, n=scope_size)
    on_disk = [y for y in range(min(years) - GROWTH_YEARS, validation_year + 1) if _flows_file(root, y).exists()]
    fitted_on = [y for y in PROXIMITY_YEARS if y in on_disk and y < validation_year]
    if not fitted_on:
        raise ValueError(f"No BACI Flows on disk for the proximity years {PROXIMITY_YEARS}")
    proximity = fit_proximity(pl.concat([pl.scan_parquet(_flows_file(root, y)) for y in fitted_on]), fitted_on)
    sources = replace(sources, proximity=proximity)
    in_sectors = _in_sectors(sectors)
    out = experiments_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    for year in years:
        flows = _year_flows(root, year, to_iso3)
        earlier = [_year_flows(root, year - GROWTH_YEARS, to_iso3)] if year - GROWTH_YEARS in on_disk else []
        with_earlier = pl.concat([flows, *earlier])
        rows = training_rows(flows, rate=rate, seed=seed, uk=uk, exporters=exporters).filter(in_sectors)
        write_parquet(build_features(rows, with_earlier, sources), out / f"training_{year}.parquet", compression="zstd")
        if year == validation_year:
            grid = uk_grid(flows, uk).filter(in_sectors)
            write_parquet(build_features(grid, with_earlier, sources), out / f"uk_grid_{year}.parquet", compression="zstd")
        _log(f"experiment rows for {year} written")
    settings = {
        "baci_version": baci.BACI_VERSION,
        "validation_year": validation_year,
        "years": years,
        "sectors": list(sectors) if sectors else None,
        "exporter_scope": exporters,
        "zero_sampling_rate": rate,
        "zero_sampling_seed": seed,
        "features": ALL_FEATURES,
        "proximity_years": fitted_on,
    }
    (out / SETTINGS_FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return out


def _flows_file(root: Path, year: int) -> Path:
    return baci.flows_dir(root) / f"flows_{year}.parquet"


def _year_flows(root: Path, year: int, to_iso3: dict[str, str]) -> pl.DataFrame:
    """One year's Flows, read from that year's file alone, with standard ISO3 codes."""
    return (
        pl.scan_parquet(_flows_file(root, year))
        .with_columns(
            pl.col(side).cast(pl.String).replace_strict(to_iso3, return_dtype=pl.String)
            for side in ("exporter", "importer")
        )
        .collect()
    )


def real_data(root: Path | None = None) -> Data:
    """The experiment rows `build` wrote, and the full feature build's row counts for the VRAM budget."""
    root = Path(root or data_dir())
    folder = experiments_dir(root)
    settings = json.loads((folder / SETTINGS_FILE).read_text(encoding="utf-8"))
    if settings["features"] != ALL_FEATURES:
        raise ValueError(f"The experiment rows in {folder} hold other features; rebuild them (`experiments build`)")
    validation_year = settings["validation_year"]
    _check_years([validation_year])
    return Data(
        training={y: folder / f"training_{y}.parquet" for y in settings["years"] if y < validation_year},
        validation=folder / f"training_{validation_year}.parquet",
        grid=folder / f"uk_grid_{validation_year}.parquet",
        validation_year=validation_year,
        built_zero_rate=settings["zero_sampling_rate"],
        full_rows=_full_rows(root, validation_year),
    )


def _full_rows(root: Path, validation_year: int) -> dict | None:
    """Rows by kind in the full feature build's years before the validation year, or None without it."""
    folder = features_dir(root)
    settings_file = folder / "features.json"
    if not settings_file.exists():
        return None
    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    if settings.get("features") != FEATURES:
        return None
    years = [y for y in settings["years"] if y < validation_year]
    is_uk = pl.col("exporter").cast(pl.String) == baci.UK
    zero = pl.col("value_usd") == 0
    counts = (
        pl.concat([pl.scan_parquet(folder / f"training_{y}.parquet").select("exporter", "value_usd") for y in years])
        .select(uk=is_uk.sum(), non_uk_flows=(~is_uk & ~zero).sum(), non_uk_zeros=(~is_uk & zero).sum())
        .collect()
        .row(0, named=True)
    )
    return {**{k: int(v) for k, v in counts.items()}, "zero_rate": settings["zero_sampling_rate"], "years": years}


# --- Saved results --------------------------------------------------------------------------


def results_path(root: Path | None = None) -> Path:
    return experiments_dir(root) / RESULTS_FILE


def save(results: dict, root: Path | None = None) -> Path:
    path = results_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=json_default), encoding="utf-8")
    return path


def load(root: Path | None = None) -> dict:
    """The saved results of the real runs (`save`)."""
    return json.loads(results_path(root).read_text(encoding="utf-8"))


def json_default(value):
    """`json.dumps(default=...)` for results holding NumPy scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def main(argv: Sequence[str]) -> None:
    command = argv[0] if argv else ""
    if command == "build":
        print(build(), flush=True)
    elif command == "ablation":
        results = run_ablation(real_data(), REAL_PLAN)
        print(save(results), flush=True)
    elif command == "curve":
        results = load()
        results.update(run_learning_curve(real_data(), REAL_PLAN, results["kept_features"]))
        print(save(results), flush=True)
    else:
        raise SystemExit("usage: python -m exportspredictor.experiments build|ablation|curve")


if __name__ == "__main__":
    main(sys.argv[1:])
